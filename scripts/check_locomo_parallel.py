#!/usr/bin/env python3
"""Preflight readiness checks for a parallel LoCoMo run (rychkov stack).

Verifies every knob that the 32-way parallel ingest/QA runs depend on —
against BOTH the config files (deploy/.env, deploy/ogmemory.yaml) and the
LIVE stack (ogmem health, gateway health, live openclaw.json via docker
exec, live container env, embedder latency probe). Config files can be
perfect while the running containers were started with stale values, so
live checks are the point.

Usage (from the oGMemory repo root, or anywhere with --repo pointing at it):

    python3 scripts/check_locomo_parallel.py --phase ingest
    python3 scripts/check_locomo_parallel.py --phase qa \
        --config /home/rychkov/projects/locomo-test/configs/rychkov-full-parallel.toml

Run it twice — once per phase. The only knob that differs between phases is
OGMEM_AUTO_CAPTURE (true for ingest, false for QA; restart ogmem between
phases after flipping it in deploy/.env).

Exit code: 0 = all checks pass (warnings allowed), 1 = at least one FAIL.

Requires: python3.11+ (tomllib), PyYAML, network access to the stack, and
docker group membership for the live openclaw.json / container-env checks
(without docker access those degrade to WARN, not FAIL).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

try:
    import tomllib  # py3.11+
except ImportError:  # pragma: no cover
    tomllib = None

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


# ── reporting ────────────────────────────────────────────────────────

RESULTS: list[tuple[str, str, str]] = []  # (status, name, detail)


def ok(name: str, detail: str = "") -> None:
    RESULTS.append(("OK", name, detail))


def warn(name: str, detail: str) -> None:
    RESULTS.append(("WARN", name, detail))


def fail(name: str, detail: str) -> None:
    RESULTS.append(("FAIL", name, detail))


def check(cond: bool, name: str, ok_detail: str, fail_detail: str,
          warn_instead: bool = False) -> bool:
    if cond:
        ok(name, ok_detail)
        return True
    (warn if warn_instead else fail)(name, fail_detail)
    return False


# ── helpers ──────────────────────────────────────────────────────────

def read_env_file(path: str) -> dict[str, str]:
    env: dict[str, str] = {}
    if not os.path.exists(path):
        return env
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def http_json(url: str, timeout: float = 10.0, bearer: str | None = None):
    req = urllib.request.Request(url)
    if bearer:
        req.add_header("Authorization", f"Bearer {bearer}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def docker_exec(container: str, *args: str, timeout: float = 15.0):
    """docker exec ... Returns (stdout, rc). (None, None) means docker is
    unavailable — live-container checks then degrade to WARN. rc!=0 means the
    command ran but failed (e.g. printenv: variable not set in the container)."""
    try:
        out = subprocess.run(
            ["docker", "exec", container, *args],
            capture_output=True, text=True, timeout=timeout,
        )
        return out.stdout, out.returncode
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        return None, None


_DOCKER_WARNED = False


def docker_exec_warn(container: str, *args: str):
    global _DOCKER_WARNED  # assignment below must not shadow the module flag
    stdout, rc = docker_exec(container, *args)
    if stdout is None and not _DOCKER_WARNED:
        warn("docker access",
             f"cannot `docker exec {container}` (no docker group access / "
             "container down) — live gateway-config and container-env checks "
             "are skipped; file-based checks still ran")
        _DOCKER_WARNED = True
    return stdout, rc


# ── checks ───────────────────────────────────────────────────────────

def check_files(repo: str, cfg_path: str | None) -> tuple[dict, dict, dict]:
    env_path = os.path.join(repo, "deploy", ".env")
    yaml_path = os.path.join(repo, "deploy", "ogmemory.yaml")
    env = read_env_file(env_path)
    check(bool(env), "deploy/.env readable",
          f"{len(env)} vars", f"{env_path} missing or empty")
    if yaml is None:
        fail("PyYAML", "PyYAML is required to parse deploy/ogmemory.yaml "
             "(pip install pyyaml)")
        return env, {}, {}
    try:
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        ok("deploy/ogmemory.yaml parsed")
    except Exception as e:  # noqa: BLE001
        fail("deploy/ogmemory.yaml parsed", f"{yaml_path}: {e}")
        return env, {}, {}

    toml: dict = {}
    if cfg_path:
        if tomllib is None:
            warn("tomllib", "python <3.11: locomo-test config not checked")
        else:
            try:
                with open(cfg_path, "rb") as f:
                    toml = tomllib.load(f)
                ok("locomo-test config parsed", cfg_path)
            except Exception as e:  # noqa: BLE001
                fail("locomo-test config parsed", f"{cfg_path}: {e}")
    return env, cfg, toml


def check_ogmem_health(env: dict) -> None:
    port = env.get("OGMEM_HTTP_PORT", "8090")
    url = f"http://127.0.0.1:{port}/api/v1/health"
    try:
        h = http_json(url, timeout=10)
    except Exception as e:  # noqa: BLE001
        fail("ogmem health", f"{url}: {e}")
        return
    check(h.get("status") == "ok", "ogmem health", url,
          f"status={h.get('status')!r}")
    check(h.get("sql") == "connected", "ogmem sql connected",
          f"dialect={h.get('sql_dialect')}", f"sql={h.get('sql')!r}")
    want = env.get("LLM_MODEL", "")
    check(not want or h.get("llm") == want, "ogmem llm model",
          str(h.get("llm")), f"health reports {h.get('llm')!r}, "
          f".env LLM_MODEL={want!r}")


def check_phase_knobs(env: dict, phase: str, workers: int) -> None:
    """Phase-dependent + parallel-critical .env knobs, file AND live."""
    gt = env.get("OGMEM_GUNICORN_TIMEOUT", "900")
    check(gt.isdigit() and int(gt) >= 3600, "OGMEM_GUNICORN_TIMEOUT",
          f"{gt}s (>=3600)",
          f"{gt}s — under 32-way load long composes exceed the 900s default "
          "and gunicorn kills them mid-request; set 3600 in deploy/.env")

    want_capture = "true" if phase == "ingest" else "false"
    cap = env.get("OGMEM_AUTO_CAPTURE", "false").lower()
    check(cap == want_capture, f"OGMEM_AUTO_CAPTURE ({phase})",
          f"{cap} (correct for {phase})",
          f"deploy/.env says {cap!r} but phase {phase!r} needs {want_capture!r}"
          + (" — ingest stalls silently without after_turn extraction"
             if phase == "ingest" else
             " — eager re-capture of 60-120k-token recall injections blocks "
             "every QA turn in 503-retry loops"))

    recall = env.get("OGMEM_AUTO_RECALL", "false").lower()
    check(recall == "true", "OGMEM_AUTO_RECALL", recall,
          "must be true — otherwise no memory injection during compose")

    for var in ("OGMEM_ACCOUNT_ID", "OGMEM_USER_ID", "OGMEM_AGENT_ID"):
        check(bool(env.get(var)), f"{var} set", env.get(var, ""),
              "the gateway does not forward the /v1/responses 'user' field "
              "to the context-engine plugin — identity must be pinned in env")

    # Live container env: the containers may predate the latest .env edit.
    # NOTE: OGMEM_AUTO_CAPTURE / OGMEM_AUTO_RECALL are PLUGIN settings — they
    # are passed to the OPENCLAW container (context-engine extension), not to
    # ogmem. OGMEM_GUNICORN_TIMEOUT is an ogmem container setting.
    # Compose caveat: "a restart does not update container environment" —
    # after editing deploy/.env you must RECREATE (docker compose up -d),
    # not merely restart, the containers.
    gw_c = env.get("OPENCLAW_CONTAINER_NAME", "openclaw_ogmem")
    live_cap, rc = docker_exec_warn(gw_c, "printenv", "OGMEM_AUTO_CAPTURE")
    if live_cap is not None:
        if rc != 0:
            fail(f"live OGMEM_AUTO_CAPTURE ({phase})",
                 f"not set in container {gw_c} — recreate the openclaw "
                 "container (docker compose up -d) so compose env applies")
        else:
            live_cap = live_cap.strip().lower()
            check(live_cap == want_capture, f"live OGMEM_AUTO_CAPTURE ({phase})",
                  f"container env: {live_cap}",
                  f"running gateway has {live_cap!r} but phase {phase!r} needs "
                  f"{want_capture!r} — flip deploy/.env and RECREATE the "
                  "openclaw container (docker compose up -d; a plain restart "
                  "does not update container env)")
    ogmem_c = env.get("OGMEM_CONTAINER_NAME", "ogmem")
    live_gt, rc = docker_exec_warn(ogmem_c, "printenv", "OGMEM_GUNICORN_TIMEOUT")
    if live_gt is not None and rc == 0:
        live_gt = live_gt.strip()
        check(live_gt.isdigit() and int(live_gt) >= 3600,
              "live OGMEM_GUNICORN_TIMEOUT", f"container env: {live_gt}s",
              f"running ogmem has {live_gt!r} — recreate the ogmem container "
              "(docker compose up -d) after editing deploy/.env")


def check_yaml_knobs(cfg: dict, env: dict, workers: int) -> None:
    if not cfg:
        return
    llm = cfg.get("llm") or {}
    mt = llm.get("max_tokens")
    check(isinstance(mt, int) and mt >= 32768, "llm.max_tokens",
          f"{mt} (>=32768 for reasoning models)",
          f"{mt!r} — deepseek-v4.1-flash burns ~7k tokens on reasoning per "
          "extraction span; with 8192 tool calls end finish_reason=length "
          "and extraction fails hard (LLMCallError)")

    svc = cfg.get("service") or {}
    threads = svc.get("threads")
    check(isinstance(threads, int) and threads >= workers,
          "service.threads", f"{threads} (>= {workers} workers)",
          f"{threads!r} < {workers} parallel workers — gunicorn thread pool "
          "starves the lanes; set service.threads in deploy/ogmemory.yaml")

    emb = cfg.get("embedding") or {}
    base = str(emb.get("base_url", ""))
    check(base.rstrip("/").endswith("/v1"), "embedding.base_url has /v1",
          base, f"{base!r} — the openai SDK appends /embeddings; a bare host "
          "gives Flask 404s")

    dim = (cfg.get("vector_db") or {}).get("dimension")
    check(isinstance(dim, int) and dim > 0, "vector_db.dimension", str(dim),
          "must equal the embedder's output dimension (Qwen3-Embedding-0.6B "
          "→ 1024)")

    # Embedder live probe: health + a real embeddings call (latency + dims).
    if base:
        health_url = base.rstrip("/").rsplit("/v1", 1)[0] + "/health"
        try:
            h = http_json(health_url, timeout=10)
            model = str(h.get("model", ""))
            want_model = str(emb.get("model", ""))
            check(not want_model or model == want_model,
                  "embedder health", f"{health_url} model={model}",
                  f"embedder reports model {model!r}, yaml wants "
                  f"{want_model!r}")
        except Exception as e:  # noqa: BLE001
            fail("embedder health", f"{health_url}: {e}")
            return
        key = env.get("M3_EMBED_API_KEY") or str(emb.get("api_key", ""))
        payload = json.dumps({
            "model": emb.get("model"),
            "input": ["preflight probe"],
        }).encode()
        req = urllib.request.Request(base.rstrip("/") + "/embeddings",
                                     data=payload,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {key}"})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = json.loads(r.read())
            dt = time.monotonic() - t0
            got_dim = len(body["data"][0]["embedding"])
            check(got_dim == dim, "embedder dimension",
                  f"{got_dim} == vector_db.dimension ({dim})",
                  f"embedder returns {got_dim} dims, vector_db.dimension="
                  f"{dim} — every embed/index write will fail or corrupt")
            check(dt < 10.0, "embedder latency", f"{dt:.1f}s for 1 embed",
                  f"{dt:.1f}s for a single embed — the shared :8000 service "
                  "serializes requests (8 concurrent → 10-16s each) and "
                  "causes APITimeoutError retry storms; use a dedicated "
                  "instance", warn_instead=True)
        except Exception as e:  # noqa: BLE001
            fail("embedder probe", f"POST {base}/embeddings: {e}")


def check_gateway(env: dict, workers: int) -> None:
    port = env.get("GATEWAY_PORT", "18789")
    try:
        h = http_json(f"http://127.0.0.1:{port}/health", timeout=10)
        check(h.get("ok") is True or h.get("status") == "live",
              "gateway health", f":{port}", f"unexpected: {h}")
    except Exception as e:  # noqa: BLE001
        fail("gateway health", f"http://127.0.0.1:{port}/health: {e}")

    # Live openclaw.json — the ONLY source of truth for maxConcurrent.
    # There is no env knob for it: the entrypoint regenerates openclaw.json
    # from env on every start, so a manual edit can be silently lost on
    # restart/reset. That is exactly what this check exists to catch.
    raw, rc = docker_exec_warn(
        env.get("OPENCLAW_CONTAINER_NAME", "openclaw_ogmem"),
        "cat", "/home/node/.openclaw/openclaw.json")
    if raw is None:
        return
    try:
        oc = json.loads(raw)
    except json.JSONDecodeError as e:
        fail("gateway openclaw.json", f"unparseable: {e}")
        return
    defaults = (oc.get("agents") or {}).get("defaults") or {}
    mc = defaults.get("maxConcurrent")
    check(isinstance(mc, int) and mc >= workers,
          "gateway agents.defaults.maxConcurrent",
          f"{mc} (>= {workers} workers)",
          f"{mc!r} — the openclaw default (4) starves parallel lanes; there "
          "is NO env knob: edit agents.defaults.maxConcurrent in the live "
          "openclaw.json (deploy/_rychkov/openclaw-home/openclaw.json) and "
          "restart the gateway. NOTE: the container entrypoint regenerates "
          "the file from env on start and can drop manual edits — re-run "
          "this preflight after every gateway restart")
    primary = str(((defaults.get("model") or {}).get("primary")) or "")
    want = env.get("LLM_MODEL", "")
    check(not want or primary.endswith("/" + want) or primary == want,
          "gateway primary model", primary or "(unset)",
          f"gateway primary={primary!r} but .env LLM_MODEL={want!r}")


def check_locomo_config(toml: dict, env: dict, workers: int) -> None:
    if not toml:
        return
    g = toml.get("general") or {}
    par = g.get("parallel")
    check(isinstance(par, int) and par >= 1, "config general.parallel",
          f"{par}", f"{par!r} — set parallel = N in the locomo-test config "
          "(see configs/rychkov-full-parallel.toml)")
    if isinstance(par, int) and par >= 1:
        check(workers >= par, "workers vs config",
              f"config parallel={par} covered by preflight --workers={workers}",
              f"config parallel={par} exceeds the stack knobs verified for "
              f"{workers} workers — re-run preflight with --workers {par}")
    user = str(g.get("user", ""))
    check(bool(user), "config general.user set", user,
          "gateway session user must be set (deterministic AND disposable "
          "session keys)")
    ogmem_user = env.get("OGMEM_USER_ID", "")
    if user and ogmem_user and user != ogmem_user:
        warn("config user vs OGMEM_USER_ID",
             f"config user={user!r} != .env OGMEM_USER_ID={ogmem_user!r} — "
             "intentional for scoped runs (fresh namespace per run), but "
             "confirm ingest and QA share one identity")


# ── main ─────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", required=True, choices=["ingest", "qa"],
                    help="which run phase (OGMEM_AUTO_CAPTURE differs)")
    ap.add_argument("--repo", default=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))),
        help="oGMemory repo root (default: parent of this script)")
    ap.add_argument("--config",
                    help="locomo-test run config .toml (e.g. "
                    "configs/rychkov-full-parallel.toml)")
    ap.add_argument("--workers", type=int, default=32,
                    help="expected parallelism (default 32)")
    args = ap.parse_args()

    print(f"Parallel LoCoMo preflight — phase={args.phase}, "
          f"workers={args.workers}\n")

    env, cfg, toml = check_files(args.repo, args.config)
    check_ogmem_health(env)
    check_phase_knobs(env, args.phase, args.workers)
    check_yaml_knobs(cfg, env, args.workers)
    check_gateway(env, args.workers)
    check_locomo_config(toml, env, args.workers)

    width = max(len(n) for _, n, _ in RESULTS) + 2
    fails = warns = 0
    for status, name, detail in RESULTS:
        mark = {"OK": "✓", "WARN": "!", "FAIL": "✗"}[status]
        print(f" [{mark}] {status:<4} {name.ljust(width)} {detail}")
        fails += status == "FAIL"
        warns += status == "WARN"
    print(f"\n{len(RESULTS)} checks: "
          f"{len(RESULTS) - fails - warns} ok, {warns} warnings, {fails} failures")
    if fails:
        print("FIX the failures above before starting the run — see "
              "docs/locomo_parallel_runbook.md for every knob's why/how.")
        return 1
    print(f"Stack is ready for the {args.phase} phase.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
