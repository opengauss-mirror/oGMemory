# Compose face detection for M3-Bench

The existing `scripts/cv_service.py` remains an independent HTTP service.
Compose now builds and manages it: restart policy, health check, logs, and
an optional `with-cv` profile. It is not a dependency of ordinary memory runs.

## Configure the client in ogmemory.yaml

M3 reads the top-level `cv_service` block from the file selected by `--config`
(or `OGMEM_CONFIG`, falling back to `deploy/ogmemory.yaml` for direct CLI use):

```yaml
cv_service:
  base_url: "http://127.0.0.1:8081"
  timeout_seconds: 30
  # Optional Bearer token for an authenticated external proxy:
  # auth_token: "${OGMEM_CV_AUTH_TOKEN}"
```

This default connects to the bundled Compose service's published host port.
It does **not** start that service automatically; use the commands below.
The bundled service remains CPU InsightFace. For an already deployed GPU face
service, change `base_url`, for example to `http://gpu-server:8081`, and optionally
increase `timeout_seconds`. A URL change selects the remote service; its deployment
controls GPU usage. In a bridge-networked client sharing CV's network, use
`http://cv:8081` rather than loopback.

External services must support `POST /detect/faces` with raw image bytes
(`application/octet-stream`) and the `threshold` query parameter, returning
`{"detections": [...]}` with `label`, `confidence`, normalized xyxy `bbox`,
and face `embedding` for identity clustering. The URL is the service root,
not the detection endpoint or an OpenAI `/v1` endpoint.

Priority is explicit CLI/pipeline options > exported `OGMEM_CV_SERVICE_URL`
and `OGMEM_CV_AUTH_TOKEN` > YAML > defaults. Timeout is explicit option > YAML
> 30 seconds. Existing `--cv-service-url`, `--cv-timeout`, and standalone
`m3.detect_faces --cv-auth-token` options remain available. Pipeline equivalents
are `detect_faces.params.cv_service_url`, `cv_timeout`, and `cv_auth_token`.
Prefer YAML/environment for tokens: command-line arguments can appear in logs.

The benchmark wrapper `scripts/run_m3_random10.sh` reads `CONFIG` to choose its
YAML (its historical default is `data/ogmem_test_lyj.resolved.yaml`). To use the
deployment YAML, set `CONFIG=deploy/ogmemory.yaml`. The wrapper no longer inserts
a localhost URL: it only overrides YAML if `CV_SERVICE_URL` is explicitly set.
Unset both `CV_SERVICE_URL` and `OGMEM_CV_SERVICE_URL` to use YAML unconditionally.
`deploy/.env` is not automatically loaded by host-side M3 processes; `${NAME}`
references require the corresponding environment variable to be exported.

Changing the endpoint does not invalidate existing face extraction caches.
To evaluate a different face model, rerun detection (`force=true`) and dependent
stages in an appropriately isolated experiment; a disabled `detect_faces` stage
will never call the new service.

## Start just CV

From the repository root:

```bash
docker compose -f deploy/docker-compose.cv.yml up -d --build --wait cv
curl --fail http://127.0.0.1:8081/health
export OGMEM_CV_SERVICE_URL=http://127.0.0.1:8081
# Run the M3 pipeline/benchmark from this same shell.
```

No database or LLM credentials are needed for this CV-only command.
Stop an existing manually launched CV service first if it occupies port 8081,
or choose a different host port:

```bash
OGMEM_CV_PORT=18081 docker compose -f deploy/docker-compose.cv.yml up -d --build --wait cv
export OGMEM_CV_SERVICE_URL=http://127.0.0.1:18081
```

`OGMEM_CV_PORT` changes the published port, not the container's internal 8081.
Keep using the chosen port for subsequent Compose commands, or put it in
`deploy/.env`. That file configures Compose; it does **not** export variables
to a benchmark process running on the host.

## Include CV in the existing stack

With the existing stack's `deploy/.env` and `ogmemory.yaml` configured:

```bash
docker compose -f deploy/docker-compose.yml --profile with-db --profile with-cv up -d --build --wait
```

Or start just the database and CV for a host-side M3 run:

```bash
docker compose -f deploy/docker-compose.yml --profile with-m3 --profile with-cv up -d --build --wait
```

The main Compose file still interpolates its required DB/LLM variables,
including services outside the selected profiles. Use the independent
`docker-compose.cv.yml` entry point when you only need CV and do not have
those variables configured. Both files reuse the same service definition
and project name; don't launch duplicate CV instances from them.

Host processes and the existing host-networked containers reach CV through
`http://127.0.0.1:8081` (or the chosen host port). Another bridge-networked
container must join CV's network and use `http://cv:8081`; its own
`127.0.0.1` does not refer to the host. The main stack attaches CV to the
existing `ant-trail-net`; the CV-only file uses a Compose default network.

If the main stack fails creating `<project>_default` with “could not find
an available, non-overlapping IPv4 address pool”, check that its `cv`
service explicitly lists `networks: [ant-trail-net]`. Earlier CV wiring
unnecessarily requested a second bridge. Update the main Compose file
and rerun the same `up -d` command, keeping the same project name.
No `down -v`, network pruning, or Docker restart is needed for this fix.
If it instead fails creating `ant-trail-net`, that is a separate subnet
conflict: inspect existing Docker networks and host routes before selecting
a non-overlapping `NETWORK_SUBNET`. The standalone CV-only file still
requires Docker to have an available default address pool.

## Logs, stop, and request smoke test

```bash
docker compose -f deploy/docker-compose.cv.yml logs --tail=100 cv
docker compose -f deploy/docker-compose.cv.yml ps cv
curl --fail -H 'Content-Type: application/octet-stream' \
  --data-binary @/path/to/test-frame.jpg \
  'http://127.0.0.1:8081/detect/faces?threshold=0.5&max_items=20'
docker compose -f deploy/docker-compose.cv.yml stop cv
```

Replace the image path before running the request. An empty `detections`
array can be a valid response; it does not establish detection accuracy.
Use service-scoped `stop cv` rather than bringing down the whole shared
project. Rebuild and repeat the `up` command after changing the service.

## Deliberate limits

- The supplied image now runs **CPU InsightFace, buffalo_l, 640×640 detection**,
  using the same backend implementation as standalone `--backend insightface`.
  It returns normalized face embeddings (512 dimensions for buffalo_l), not
  just Haar boxes. The standalone script's lightweight default remains Haar.
  Missing dependencies, weights, or recognition models fail startup instead
  of silently falling back to Haar. No NPU/GPU is used.
- Fix the detector and all other extraction settings across accuracy
  experiments. Containerizing the service alone is not an accuracy gain.
- `/health` checks that the selected backend initialized. Test actual
  `/detect/faces` requests before committing to a full benchmark.
- Object and pose endpoints in the existing script are still stubs.
- There is no authentication in this service. The host port binds to
  loopback by default; do not expose it publicly. Remote access needs a
  separately secured network/proxy, not just a `0.0.0.0` bind.
- The image runs as a non-root user, drops Linux capabilities, and copies
  only the CV script and dependency file—not videos, database credentials,
  or the host Python environment. Dependency versions are pinned at the
  direct-dependency level; this is not a fully locked dependency tree.

## InsightFace models and migration from the Haar image

The default `cv-models` named volume preserves models across container
recreation. Models load before the HTTP server starts; on first use,
InsightFace downloads missing weights. The health check allows five minutes
for initialization, but restricted networks may require pre-provisioned weights.
To reuse an existing standalone cache without downloading again:

```bash
# Absolute host path; must contain models/buffalo_l/*.onnx.
export OGMEM_CV_MODEL_DIR="$HOME/.insightface"
docker compose -f deploy/docker-compose.cv.yml up -d --build --wait cv
```

The container runs as UID 10001. Ensure the mounted directory and existing
model files are readable/traversable by that user; downloading missing
weights additionally requires write permission. Do not make the cache
world-writable. Use the same model-directory setting and Compose project
when switching between standalone and main-stack entry points.

After rebuilding, `/health` must show `backend: insightface`,
`face_embeddings: true`, and `model_name: buffalo_l`. Send a real frame
containing a visible face to `/detect/faces` and check that detected faces
have finite, nonzero 512-element embeddings. An empty detection list is
not evidence that recognition works.

Old Haar extraction caches contain no embeddings. Rerun face extraction
and subsequent face clustering/episodic/index stages into a new run directory;
restarting CV or rerunning clustering over old detections cannot repair them.
Keep existing benchmark runs for comparison. This change does not resolve
the separate pyannote model-access/diarization failure.
