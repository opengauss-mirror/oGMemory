"""Run the shared LoCoMo evaluation against an application's Deep Agents host.

The host factory receives the evaluation namespace, including the memory proxy
URL, test identity and stage-specific runtime directory. Return a host with
prepare(), invoke(prompt, key) and close(). Invoke must map each logical key
to a native session, wait for the turn's middleware hooks and return
(assistant_text, token_usage). Use --parallel 1 for a serial host.
"""

from importlib import import_module

import eval as evaluation
from agent_plugin_locomo import PluginAdapter


class DeepAgentsAdapter(PluginAdapter):
    def settle(self, sid):
        # Some server versions expose recall evidence rather than archives.
        settle = getattr(self.host, "settle", None)
        if settle is not None:
            evidence = settle(sid)
            if not isinstance(evidence, dict) or evidence.get("session_id") != sid:
                raise RuntimeError(f"Host settle() must return evidence for source session {sid}")
            return evidence
        return super().settle(sid)


def main(argv=None):
    parser = evaluation.build_parser("deepagents", transports=("plugin",))
    parser.set_defaults(read_only_qa=True)
    parser.add_argument(
        "--host-factory",
        metavar="MODULE:CALLABLE",
        help="Importable factory that connects the real Deep Agents host (required for ingest and qa)",
    )

    def create_adapter(args):
        module, sep, name = (args.host_factory or "").partition(":")
        if not sep or not module or not name:
            parser.error("ingest and qa require --host-factory MODULE:CALLABLE")
        try:
            factory = getattr(import_module(module), name)
        except (ImportError, AttributeError) as exc:
            parser.error(f"cannot load --host-factory {args.host_factory}: {exc}")
        if not callable(factory):
            parser.error(f"--host-factory {args.host_factory} must be callable")
        adapter = DeepAgentsAdapter(args, evaluation, host_factory=factory)
        adapter.description["host_factory"] = args.host_factory
        return adapter

    return evaluation.main(argv, parser=parser, adapter_factory=create_adapter)


if __name__ == "__main__":
    raise SystemExit(main())
