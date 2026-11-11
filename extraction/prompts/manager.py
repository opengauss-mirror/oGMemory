"""Prompt template manager for extraction prompts.

Loads Jinja2 YAML templates, renders them with variables, and provides
fallback to ensure robustness when template files are missing or corrupted.
"""

from collections.abc import Collection
from inspect import signature
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError

from core.logging_config import get_logger
from extraction.assembly_contract import UnsupportedAssemblyTemplateError

logger = get_logger(__name__)

# Default template directory (sibling of this file)
_DEFAULT_TEMPLATE_DIR = Path(__file__).parent / "templates"
_REQUIRED_SECTIONS = ("system_prompt", "output_instruction")


class PromptManager:
    """Load and render extraction prompt templates.

    Templates are YAML files with sections like system_prompt, examples,
    conversation_header, output_instruction. Each section can contain
    Jinja2 variables for dynamic rendering.

    Usage:
        mgr = PromptManager()
        prompt = mgr.render("extraction", "system_prompt",
                            session_summary="Previously extracted...")
        has = mgr.has_template("extraction")
    """

    def __init__(self, template_dir: str | Path | None = None):
        """Initialize PromptManager.

        Args:
            template_dir: Path to templates directory.
                         Defaults to extraction/prompts/templates/
        """
        if template_dir is None:
            template_dir = _DEFAULT_TEMPLATE_DIR
        self._template_dir = Path(template_dir)
        # Templates produce plain-text prompts, never HTML.
        self._env = Environment(
            loader=FileSystemLoader(str(self._template_dir)),
            autoescape=False,  # nosec B701
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._cache: dict[str, dict] = {}
        self._validated_assembly: set[str] = set()

    def load(self, template_name: str) -> dict:
        """Load and cache a YAML template file.

        Args:
            template_name: Template name without .yaml extension

        Returns:
            Dict parsed from YAML file

        Raises:
            FileNotFoundError: If template file does not exist
        """
        if template_name not in self._cache:
            path = self._template_dir / f"{template_name}.yaml"
            if not path.exists():
                raise FileNotFoundError(f"Template not found: {path}")
            try:
                with open(path, encoding="utf-8") as f:
                    template_data = yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                mark = getattr(exc, "problem_mark", None)
                location = ""
                if mark is not None:
                    location = f" line {mark.line + 1}, column {mark.column + 1}"
                raise ValueError(f"Failed to parse template {path}{location}: {exc}") from exc
            if not isinstance(template_data, dict):
                raise ValueError(f"Template {path} must be a YAML mapping")
            missing = [key for key in _REQUIRED_SECTIONS if not template_data.get(key)]
            if missing:
                raise ValueError(
                    f"Template {path} missing required section(s): {', '.join(missing)}"
                )
            if (
                template_name in {"extraction", "extraction_graph"}
                or "selection" in template_data
                or isinstance(template_data.get("system_prompt"), list)
                or isinstance(template_data.get("examples"), list)
            ):
                self._validate_components(template_data, template_name)
                self._validated_assembly.add(template_name)
            self._cache[template_name] = template_data
            logger.debug("Loaded template: %s", template_name)
        return self._cache[template_name]

    def render(
        self,
        template_name: str,
        section: str,
        *,
        categories: Collection[str] | None = None,
        **variables: Any,
    ) -> str:
        """Render a specific section of a template with Jinja2.

        Args:
            template_name: Template name without .yaml extension
            section: Key within the YAML file (e.g. "system_prompt")
            **variables: Jinja2 template variables

        Returns:
            Rendered string, or empty string if section not found
        """
        try:
            template_data = self.load(template_name)
        except FileNotFoundError as exc:
            logger.warning("%s, returning empty", exc)
            return ""

        raw = template_data.get(section, "")
        if categories is not None and section in ("system_prompt", "examples"):
            _, declared = self._assembly_data(template_name)
            selected = frozenset(categories)
            if selected - declared:
                raise UnsupportedAssemblyTemplateError(
                    "Template does not cover selected categories"
                )
            label = "types" if section == "system_prompt" else "requires_types"
            raw = [
                block
                for block in raw
                if not block.get(label)
                or (
                    set(block[label]) <= selected
                    if section == "examples"
                    else bool(set(block[label]) & selected)
                )
            ]
        if (
            section == "selection"
            and isinstance(raw, dict)
            and "input" in raw
            and "runtime_input_json" not in variables
        ):
            raise ValueError("selection.input requires runtime_input_json")
        raw_text = self._section_text(raw, section)
        return self._render_text(template_name, section, raw_text, variables)

    def _render_text(self, template_name, section, raw_text, variables):
        if not raw_text:
            return ""
        try:
            env = (
                self._env.overlay(undefined=StrictUndefined)
                if section == "selection"
                else self._env
            )
            tmpl = env.from_string(raw_text)
            return tmpl.render(**variables)
        except TemplateError as e:
            logger.error("Template render error in %s/%s: %s", template_name, section, e)
            if section == "selection":
                raise ValueError(
                    f"Template configuration error in {template_name}/{section}: {e}"
                ) from e
            # Return raw text as fallback
            return str(raw_text)

    @staticmethod
    def _section_text(value, section):
        """Join component text verbatim; dependency errors do not lose full fallback."""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            if any(not isinstance(block, dict) or not isinstance(block.get("text"), str)
                   for block in value):
                raise ValueError(f"Invalid component text in {section}")
            return "".join(block["text"] for block in value)
        if section == "selection" and isinstance(value, dict):
            text = [value.get("instruction", "")]
            text.extend(f"{name}: {description}" for name, description in value.get("types", {}).items())
            text.extend([value.get("classification_rules", ""), value.get("input", "")])
            return "\n\n".join(text)
        if value is None:
            return ""
        raise ValueError(f"Invalid template section: {section}")

    def _assembly_data(self, template_name):
        data = self.load(template_name)
        if template_name not in self._validated_assembly:
            raise UnsupportedAssemblyTemplateError(f"{template_name}: missing component structure")
        return data, frozenset(data["selection"]["types"])

    @staticmethod
    def _validate_components(data, template_name):
        if not isinstance(data, dict):
            raise UnsupportedAssemblyTemplateError(f"{template_name}: template must be a mapping")
        selection = data.get("selection")
        if not isinstance(selection, dict):
            raise UnsupportedAssemblyTemplateError(f"{template_name}/selection: expected mapping")
        for field in ("instruction", "classification_rules"):
            if not isinstance(selection.get(field), str):
                raise UnsupportedAssemblyTemplateError(
                    f"{template_name}/selection.{field}: expected string"
                )
        types = selection.get("types")
        if (
            not isinstance(types, dict)
            or not types
            or any(
                not isinstance(name, str) or not name or not isinstance(description, str)
                for name, description in types.items()
            )
        ):
            raise UnsupportedAssemblyTemplateError(
                f"{template_name}/selection.types: invalid catalog"
            )
        if template_name != "extraction_graph":
            input_text = selection.get("input")
            if not isinstance(input_text, str) or input_text.count("{{ runtime_input_json }}") != 1:
                raise UnsupportedAssemblyTemplateError(
                    f"{template_name}/selection.input requires one runtime_input_json placeholder"
                )
        categories = frozenset(selection["types"])
        for section, label in (("system_prompt", "types"), ("examples", "requires_types")):
            blocks = data.get(section)
            if not isinstance(blocks, list) or (section == "system_prompt" and not blocks):
                raise UnsupportedAssemblyTemplateError(
                    f"{template_name}/{section}: invalid component section"
                )
            ids = set()
            for block in blocks:
                if (
                    not isinstance(block, dict)
                    or not isinstance(block.get("id"), str)
                    or not block["id"]
                    or block["id"] in ids
                    or not isinstance(block.get("text"), str)
                ):
                    raise UnsupportedAssemblyTemplateError(
                        f"{template_name}/{section}: invalid or duplicate block {block.get('id') if isinstance(block, dict) else '<non-object>'}"
                    )
                ids.add(block["id"])
                labels = block.get(label, [])
                if (
                    not isinstance(labels, list)
                    or any(not isinstance(item, str) for item in labels)
                    or len(labels) != len(set(labels))
                    or set(labels) - categories
                ):
                    raise UnsupportedAssemblyTemplateError(
                        f"{template_name}/{section}/{block['id']}: invalid {label}"
                    )
        return data, categories

    def validate_assembly_template(self, template_name: str) -> None:
        self._assembly_data(template_name)

    def supports_assembly(self, template_name: str) -> bool:
        try:
            self._assembly_data(template_name)
        except (FileNotFoundError, UnsupportedAssemblyTemplateError):
            return False
        return True

    def has_template(self, template_name: str) -> bool:
        """Check if a template file exists."""
        return (self._template_dir / f"{template_name}.yaml").exists()


def validate_prompt_manager(manager, template_name="extraction"):
    """Require injected managers to implement the same category-aware render contract."""
    render = getattr(manager, "render", None)
    if not callable(render):
        raise ValueError("Prompt manager must provide render()")
    try:
        signature(render).bind(
            template_name, "system_prompt", categories={"entity"}, session_summary=""
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Prompt manager render() must accept categories and template variables"
        ) from exc
    load = getattr(manager, "load", None)
    if not callable(load):
        raise ValueError("Prompt manager must provide load() for template validation")
    data = load(template_name)
    uses_builtin_load = (
        isinstance(manager, PromptManager)
        and getattr(load, "__func__", None) is PromptManager.load
        and getattr(load, "__self__", None) is manager
    )
    if not uses_builtin_load or template_name not in manager._validated_assembly:
        PromptManager._validate_components(data, template_name)
        if isinstance(manager, PromptManager):
            manager._validated_assembly.add(template_name)
