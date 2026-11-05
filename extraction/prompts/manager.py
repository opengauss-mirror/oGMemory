"""Prompt template manager for extraction prompts.

Loads Jinja2 YAML templates, renders them with variables, and provides
fallback to ensure robustness when template files are missing or corrupted.
"""

from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, TemplateError

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
        self._env = Environment(
            loader=FileSystemLoader(str(self._template_dir)),
            autoescape=False,  # We're generating prompts, not HTML
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._cache: dict[str, dict] = {}

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
            self._cache[template_name] = template_data
            logger.debug("Loaded template: %s", template_name)
        return self._cache[template_name]

    def render(
        self,
        template_name: str,
        section: str,
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

        raw_text = self._section_text(template_data.get(section, ""), section)
        return self._render_text(template_name, section, raw_text, variables)

    def _render_text(self, template_name, section, raw_text, variables):
        if not raw_text:
            return ""
        try:
            tmpl = self._env.from_string(raw_text)
            return tmpl.render(**variables)
        except TemplateError as e:
            logger.error("Template render error in %s/%s: %s", template_name, section, e)
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
            text = [value.get("instruction", ""), value.get("classification_rules", "")]
            text.extend(f"{name}: {description}" for name, description in value.get("types", {}).items())
            return "\n\n".join(text)
        if value is None:
            return ""
        raise ValueError(f"Invalid template section: {section}")

    def _assembly_data(self, template_name):
        data = self.load(template_name)
        if type(data.get("assembly_version")) is not int or data["assembly_version"] != 1:
            raise UnsupportedAssemblyTemplateError("Unsupported component template version")
        selection = data.get("selection")
        if (not isinstance(selection, dict)
                or not isinstance(selection.get("instruction"), str)
                or not isinstance(selection.get("classification_rules"), str)
                or not isinstance(selection.get("types"), dict)
                or not selection["types"]
                or any(not isinstance(name, str) or not name
                       or not isinstance(description, str)
                       for name, description in selection["types"].items())):
            raise UnsupportedAssemblyTemplateError("Invalid selection catalog")
        categories = frozenset(selection["types"])
        for section, label in (("system_prompt", "types"), ("examples", "requires_types")):
            blocks = data.get(section)
            if not isinstance(blocks, list):
                raise UnsupportedAssemblyTemplateError("Legacy or invalid component section")
            ids = set()
            for block in blocks:
                if (not isinstance(block, dict)
                        or set(block) - {"id", "text", label}
                        or not isinstance(block.get("id"), str) or not block["id"]
                        or block["id"] in ids
                        or not isinstance(block.get("text"), str)):
                    raise UnsupportedAssemblyTemplateError("Invalid component definition")
                ids.add(block["id"])
                labels = block.get(label, [])
                if (not isinstance(labels, list)
                        or any(not isinstance(item, str) for item in labels)
                        or len(labels) != len(set(labels))
                        or set(labels) - categories):
                    raise UnsupportedAssemblyTemplateError("Invalid component dependencies")
        for section in ("identity_block", "output_instruction", "lazy_tools_instruction", "conversation_header"):
            if section in data and not isinstance(data[section], str):
                raise UnsupportedAssemblyTemplateError("Invalid whole-section template")
        return data, categories

    def supports_assembly(self, template_name: str) -> bool:
        """Recognize component YAML without bypassing a legacy render override."""
        def owner_index(name):
            if name in self.__dict__:
                return -1
            return next(index for index, cls in enumerate(type(self).__mro__)
                        if name in cls.__dict__)

        if owner_index("render") < owner_index("render_selected"):
            return False
        try:
            self._assembly_data(template_name)
        except (FileNotFoundError, UnsupportedAssemblyTemplateError):
            return False
        return True

    def render_selected(self, template_name: str, section: str, *, decision, **variables: Any) -> str:
        """Select applicable raw components, then render through this manager."""
        if not self.supports_assembly(template_name):
            raise UnsupportedAssemblyTemplateError("Manager or template does not support assembly")
        data, categories = self._assembly_data(template_name)
        if decision.scope == "full":
            return self.render(template_name, section, **variables)
        selected = frozenset(decision.categories)
        if selected - categories:
            raise UnsupportedAssemblyTemplateError("Template does not cover selected categories")
        raw = data.get(section, "")
        if section in ("system_prompt", "examples"):
            label = "requires_types" if section == "examples" else "types"
            raw = [block for block in raw
                   if not block.get(label)
                   or (set(block[label]) <= selected if section == "examples"
                       else bool(set(block[label]) & selected))]
        text = self._section_text(raw, section)
        return self._render_text(template_name, section, text, variables)

    def has_template(self, template_name: str) -> bool:
        """Check if a template file exists."""
        return (self._template_dir / f"{template_name}.yaml").exists()
