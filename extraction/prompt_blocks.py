"""Render selected template components through the manager's public protocol."""

from extraction.assembly_contract import UnsupportedAssemblyTemplateError


def render_blocks(manager, section, contract, **variables):
    supports = getattr(manager, "supports_assembly", None)
    render = getattr(manager, "render_selected", None)
    if not callable(supports) or not callable(render) or not supports("extraction"):
        raise UnsupportedAssemblyTemplateError("Manager or template does not support assembly")
    return render("extraction", section, decision=contract, **variables)
