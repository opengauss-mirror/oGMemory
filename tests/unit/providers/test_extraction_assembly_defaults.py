"""Shared category assembly is standard extraction behavior, with no opt-in switch."""

from unittest.mock import Mock

import pytest

from providers.llm.mock_llm import MockLLM
from providers.unified_config import OgMemConfig
from service.api import MemoryWriteAPI


def test_default_configuration_has_no_write_planning_switch(tmp_path):
    path = tmp_path / "ogmemory.yaml"
    path.write_text("{}")
    cfg = OgMemConfig.load(str(path))
    assert (cfg.extraction_mode, cfg.after_turn_threshold, cfg.dual_pass_enabled) == ("eager", 400, False)
    assert not hasattr(OgMemConfig(), "eager_write_plan_enabled")


@pytest.mark.parametrize("factory", ["direct", "global"])
def test_api_factories_use_standard_extraction_without_an_opt_in(factory, monkeypatch):
    if factory == "global":
        import service.api as api_module
        monkeypatch.setattr(api_module, "_default_write_api", None)
        create = api_module.init_write_api
    else:
        create = MemoryWriteAPI
    writer = create(Mock(), MockLLM())
    extractor = writer._pipeline._extractors[0]
    assert extractor._mode == "eager"
    assert extractor._native_temperatures() == (None,)


def test_direct_api_keeps_lazy_mode_available_without_an_opt_in():
    writer = MemoryWriteAPI(Mock(), MockLLM(), extraction_mode="lazy")
    assert writer._pipeline._extractors[0]._mode == "lazy"
