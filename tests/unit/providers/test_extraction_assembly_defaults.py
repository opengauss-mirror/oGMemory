"""Shared category assembly is standard extraction behavior, with no opt-in switch."""

from unittest.mock import Mock

import pytest

from providers.llm.mock_llm import MockLLM
from providers.unified_config import OgMemConfig
from service.api import MemoryWriteAPI


def test_default_configuration_has_no_write_planning_switch(tmp_path, monkeypatch):
    monkeypatch.delenv("OGMEM_EXTRACTION_MODE", raising=False)
    path = tmp_path / "ogmemory.yaml"
    path.write_text("{}", encoding="utf-8")

    cfg = OgMemConfig.load(str(path))

    assert OgMemConfig().extraction_mode == "eager"
    assert (cfg.extraction_mode, cfg.after_turn_threshold, cfg.dual_pass_enabled) == ("eager", 400, False)
    assert not hasattr(OgMemConfig(), "eager_write_plan_enabled")


@pytest.mark.parametrize("mode", ["fast", "lazy"])
def test_explicit_extraction_modes_override_default(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("OGMEM_EXTRACTION_MODE", "lazy")
    path = tmp_path / "ogmemory.yaml"
    path.write_text("{}", encoding="utf-8")

    assert OgMemConfig.load(str(path)).extraction_mode == "lazy"

    path.write_text(f"memory:\n  extraction:\n    mode: {mode}\n", encoding="utf-8")

    assert OgMemConfig.load(str(path)).extraction_mode == mode


def test_default_mode_preserves_explicit_deterministic_engine(tmp_path, monkeypatch):
    monkeypatch.delenv("OGMEM_EXTRACTION_MODE", raising=False)
    path = tmp_path / "ogmemory.yaml"
    path.write_text(
        "memory:\n  extraction:\n    engine: deterministic\n"
        "    deterministic:\n      spacy_model_en: en_core_web_trf\n",
        encoding="utf-8",
    )

    cfg = OgMemConfig.load(str(path))

    assert cfg.evidence_enabled
    assert cfg.deterministic_config().spacy_model_en == "en_core_web_trf"


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
