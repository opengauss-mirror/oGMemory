"""Shared scope qualification keeps public node names concise."""

import pytest

from core.memory_names import (
    canonical_memory_key,
    canonical_object_name,
    canonical_scope,
    is_meaningful_memory_key,
    qualify_memory_key,
    unscoped_memory_key,
)


@pytest.mark.parametrize("key,scope,expected", [
    ("receiver", "project_aurora", "project_aurora_receiver"),
    ("project_aurora_receiver", "project_aurora", "project_aurora_receiver"),
    ("receiver_project_aurora", "project_aurora", "receiver_project_aurora"),
    ("receiver_project_aurora_optical", "project_aurora", "receiver_project_aurora_optical"),
    ("project_aurora", "project_aurora", "project_aurora"),
    ("receiver", "", "receiver"),
])
def test_scope_qualification_produces_one_scoped_name(key, scope, expected):
    assert qualify_memory_key(key, scope) == expected
    assert qualify_memory_key(expected, scope) == expected


@pytest.mark.parametrize("name,expected", [
    ("Bo", "bo"),
    ("Ed", "ed"),
    ("Li", "li"),
    ("Ｂｏ", "bo"),
    ("Ås", "ås"),
    ("李雷", "李雷"),
    ("ProjectAtlas", "projectatlas"),
    ("Atlas.v2", "atlasv2"),
    ("Ｐｒｏｊｅｃｔ－Ａｔｌａｓ", "project_atlas"),
    ("张伟", "张伟"),
    ("星河项目 / 接收器", "星河项目_接收器"),
])
def test_names_preserve_readable_unicode_identity(name, expected):
    assert canonical_memory_key(name) == expected
    assert canonical_memory_key(expected) == expected
    assert is_meaningful_memory_key(expected)


@pytest.mark.parametrize("name,expected", [
    (" Ｃ＋＋ ", "c++"),
    ("C#", "c#"),
    ("AT&T", "at&t"),
    ("AT-T", "at-t"),
    (" Atlas\t  Sensor ", "atlas sensor"),
    (" 星河　接收器 ", "星河 接收器"),
])
def test_object_identity_preserves_symbols_and_normalizes_unicode_space(name, expected):
    assert canonical_object_name(name) == expected
    assert canonical_object_name(expected) == expected


@pytest.mark.parametrize("scope,expected", [
    ("ProjectAtlas", "project_atlas"),
    ("project_atlas", "project_atlas"),
    ("project-atlas", "project_atlas"),
    (" Project\tAtlas ", "project_atlas"),
    (" Ｃ＋＋ ", "c++"),
    ("C#", "c#"),
    ("AT&T", "at&t"),
    ("AT-T", "at_t"),
    ("module/a", "module/a"),
    ("module#a", "module#a"),
    ("module\\a", "module\\a"),
])
def test_scope_identity_preserves_symbols_and_normalizes_word_separators(scope, expected):
    assert canonical_scope(scope) == expected
    assert canonical_scope(expected) == expected


@pytest.mark.parametrize("key,scope,expected", [
    ("melanie_pets", "melanie", "pets"),
    ("pets_melanie", "melanie", "pets"),
    ("pets", "melanie", "pets"),
    ("melanie", "melanie", "melanie"),
    ("pets", "", "pets"),
])
def test_unscoped_key_compares_scoped_spellings_equal(key, scope, expected):
    assert unscoped_memory_key(key, scope) == expected
