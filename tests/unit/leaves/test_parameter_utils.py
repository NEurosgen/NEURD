"""Unit tests for neurd.parameter_utils — Parameters/PackageParameters classes
plus a few of the smaller pure-Python helpers.
"""
import json
import os
from pathlib import Path

import pytest

from tests.unit.leaves import skip_if_datasci_tools_unusable

skip_if_datasci_tools_unusable()

from neurd import parameter_utils as paru


# ---------------- Parameters ----------------


def test_parameters_strips_global_suffix():
    p = paru.Parameters(data={"a": 1, "b_global": 2})
    assert p["a"] == 1
    assert p["b"] == 2
    assert "b_global" not in p
    assert "a" in p


def test_parameters_attr_and_item_access():
    p = paru.Parameters(data={"a": 1})
    assert p.a == 1
    assert p["a"] == 1


def test_parameters_setitem_and_setattr_update_internal_dict():
    p = paru.Parameters(data={"a": 1})
    p["a"] = 10
    assert p.a == 10
    p.a = 20
    assert p["a"] == 20


def test_parameters_update_merges_keys():
    p = paru.Parameters(data={"a": 1})
    p.update({"a": 99, "c": 3})
    assert p.a == 99
    assert p.c == 3


def test_parameters_dict_property_returns_internal_state():
    p = paru.Parameters(data={"a": 1, "b": 2})
    assert p.dict == {"a": 1, "b": 2}


def test_parameters_from_json_filepath(tmp_path):
    fp = tmp_path / "params.json"
    fp.write_text(json.dumps({"x": 1, "y_global": 2}))
    p = paru.Parameters(filepath=str(fp))
    assert p.x == 1
    assert p.y == 2


def test_parameters_init_from_other_parameters_instance():
    p1 = paru.Parameters(data={"a": 1})
    p2 = paru.Parameters(data=p1)
    p2["a"] = 99
    assert p1.a == 1  # original unaffected


# ---------------- PackageParameters ----------------


def test_package_parameters_holds_per_module_dicts():
    pp = paru.PackageParameters(data={"mod_a": {"x": 1}, "mod_b": {"y": 2}})
    assert "mod_a" in pp
    assert pp["mod_a"].x == 1
    assert pp["mod_b"].y == 2


def test_package_parameters_dict_property():
    pp = paru.PackageParameters(data={"mod_a": {"x": 1}})
    assert pp.dict == {"mod_a": {"x": 1}}


# ---------------- Pure helpers ----------------


def test_add_global_name_to_dict():
    out = paru._add_global_name_to_dict({"a": 1, "b": 2})
    assert out == {"a_global": 1, "b_global": 2}


def test_parameter_config_folder_returns_existing_dir():
    folder = paru.parameter_config_folder(return_str=True)
    assert isinstance(folder, str)
    assert os.path.isdir(folder)
    assert os.path.basename(folder) == "parameter_configs"


def test_parameter_config_folder_can_return_path():
    folder = paru.parameter_config_folder(return_str=False)
    assert isinstance(folder, Path)


def test_this_directory_returns_package_root():
    assert os.path.isdir(paru._this_directory())


def test_jsonable_dict_drops_non_serialisable_values():
    class _NotJsonable:
        pass

    out = paru._jsonable_dict({"a": 1, "b": _NotJsonable()})
    assert "a" in out
    assert "b" not in out


# ---------------- Regression tests ----------------


def test_clean_modules_dict_strips_non_jsonable_in_leaf_dicts():
    """Regression for B1: clean_modules_dict iterated `data` instead of the
    nested dicts and therefore corrupted the structure / left non-jsonable
    values in place.
    """
    class _NotJsonable:
        pass

    data = {
        "mod_a": {
            "global_parameters": {
                "no_category": {"a": 1, "b": _NotJsonable()},
            },
            "attributes": {
                "no_category": {"c": "ok", "d": _NotJsonable()},
            },
        }
    }
    out = paru._clean_modules_dict(data)
    # Shape preserved.
    assert set(out) == {"mod_a"}
    assert set(out["mod_a"]) == {"global_parameters", "attributes"}
    # Non-jsonable values dropped from leaves.
    assert out["mod_a"]["global_parameters"]["no_category"] == {"a": 1}
    assert out["mod_a"]["attributes"]["no_category"] == {"c": "ok"}


def test_package_parameters_can_be_constructed_from_another_instance():
    """Regression for B2: copying via ``PackageParameters(other_obj)``
    previously set ``self.data`` (wrong attr name) and only triggered when
    the instance was passed as the ``filepath=`` kwarg.
    """
    pp1 = paru.PackageParameters(data={"mod_a": {"x": 1}})
    pp2 = paru.PackageParameters(pp1)  # positional → data=pp1

    # Has the correct internal attribute.
    assert hasattr(pp2, "_data")
    assert "mod_a" in pp2
    assert pp2["mod_a"].x == 1

    # Independent copy: mutating one does not affect the other.
    pp2["mod_a"]["x"] = 99
    assert pp1["mod_a"].x == 1
    assert pp2["mod_a"].x == 99


def test_package_parameters_update_round_trip_uses_copy_branch():
    """``update`` internally wraps `other_obj` via ``self.__class__(other_obj)``
    — exercises the same copy branch as the regression above.
    """
    pp1 = paru.PackageParameters(data={"mod_a": {"x": 1}})
    pp2 = paru.PackageParameters(data={"mod_a": {"x": 42}, "mod_b": {"y": 7}})
    pp1.update(pp2)
    assert pp1["mod_a"].x == 42
    assert pp1["mod_b"].y == 7


# ---------------- attr_map / B3 ----------------


def test_attr_map_strips_trailing_global_suffix():
    """attr_map("foo_global") → resolves to "foo" in the internal dict."""
    p = paru.Parameters(data={"foo": 42})
    result = p.attr_map(["foo_global"])
    assert result == {"foo_global": 42}


def test_attr_map_does_not_strip_infix_global():
    """Regression for B3: .replace(suf, '') removed suffix anywhere in the
    string, so "foo_global_bar" would incorrectly match "foo_bar".
    With .removesuffix it only strips from the end.
    """
    p = paru.Parameters(data={"foo_bar": 99})
    result = p.attr_map(["foo_global_bar"])
    # "foo_global_bar" should NOT resolve to "foo_bar" — suffix is not at end.
    assert "foo_global_bar" not in result


# ---------------- modes_global_param_and_attributes_dict_from_module ----------------

import types


def _fake_module(**attrs):
    """Build a fake module with the given top-level attributes."""
    m = types.ModuleType("neurd.fake_mod")
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def test_modes_dict_basic_default_mode():
    """Module with one global_parameters_dict and one attributes_dict
    for the 'default' mode produces a single mode entry with one
    'no_category' bucket per att type.
    """
    m = _fake_module(
        global_parameters_dict_default={"k": 1},
        attributes_dict_default={"x": "hi"},
    )
    out = paru.modes_global_param_and_attributes_dict_from_module(m, modes="default")
    assert out == {
        "default": {
            "fake_mod": {
                "global_parameters": {"no_category": {"k": 1}},
                "attributes": {"no_category": {"x": "hi"}},
            }
        }
    }


def test_modes_dict_multi_category_splits_subdicts():
    """When module has both base `_dict_default` and a category-suffixed
    `_dict_default_celltype1`, they split into separate categories;
    keys present in a category dict are stripped from no_category."""
    m = _fake_module(
        global_parameters_dict_default={"shared": 1, "a": 0},
        global_parameters_dict_default_celltype1={"a": 2},
    )
    out = paru.modes_global_param_and_attributes_dict_from_module(m, modes="default")
    cats = out["default"]["fake_mod"]["global_parameters"]
    assert cats["celltype1"] == {"a": 2}
    # "shared" is in base but not in celltype1 → kept in no_category;
    # "a" is in celltype1 → removed from no_category leftover.
    assert cats["no_category"] == {"shared": 1}


def test_modes_dict_add_global_suffix_appends_suffix():
    """`add_global_suffix=True` rewrites keys in global_parameters with
    `_global` suffix, but does not touch `attributes`."""
    m = _fake_module(
        global_parameters_dict_default={"k": 1},
        attributes_dict_default={"x": 2},
    )
    out = paru.modes_global_param_and_attributes_dict_from_module(
        m, modes="default", add_global_suffix=True
    )
    fm = out["default"]["fake_mod"]
    assert fm["global_parameters"]["no_category"] == {"k_global": 1}
    assert fm["attributes"]["no_category"] == {"x": 2}  # untouched


def test_modes_dict_empty_module_returns_empty():
    """A module with no parameter dicts at all yields an empty mapping."""
    m = _fake_module()
    out = paru.modes_global_param_and_attributes_dict_from_module(m, modes="default")
    assert out == {}


def test_modes_dict_multiple_modes_filtered_independently():
    """Each mode looks for its own `_dict_<mode>` suffix and is independent."""
    m = _fake_module(
        global_parameters_dict_default={"k": 1},
        global_parameters_dict_h01={"k": 99},
        attributes_dict_default={"x": "d"},
        attributes_dict_h01={"x": "h"},
    )
    out = paru.modes_global_param_and_attributes_dict_from_module(
        m, modes=["default", "h01"]
    )
    assert out["default"]["fake_mod"]["global_parameters"]["no_category"] == {"k": 1}
    assert out["h01"]["fake_mod"]["global_parameters"]["no_category"] == {"k": 99}
    assert out["default"]["fake_mod"]["attributes"]["no_category"] == {"x": "d"}
    assert out["h01"]["fake_mod"]["attributes"]["no_category"] == {"x": "h"}


def test_modes_dict_clean_dict_false_preserves_non_jsonable():
    """`clean_dict=False` bypasses `_jsonable_dict` filtering on leaf cats."""
    class _NotJsonable:
        pass

    sentinel = _NotJsonable()
    m = _fake_module(global_parameters_dict_default={"a": 1, "b": sentinel})
    out = paru.modes_global_param_and_attributes_dict_from_module(
        m, modes="default", clean_dict=False
    )
    leaf = out["default"]["fake_mod"]["global_parameters"]["no_category"]
    assert leaf == {"a": 1, "b": sentinel}


# ---------------- category_param_from_module ----------------


def test_category_param_default_category_resolves_global_attrs():
    """Default category 'no_category' pulls keys from the base dict, applies
    the `_global` suffix (since att_type is global_parameters), then resolves
    each suffixed name to the module's top-level attribute."""
    m = _fake_module(
        global_parameters_dict_default={"foo": 1, "bar": 2},
        foo_global=100,
        bar_global=200,
    )
    assert paru.category_param_from_module(m) == {
        "foo_global": 100,
        "bar_global": 200,
    }


def test_category_param_specific_category_picks_only_overrides():
    """Asking for a sub-category returns only the keys that the sub-category
    overrides — not the unrelated keys from the base dict."""
    m = _fake_module(
        global_parameters_dict_default={"foo": 0, "shared": 5},
        global_parameters_dict_default_special={"foo": 1},
        foo_global=100,
        shared_global=500,
    )
    assert paru.category_param_from_module(m, category="special") == {
        "foo_global": 100,
    }


def test_category_param_missing_category_returns_empty():
    m = _fake_module(
        global_parameters_dict_default={"foo": 1},
        foo_global=100,
    )
    assert paru.category_param_from_module(m, category="nonexistent") == {}
