"""The four packages are one product: lockstep versions, exact pins, a starter policy
that cannot drift from the image's default, and dashboard assets that ship."""
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

REPO = Path(__file__).resolve().parents[1]
PACKAGES = REPO / "packages"
NAMES = ("znyx", "znyx-core", "znyx-runtime", "znyx-inference")


def _pyproject(name):
    return tomllib.loads((PACKAGES / name / "pyproject.toml").read_text())


def _versions():
    return {name: _pyproject(name)["project"]["version"] for name in NAMES}


def test_all_four_packages_have_the_same_version():
    versions = _versions()
    assert len(set(versions.values())) == 1, versions


def test_znyx_pins_every_component_dependency_to_exactly_that_version():
    project = _pyproject("znyx")["project"]
    version = project["version"]
    found = {}
    for spec in project["dependencies"]:
        req = Requirement(spec)
        if req.name.startswith("znyx-"):
            found[req.name] = str(req.specifier)
    assert set(found) == {"znyx-core", "znyx-runtime", "znyx-inference"}
    for name, specifier in found.items():
        assert specifier == f"=={version}", f"{name}: {specifier}"


def test_znyx_extras_pin_component_extras_to_exactly_that_version():
    project = _pyproject("znyx")["project"]
    version = project["version"]
    seen = 0
    for extra, specs in project["optional-dependencies"].items():
        for spec in specs:
            req = Requirement(spec)
            if req.name.startswith("znyx-"):
                seen += 1
                assert str(req.specifier) == f"=={version}", f"extra {extra}: {spec}"
    assert seen >= 3, "the ml, export and torch extras each pull a component extra"


def test_the_extras_name_extras_the_component_actually_defines():
    project = _pyproject("znyx")["project"]
    inference_extras = _pyproject("znyx-inference")["project"].get("optional-dependencies", {})
    for specs in project["optional-dependencies"].values():
        for spec in specs:
            req = Requirement(spec)
            if req.name == "znyx-inference":
                assert req.extras, spec
                assert req.extras <= set(inference_extras), (spec, sorted(inference_extras))


@pytest.mark.parametrize("name", ["znyx-runtime", "znyx-inference"])
def test_components_depend_on_a_compatible_znyx_core(name):
    version = _versions()["znyx-core"]
    for spec in _pyproject(name)["project"]["dependencies"]:
        req = Requirement(spec)
        if req.name == "znyx-core":
            assert req.specifier.contains(version), f"{name} needs {spec}, repo has {version}"


def test_the_starter_policy_is_byte_identical_to_the_default_policy():
    starter = PACKAGES / "znyx" / "src" / "znyx" / "templates" / "policies.yaml"
    assert starter.read_bytes() == (REPO / "config" / "policies.yaml").read_bytes(), (
        "znyx init would write a different policy from the one the image ships; "
        "copy config/policies.yaml over the packaged starter")


def test_the_starter_policy_ships_in_the_package():
    text = (PACKAGES / "znyx" / "pyproject.toml").read_text()
    assert "templates/*.yaml" in text


def test_the_dashboard_assets_exist_and_are_packaged():
    folder = PACKAGES / "znyx-runtime" / "src" / "znyx_runtime" / "dashboard"
    for name in ("index.html", "app.js", "app.css"):
        assert (folder / name).is_file(), name
    package_data = _pyproject("znyx-runtime")["tool"]["setuptools"]["package-data"]["znyx_runtime"]
    for pattern in ("dashboard/*.html", "dashboard/*.js", "dashboard/*.css"):
        assert pattern in package_data, pattern


def test_the_znyx_command_is_declared():
    assert _pyproject("znyx")["project"]["scripts"]["znyx"] == "znyx.cli:main"


def test_the_installed_distributions_match_the_source_versions():
    from importlib.metadata import version

    for name, expected in _versions().items():
        assert version(name) == expected, f"{name}: installed {version(name)}, source {expected}"
