"""The ``znyx`` command (znyx.cli), run in-process with HOME and the working
directory redirected so nothing touches the developer's real files."""
import json

import pytest

import znyx
from znyx import cli
from znyx.config import SETTINGS, load_settings, starter_policy_path


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """HOME and cwd in tmp_path, no znyx setting inherited from the shell, and no .env
    file read (a stray one would leak into os.environ for the whole session)."""
    home = tmp_path / "home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    for setting in SETTINGS:
        for name in setting.env:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("ZNYX_CONFIG", raising=False)
    monkeypatch.setattr(cli, "load_dotenv_once", lambda: None)
    return {"home": home, "work": work}


def run(*argv):
    """``znyx.cli.main(argv)``; returns the exit code (main always exits)."""
    with pytest.raises(SystemExit) as exc:
        cli.main(list(argv))
    return exc.value.code


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

def test_version_flag(capsys):
    assert run("--version") == 0
    assert capsys.readouterr().out.strip() == f"znyx {znyx.__version__}"


def test_version_json_lists_the_three_components(capsys):
    assert run("version", "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["version"] == znyx.__version__
    assert set(data["components"]) == {"znyx-core", "znyx-runtime", "znyx-inference"}
    assert all(data["components"].values()), "every component is installed in the test env"


def test_version_text_names_each_component(capsys):
    assert run("version") == 0
    out = capsys.readouterr().out
    assert f"znyx {znyx.__version__}" in out
    for dist in znyx.COMPONENTS:
        assert dist in out


def test_no_command_prints_help_and_exits_zero(capsys):
    assert run() == 0
    assert "usage: znyx" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

def test_init_writes_a_loadable_config_and_the_starter_policy(tmp_path, capsys):
    out = tmp_path / "proj"
    assert run("init", "-o", str(out)) == 0
    config, policy = out / "znyx.yaml", out / "policies.yaml"
    assert config.is_file() and policy.is_file()
    assert policy.read_bytes() == starter_policy_path().read_bytes()
    settings = load_settings(str(config), env={}, cwd=out)      # no ConfigError
    assert settings["server.port"] == 8080
    assert settings["mode"] == "local"
    assert settings.sources["server.port"] == "file"
    assert "wrote" in capsys.readouterr().out


def test_init_refuses_to_overwrite_without_force(tmp_path, capsys):
    out = tmp_path / "proj"
    assert run("init", "-o", str(out)) == 0
    (out / "znyx.yaml").write_text("# mine\n")
    (out / "policies.yaml").write_text("# also mine\n")
    capsys.readouterr()
    assert run("init", "-o", str(out)) == 1
    assert "already exists" in capsys.readouterr().err
    assert (out / "znyx.yaml").read_text() == "# mine\n"
    assert (out / "policies.yaml").read_text() == "# also mine\n"


def test_init_refuses_when_only_one_file_exists(tmp_path):
    out = tmp_path / "proj"
    out.mkdir()
    (out / "policies.yaml").write_text("# mine\n")
    assert run("init", "-o", str(out)) == 1
    assert not (out / "znyx.yaml").exists(), "nothing is written when anything would be overwritten"
    assert (out / "policies.yaml").read_text() == "# mine\n"


def test_init_force_overwrites(tmp_path):
    out = tmp_path / "proj"
    assert run("init", "-o", str(out)) == 0
    (out / "znyx.yaml").write_text("# mine\n")
    (out / "policies.yaml").write_text("# also mine\n")
    assert run("init", "-o", str(out), "--force") == 0
    assert (out / "znyx.yaml").read_text().startswith("# Znyx configuration")
    assert (out / "policies.yaml").read_bytes() == starter_policy_path().read_bytes()


def test_init_defaults_to_the_current_directory(sandbox):
    assert run("init") == 0
    assert (sandbox["work"] / "znyx.yaml").is_file()


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_config_json_shows_sources_and_masks_the_token(monkeypatch, capsys):
    token = "rt_supersecrettoken123"
    monkeypatch.setenv("ZNYX_MODE", "managed")
    monkeypatch.setenv("ZNYX_CONTROL_PLANE_URL", "https://cp.example.com")
    monkeypatch.setenv("ZNYX_RUNTIME_TOKEN", token)
    assert run("config", "--json") == 0
    out = capsys.readouterr().out
    assert token not in out
    data = json.loads(out)
    assert data["config_file"] is None
    shown = data["settings"]
    assert shown["control_plane.token"] == {"value": "rt_s***", "source": "env ZNYX_RUNTIME_TOKEN"}
    assert shown["mode"] == {"value": "managed", "source": "env ZNYX_MODE"}
    assert shown["server.port"] == {"value": "8080", "source": "default"}


def test_config_json_reports_the_file_in_use(sandbox, capsys):
    (sandbox["work"] / "znyx.yaml").write_text("server:\n  port: 9321\n")
    assert run("config", "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["config_file"].endswith("znyx.yaml")
    assert data["settings"]["server.port"] == {"value": "9321", "source": "file"}


def test_config_text_lists_every_setting(capsys):
    assert run("config") == 0
    out = capsys.readouterr().out
    assert "config file: (none)" in out
    for setting in SETTINGS:
        assert setting.key in out


def test_config_with_a_bad_file_exits_two(sandbox, capsys):
    (sandbox["work"] / "znyx.yaml").write_text("server:\n  prot: 1\n")
    assert run("config") == 2
    assert "did you mean server.port" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# key
# ---------------------------------------------------------------------------

def test_key_prints_a_key_and_creates_the_file(sandbox, capsys):
    assert run("key") == 0
    captured = capsys.readouterr()
    key = captured.out.strip()
    key_file = sandbox["home"] / ".znyx" / "api-key"
    assert key.startswith("zx_")
    assert key_file.read_text().strip() == key
    assert str(key_file) in captured.err


def test_key_is_stable_until_rotated(sandbox, capsys):
    assert run("key") == 0
    first = capsys.readouterr().out.strip()
    assert run("key") == 0
    assert capsys.readouterr().out.strip() == first
    assert run("key", "--rotate") == 0
    rotated = capsys.readouterr().out.strip()
    assert rotated != first
    assert (sandbox["home"] / ".znyx" / "api-key").read_text().strip() == rotated


def test_key_with_a_configured_key_exits_one_and_writes_nothing(sandbox, monkeypatch, capsys):
    monkeypatch.setenv("RUNTIME_API_KEY", "from-the-environment")
    assert run("key") == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "RUNTIME_API_KEY" in captured.err
    assert "from-the-environment" not in captured.err
    assert not (sandbox["home"] / ".znyx").exists()


def test_key_rotate_with_a_configured_key_exits_one(monkeypatch, capsys):
    monkeypatch.setenv("RUNTIME_API_KEY", "from-the-environment")
    assert run("key", "--rotate") == 1
    assert "change it there" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

def test_configuration_error_exits_two_without_starting_anything(monkeypatch, capsys):
    import znyx.supervisor as supervisor

    def boom(*a, **k):
        raise AssertionError("the supervisor must not be constructed on a bad configuration")

    monkeypatch.setattr(supervisor, "Supervisor", boom)
    assert run("serve", "--port", "0") == 2
    err = capsys.readouterr().err
    assert "configuration error" in err
    assert "server.port" in err


def test_start_is_another_name_for_serve(monkeypatch, capsys):
    assert run("start", "--workers", "0") == 2
    assert "configuration error" in capsys.readouterr().err


def test_status_without_workers_exits_one(capsys):
    assert run("status") == 1
    assert "no running workers" in capsys.readouterr().out


def test_status_json_without_workers_is_an_empty_list(capsys):
    assert run("status", "--json") == 1
    assert json.loads(capsys.readouterr().out) == []


# ---------------------------------------------------------------------------
# models fetch on an air-gapped install
# ---------------------------------------------------------------------------

def test_models_fetch_refuses_when_airgapped_and_never_downloads(monkeypatch, capsys):
    import znyx_inference.runners._fetch as fetch

    def fail(*a, **k):
        raise AssertionError("fetch_model must not be called on an air-gapped install")

    monkeypatch.setattr(fetch, "fetch_model", fail)
    monkeypatch.setenv("ZNYX_AIRGAPPED", "true")
    assert run("models", "fetch", "--task", "prompt_injection") == 2
    assert "air-gapped" in capsys.readouterr().err


def test_models_without_an_action_exits_two(capsys):
    assert run("models") == 2
    assert "usage" in capsys.readouterr().err
