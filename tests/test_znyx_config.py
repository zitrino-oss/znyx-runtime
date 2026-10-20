"""The ``znyx.yaml`` config layer (znyx.config): precedence, validation, derived
values, the child-process environment and the generated API key."""
import os
import stat
from pathlib import Path

import pytest

from znyx import config as zc
from znyx.config import ConfigError, load_settings


def _load(tmp_path, text=None, env=None, overrides=None, config_path=None):
    """Load settings with a hermetic environment (never the real os.environ)."""
    if text is not None:
        (tmp_path / "znyx.yaml").write_text(text)
    return load_settings(config_path, overrides, env={} if env is None else env, cwd=tmp_path)


def _service_env(settings, workers=1, policy="/p/policies.yaml", run_dir=Path("/run/znyx")):
    return zc.service_env(settings, workers, policy_path=policy, run_directory=run_dir,
                          platform_version="9.9.9")


# ---------------------------------------------------------------------------
# Defaults, file, env, command line
# ---------------------------------------------------------------------------

def test_defaults_with_empty_env_and_no_file(tmp_path):
    s = _load(tmp_path)
    assert s.config_file is None
    assert s["server.host"] == "127.0.0.1"
    assert s["server.port"] == 8080
    assert s["server.workers"] == "auto"
    assert s["mode"] == "local"
    assert s["fail_mode"] == "closed"
    assert s["state.backend"] == "auto"
    assert s["dashboard.enabled"] is True
    assert s["airgapped"] is False
    assert s["telemetry.heartbeat"] is False
    assert s["auth.require"] is None
    assert set(s.sources.values()) == {"default"}


def test_file_values_apply_and_are_marked_as_file(tmp_path):
    s = _load(tmp_path, "version: 1\nserver:\n  port: 9123\n  workers: 3\nfail_mode: open\n")
    assert s["server.port"] == 9123
    assert s["server.workers"] == 3
    assert s["fail_mode"] == "open"
    assert s.sources["server.port"] == "file"
    assert s.sources["server.host"] == "default"
    assert s.config_file == (tmp_path / "znyx.yaml").resolve()


def test_env_beats_file(tmp_path):
    s = _load(tmp_path, "server:\n  port: 9123\n", env={"PORT": "9200"})
    assert s["server.port"] == 9200
    assert s.sources["server.port"] == "env PORT"


def test_command_line_overrides_beat_env(tmp_path):
    s = _load(tmp_path, "server:\n  port: 9123\n", env={"PORT": "9200"},
              overrides={"server.port": 9300, "mode": None})
    assert s["server.port"] == 9300
    assert s.sources["server.port"] == "command line"
    assert s.sources["mode"] == "default"      # a flag that was not given changes nothing


def test_unknown_override_key_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="unknown setting"):
        _load(tmp_path, overrides={"nope": 1})


def test_legacy_env_names_are_honoured(tmp_path):
    s = _load(tmp_path, env={"GUARDRAILS_FAIL_MODE": "open",
                             "GUARDRAILS_POLICY_PATH": "/etc/p.yaml"})
    assert s["fail_mode"] == "open"
    assert s["policy"] == "/etc/p.yaml"
    assert s.sources["fail_mode"] == "env GUARDRAILS_FAIL_MODE"
    assert s.sources["policy"] == "env GUARDRAILS_POLICY_PATH"


def test_legacy_guardrails_mode_selects_managed(tmp_path):
    s = _load(tmp_path, env={"GUARDRAILS_MODE": "managed",
                             "GUARDRAILS_CONTROL_PLANE_URL": "https://cp",
                             "GUARDRAILS_RUNTIME_TOKEN": "tok-123456789"})  # gitleaks:allow
    assert s["mode"] == "managed"
    assert s.sources["mode"] == "env GUARDRAILS_MODE"


def test_canonical_env_name_beats_legacy_name(tmp_path):
    s = _load(tmp_path, env={"ZNYX_FAIL_MODE": "open", "GUARDRAILS_FAIL_MODE": "closed"})
    assert s["fail_mode"] == "open"
    assert s.sources["fail_mode"] == "env ZNYX_FAIL_MODE"


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def test_reference_to_environment_variable_is_resolved(tmp_path):
    s = _load(tmp_path, "policy: ${MY_POLICY}\n", env={"MY_POLICY": "/x/y.yaml"})
    assert s["policy"] == "/x/y.yaml"
    assert s.sources["policy"] == "file"


def test_reference_default_is_used_when_variable_is_missing(tmp_path):
    s = _load(tmp_path, "policy: ${MY_POLICY:-/fallback.yaml}\n")
    assert s["policy"] == "/fallback.yaml"


def test_reference_default_is_ignored_when_variable_is_set(tmp_path):
    s = _load(tmp_path, "policy: ${MY_POLICY:-/fallback.yaml}\n", env={"MY_POLICY": "/set.yaml"})
    assert s["policy"] == "/set.yaml"


def test_reference_can_be_embedded_in_longer_text(tmp_path):
    s = _load(tmp_path, "policy: /srv/${TEAM}/policies.yaml\n", env={"TEAM": "red"})
    assert s["policy"] == "/srv/red/policies.yaml"


def test_missing_reference_without_default_names_the_variable(tmp_path):
    with pytest.raises(ConfigError, match="NOPE_NOT_SET"):
        _load(tmp_path, "control_plane:\n  token: ${NOPE_NOT_SET}\n")


def test_reference_in_numeric_field_is_parsed_after_substitution(tmp_path):
    s = _load(tmp_path, 'server:\n  port: "${P:-9001}"\n')
    assert s["server.port"] == 9001


# ---------------------------------------------------------------------------
# Structure errors
# ---------------------------------------------------------------------------

def test_unknown_key_suggests_the_nearest_setting(tmp_path):
    with pytest.raises(ConfigError) as exc:
        _load(tmp_path, "server:\n  prot: 1\n")
    assert "unknown setting server.prot" in str(exc.value)
    assert "did you mean server.port" in str(exc.value)


def test_unknown_top_level_key_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="unknown setting"):
        _load(tmp_path, "zzzzzzzz: 1\n")


def test_unsupported_version_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="version 2"):
        _load(tmp_path, "version: 2\n")


def test_section_that_is_not_a_mapping_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="server"):
        _load(tmp_path, "server: 5\n")


def test_top_level_that_is_not_a_mapping_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="mapping"):
        _load(tmp_path, "- a\n- b\n")


def test_empty_file_and_empty_section_are_fine(tmp_path):
    s = _load(tmp_path, "server:\n")
    assert s["server.port"] == 8080


def test_invalid_yaml_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not valid YAML"):
        _load(tmp_path, "server: [unclosed\n")


# ---------------------------------------------------------------------------
# Value validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("env", [
    {"PORT": "0"},
    {"PORT": "70000"},
    {"PORT": "http"},
    {"ZNYX_WORKERS": "0"},
    {"ZNYX_WORKERS": "many"},
    {"RUNTIME_REQUIRE_AUTH": "maybe"},
    {"ZNYX_MODE": "cloud"},
    {"ZNYX_STATE_BACKEND": "postgres"},
    {"ZNYX_GRACEFUL_TIMEOUT": "-1"},
], ids=["port-0", "port-70000", "port-text", "workers-0", "workers-many", "bool-maybe",
        "mode-cloud", "backend-postgres", "negative-timeout"])
def test_invalid_values_are_rejected(tmp_path, env):
    with pytest.raises(ConfigError):
        _load(tmp_path, env=env)


@pytest.mark.parametrize("flag,value", [("server.port", 0), ("server.port", 70000),
                                        ("server.workers", "0")])
def test_invalid_command_line_values_are_rejected(tmp_path, flag, value):
    with pytest.raises(ConfigError, match="command line"):
        _load(tmp_path, overrides={flag: value})


def test_error_message_names_the_setting_and_where_it_came_from(tmp_path):
    with pytest.raises(ConfigError) as exc:
        _load(tmp_path, env={"PORT": "0"})
    assert "server.port" in str(exc.value)
    assert "PORT" in str(exc.value)


@pytest.mark.parametrize("text,expected", [
    ("yes", True), ("ON", True), ("1", True), ("false", False), ("No", False), ("0", False),
])
def test_booleans_accept_the_usual_spellings(tmp_path, text, expected):
    assert _load(tmp_path, env={"ZNYX_DASHBOARD": text})["dashboard.enabled"] is expected


def test_yaml_boolean_is_not_accepted_as_a_port(tmp_path):
    with pytest.raises(ConfigError, match="expected a number"):
        _load(tmp_path, "server:\n  port: true\n")


def test_workers_auto_is_case_insensitive(tmp_path):
    assert _load(tmp_path, env={"ZNYX_WORKERS": "AUTO"})["server.workers"] == "auto"


# ---------------------------------------------------------------------------
# Cross-field validation
# ---------------------------------------------------------------------------

def test_managed_mode_needs_url_and_token(tmp_path):
    with pytest.raises(ConfigError, match="control_plane.url"):
        _load(tmp_path, env={"ZNYX_MODE": "managed", "ZNYX_RUNTIME_TOKEN": "t"})
    with pytest.raises(ConfigError, match="control_plane.token"):
        _load(tmp_path, env={"ZNYX_MODE": "managed", "ZNYX_CONTROL_PLANE_URL": "https://cp"})


def test_managed_mode_with_url_and_token_is_valid(tmp_path):
    s = _load(tmp_path, env={"ZNYX_MODE": "managed", "ZNYX_CONTROL_PLANE_URL": "https://cp",
                             "ZNYX_RUNTIME_TOKEN": "rt_abcdefghij"})
    assert s["mode"] == "managed"


def test_redis_backend_needs_a_url(tmp_path):
    with pytest.raises(ConfigError, match="redis_url"):
        _load(tmp_path, env={"ZNYX_STATE_BACKEND": "redis"})
    s = _load(tmp_path, env={"ZNYX_STATE_BACKEND": "redis", "ZNYX_REDIS_URL": "redis://h:6379/0"})
    assert s["state.backend"] == "redis"


def test_ml_enabled_and_ml_url_are_mutually_exclusive(tmp_path):
    with pytest.raises(ConfigError, match="not both"):
        _load(tmp_path, env={"ZNYX_ML": "true", "ZNYX_INFERENCE_URL": "http://ml:9000"})


def test_ml_port_must_differ_from_server_port(tmp_path):
    with pytest.raises(ConfigError, match="ml.port"):
        _load(tmp_path, env={"ZNYX_ML": "true", "ZNYX_ML_PORT": "8080"})


def test_airgapped_contradicts_heartbeat(tmp_path):
    with pytest.raises(ConfigError, match="contradict"):
        _load(tmp_path, env={"ZNYX_AIRGAPPED": "true", "ZNYX_TELEMETRY": "true"})


def test_airgapped_with_heartbeat_off_is_valid(tmp_path):
    assert _load(tmp_path, env={"ZNYX_AIRGAPPED": "true"})["airgapped"] is True


# ---------------------------------------------------------------------------
# Finding the config file
# ---------------------------------------------------------------------------

def test_explicit_config_path_must_exist(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        _load(tmp_path, config_path=str(tmp_path / "missing.yaml"))


def test_znyx_config_env_var_is_used(tmp_path):
    custom = tmp_path / "elsewhere.yaml"
    custom.write_text("server:\n  port: 9555\n")
    s = _load(tmp_path, env={"ZNYX_CONFIG": str(custom)})
    assert s["server.port"] == 9555
    assert s.config_file == custom.resolve()


def test_znyx_config_env_var_pointing_nowhere_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="ZNYX_CONFIG"):
        _load(tmp_path, env={"ZNYX_CONFIG": str(tmp_path / "nope.yaml")})


def test_explicit_path_beats_env_var_and_cwd_file(tmp_path):
    (tmp_path / "znyx.yaml").write_text("server:\n  port: 9001\n")
    env_file = tmp_path / "env.yaml"
    env_file.write_text("server:\n  port: 9002\n")
    flag_file = tmp_path / "flag.yaml"
    flag_file.write_text("server:\n  port: 9003\n")
    s = load_settings(str(flag_file), env={"ZNYX_CONFIG": str(env_file)}, cwd=tmp_path)
    assert s["server.port"] == 9003


def test_znyx_yaml_in_cwd_is_picked_up(tmp_path):
    s = _load(tmp_path, "server:\n  port: 9444\n")
    assert s["server.port"] == 9444
    assert s.config_file == (tmp_path / "znyx.yaml").resolve()


def test_find_config_file_returns_none_without_a_file(tmp_path):
    assert zc.find_config_file(None, {}, tmp_path) is None


# ---------------------------------------------------------------------------
# worker_count and the CPU quota
# ---------------------------------------------------------------------------

def test_worker_count_explicit_number(tmp_path):
    assert zc.worker_count(_load(tmp_path, env={"ZNYX_WORKERS": "3"})) == 3


@pytest.mark.parametrize("cpus,expected", [(1, 1), (2, 2), (4, 4), (16, zc.MAX_AUTO_WORKERS)])
def test_worker_count_auto_is_cpus_capped(tmp_path, monkeypatch, cpus, expected):
    monkeypatch.setattr(zc, "available_cpus", lambda: cpus)
    assert zc.worker_count(_load(tmp_path)) == expected == min(cpus, zc.MAX_AUTO_WORKERS)


def test_cgroup_v2_quota(tmp_path):
    (tmp_path / "cpu.max").write_text("200000 100000\n")
    assert zc.cgroup_cpu_quota(tmp_path) == 2


def test_cgroup_v2_unlimited(tmp_path):
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert zc.cgroup_cpu_quota(tmp_path) is None


def test_cgroup_v2_fractional_quota_rounds_up_to_one(tmp_path):
    (tmp_path / "cpu.max").write_text("50000 100000\n")
    assert zc.cgroup_cpu_quota(tmp_path) == 1


def test_cgroup_v2_quota_rounds_up(tmp_path):
    (tmp_path / "cpu.max").write_text("250000 100000\n")
    assert zc.cgroup_cpu_quota(tmp_path) == 3


def test_cgroup_v1_quota(tmp_path):
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("300000\n")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    assert zc.cgroup_cpu_quota(tmp_path) == 3


def test_cgroup_v1_unlimited_is_minus_one(tmp_path):
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("-1\n")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    assert zc.cgroup_cpu_quota(tmp_path) is None


def test_cgroup_files_missing_or_garbled_mean_no_quota(tmp_path):
    assert zc.cgroup_cpu_quota(tmp_path) is None
    (tmp_path / "cpu.max").write_text("garbage\n")
    assert zc.cgroup_cpu_quota(tmp_path) is None


def test_available_cpus_honours_the_quota(monkeypatch):
    monkeypatch.setattr(zc, "cgroup_cpu_quota", lambda *a, **k: 1)
    assert zc.available_cpus() == 1


def test_available_cpus_is_at_least_one(monkeypatch):
    monkeypatch.setattr(zc, "cgroup_cpu_quota", lambda *a, **k: None)
    assert zc.available_cpus() >= 1


# ---------------------------------------------------------------------------
# State backend
# ---------------------------------------------------------------------------

def test_state_backend_auto_follows_the_worker_count(tmp_path):
    s = _load(tmp_path)
    assert zc.state_backend(s, 2) == "sqlite"
    assert zc.state_backend(s, 1) == "memory"
    assert s.notes == []


def test_state_backend_explicit_memory_with_many_workers_adds_a_note(tmp_path):
    s = _load(tmp_path, env={"ZNYX_STATE_BACKEND": "memory"})
    assert zc.state_backend(s, 3) == "memory"
    assert len(s.notes) == 1
    assert "3 workers" in s.notes[0]


def test_state_backend_explicit_memory_with_one_worker_is_quiet(tmp_path):
    s = _load(tmp_path, env={"ZNYX_STATE_BACKEND": "memory"})
    assert zc.state_backend(s, 1) == "memory"
    assert s.notes == []


# ---------------------------------------------------------------------------
# service_env: what the children see
# ---------------------------------------------------------------------------

def test_service_env_uses_canonical_names(tmp_path):
    s = _load(tmp_path, env={"GUARDRAILS_MODE": "local", "GUARDRAILS_FAIL_MODE": "open",
                             "RUNTIME_REQUIRE_AUTH": "no"})
    env = _service_env(s)
    assert env["ZNYX_MODE"] == "local"
    assert env["ZNYX_FAIL_MODE"] == "open"
    assert env["RUNTIME_REQUIRE_AUTH"] == "false"
    assert "GUARDRAILS_MODE" not in env and "GUARDRAILS_FAIL_MODE" not in env
    assert env["HOST"] == "127.0.0.1" and env["PORT"] == "8080"


def test_service_env_booleans_are_true_or_false_text(tmp_path):
    env = _service_env(_load(tmp_path, env={"RUNTIME_REQUIRE_AUTH": "1"}))
    assert env["RUNTIME_REQUIRE_AUTH"] == "true"
    assert env["ZNYX_DASHBOARD"] == "true"


def test_service_env_leaves_unset_optional_values_out(tmp_path):
    env = _service_env(_load(tmp_path))
    assert "RUNTIME_REQUIRE_AUTH" not in env        # None: the runtime's own default applies
    assert "ZNYX_CONTROL_PLANE_URL" not in env      # empty


def test_service_env_does_not_export_supervisor_only_keys(tmp_path):
    s = _load(tmp_path, env={"ZNYX_WORKERS": "2", "ZNYX_ML": "true", "ZNYX_GRACEFUL_TIMEOUT": "5",
                             "ZNYX_WORKER_TIMEOUT": "7", "ZNYX_LOG_LEVEL": "debug",
                             "ZNYX_API_KEY_FILE": "/k"})
    env = _service_env(s, workers=2)
    for name in ("ZNYX_WORKERS", "ZNYX_ML", "ZNYX_ML_PORT", "ZNYX_GRACEFUL_TIMEOUT",
                 "ZNYX_WORKER_TIMEOUT", "ZNYX_LOG_LEVEL", "ZNYX_API_KEY_FILE"):
        assert name not in env, name


def test_service_env_sqlite_path_is_under_the_run_dir(tmp_path):
    env = _service_env(_load(tmp_path), workers=2, run_dir=tmp_path / "run")
    assert env["ZNYX_STATE_BACKEND"] == "sqlite"
    assert env["ZNYX_STATE_SQLITE_PATH"] == str(tmp_path / "run" / "state.sqlite3")


def test_service_env_single_worker_uses_memory_and_no_sqlite_path(tmp_path):
    env = _service_env(_load(tmp_path), workers=1)
    assert env["ZNYX_STATE_BACKEND"] == "memory"
    assert "ZNYX_STATE_SQLITE_PATH" not in env


def test_service_env_inference_url_points_at_the_local_model_server(tmp_path):
    env = _service_env(_load(tmp_path, env={"ZNYX_ML": "true", "ZNYX_ML_PORT": "9123"}))
    assert env["ZNYX_INFERENCE_URL"] == "http://127.0.0.1:9123"


def test_service_env_inference_url_is_the_configured_remote_one(tmp_path):
    env = _service_env(_load(tmp_path, env={"ZNYX_INFERENCE_URL": "http://ml.internal:9000"}))
    assert env["ZNYX_INFERENCE_URL"] == "http://ml.internal:9000"


def test_service_env_without_ml_has_no_inference_url(tmp_path):
    assert "ZNYX_INFERENCE_URL" not in _service_env(_load(tmp_path))


def test_service_env_airgapped_adds_offline_switches(tmp_path):
    env = _service_env(_load(tmp_path, env={"ZNYX_AIRGAPPED": "true"}))
    assert env["ZNYX_AIRGAPPED"] == "true"
    assert env["ZNYX_TELEMETRY"] == "false"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"


def test_service_env_not_airgapped_has_no_offline_switches(tmp_path):
    env = _service_env(_load(tmp_path))
    assert "HF_HUB_OFFLINE" not in env
    assert env["ZNYX_AIRGAPPED"] == "false"


def test_service_env_always_carries_the_supervisor_layout(tmp_path):
    env = _service_env(_load(tmp_path), workers=2, run_dir=tmp_path / "r")
    assert env["ZNYX_SKIP_DOTENV"] == "1"
    assert env["ZNYX_RUN_DIR"] == str(tmp_path / "r")
    assert env["ZNYX_WORKER_COUNT"] == "2"
    assert env["ZNYX_PLATFORM_VERSION"] == "9.9.9"
    assert env["ZNYX_POLICY_PATH"] == "/p/policies.yaml"


def test_service_env_without_a_policy_path_exports_none(tmp_path):
    env = zc.service_env(_load(tmp_path), 1, policy_path=None, run_directory=tmp_path,
                         platform_version="1")
    assert "ZNYX_POLICY_PATH" not in env


# ---------------------------------------------------------------------------
# describe: secrets are masked
# ---------------------------------------------------------------------------

def test_describe_masks_every_secret(tmp_path):
    token, api_key, redis = "rt_supersecrettoken123", "zx_supersecretapikey456", "redis://:pw-9876543210@h/0"
    s = _load(tmp_path, env={
        "ZNYX_MODE": "managed", "ZNYX_CONTROL_PLANE_URL": "https://cp", "ZNYX_RUNTIME_TOKEN": token,
        "RUNTIME_API_KEY": api_key, "ZNYX_REDIS_URL": redis})
    rows = {key: shown for key, shown, _ in zc.describe(s)}
    for key, secret in (("control_plane.token", token), ("auth.api_key", api_key),
                        ("state.redis_url", redis)):
        assert rows[key] != secret
        assert rows[key].endswith("***")
        assert secret not in rows[key]
    blob = repr(zc.describe(s))
    for secret in (token, api_key, redis):
        assert secret not in blob


def test_describe_fully_masks_short_secrets(tmp_path):
    s = _load(tmp_path, env={"RUNTIME_API_KEY": "abc"})
    assert {k: v for k, v, _ in zc.describe(s)}["auth.api_key"] == "***"


def test_describe_shows_unset_values_and_sources(tmp_path):
    rows = {k: (v, src) for k, v, src in zc.describe(_load(tmp_path, env={"PORT": "9"}))}
    assert rows["control_plane.url"] == ("(not set)", "default")
    assert rows["server.port"] == ("9", "env PORT")
    assert rows["dashboard.enabled"][0] == "true"


# ---------------------------------------------------------------------------
# Policy resolution
# ---------------------------------------------------------------------------

def test_explicit_policy_is_returned_as_given_even_when_missing(tmp_path):
    s = _load(tmp_path, env={"ZNYX_POLICY_PATH": "/does/not/exist.yaml"})
    assert zc.resolve_policy(s, tmp_path) == "/does/not/exist.yaml"
    assert s.notes == []


def test_policy_in_cwd_is_found(tmp_path):
    (tmp_path / "policies.yaml").write_text("default: {}\n")
    s = _load(tmp_path)
    assert zc.resolve_policy(s, tmp_path) == str((tmp_path / "policies.yaml").resolve())
    assert s.notes == []


def test_policy_in_config_dir_is_found(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "policies.yaml").write_text("default: {}\n")
    s = _load(tmp_path)
    assert zc.resolve_policy(s, tmp_path) == str((tmp_path / "config" / "policies.yaml").resolve())


def test_cwd_policy_beats_config_dir_policy(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "policies.yaml").write_text("a: 1\n")
    (tmp_path / "policies.yaml").write_text("b: 1\n")
    resolved = zc.resolve_policy(_load(tmp_path), tmp_path)
    assert resolved == str((tmp_path / "policies.yaml").resolve())


def test_policy_falls_back_to_the_packaged_starter_with_a_note(tmp_path):
    s = _load(tmp_path)
    resolved = zc.resolve_policy(s, tmp_path)
    assert Path(resolved).is_file()
    assert Path(resolved) == zc.starter_policy_path()
    assert len(s.notes) == 1 and "starter" in s.notes[0]


def test_managed_mode_returns_the_configured_policy_or_none(tmp_path):
    managed = {"ZNYX_MODE": "managed", "ZNYX_CONTROL_PLANE_URL": "https://cp",
               "ZNYX_RUNTIME_TOKEN": "rt_abcdefghij"}
    assert zc.resolve_policy(_load(tmp_path, env=managed), tmp_path) is None
    configured = _load(tmp_path, env={**managed, "ZNYX_POLICY_PATH": "/p.yaml"})
    assert zc.resolve_policy(configured, tmp_path) == "/p.yaml"


# ---------------------------------------------------------------------------
# The API key
# ---------------------------------------------------------------------------

@pytest.fixture()
def keyed(tmp_path):
    """Settings whose generated key lives under tmp_path, never the real home."""
    key_file = tmp_path / "home" / ".znyx" / "api-key"
    return _load(tmp_path, env={"ZNYX_API_KEY_FILE": str(key_file)}), key_file


def test_api_key_path_follows_the_setting(keyed):
    settings, key_file = keyed
    assert zc.api_key_path(settings) == key_file


def test_api_key_path_defaults_under_the_home_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert zc.api_key_path(_load(tmp_path)) == tmp_path / ".znyx" / "api-key"


def test_ensure_api_key_creates_a_private_key_file(keyed):
    settings, key_file = keyed
    key, path, created = zc.ensure_api_key(settings)
    assert created is True
    assert path == key_file
    assert key.startswith("zx_") and len(key) > 20
    assert key_file.read_text().strip() == key
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600


def test_ensure_api_key_reuses_the_existing_key(keyed):
    settings, _ = keyed
    first, _, _ = zc.ensure_api_key(settings)
    second, _, created = zc.ensure_api_key(settings)
    assert second == first
    assert created is False


def test_ensure_api_key_rotate_makes_a_new_key_and_keeps_the_mode(keyed):
    settings, key_file = keyed
    first, _, _ = zc.ensure_api_key(settings)
    second, _, created = zc.ensure_api_key(settings, rotate=True)
    assert created is True
    assert second != first
    assert key_file.read_text().strip() == second
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600


def test_ensure_api_key_regenerates_when_the_file_is_empty(keyed):
    settings, key_file = keyed
    key_file.parent.mkdir(parents=True)
    key_file.write_text("\n")
    key, _, created = zc.ensure_api_key(settings)
    assert created is True and key.startswith("zx_")


def test_ensure_api_key_does_not_loosen_an_existing_file(keyed):
    settings, key_file = keyed
    key_file.parent.mkdir(parents=True)
    key_file.write_text("\n")
    os.chmod(key_file, 0o644)
    zc.ensure_api_key(settings)
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600


def test_configured_api_key_wins_and_is_not_written(tmp_path):
    key_file = tmp_path / "api-key"
    s = _load(tmp_path, env={"RUNTIME_API_KEY": "from-the-env", "ZNYX_API_KEY_FILE": str(key_file)})
    key, path, created = zc.ensure_api_key(s)
    assert (key, created) == ("from-the-env", False)
    assert path == key_file
    assert not key_file.exists()


def test_needs_api_key_is_true_by_default(tmp_path):
    assert zc.needs_api_key(_load(tmp_path), env={}) is True


def test_needs_api_key_is_true_in_production_even_with_auth_off(tmp_path):
    s = _load(tmp_path, env={"RUNTIME_REQUIRE_AUTH": "false"})
    assert zc.needs_api_key(s, env={}) is True
    assert zc.needs_api_key(s, env={"ZNYX_ENVIRONMENT": "production"}) is True


def test_needs_api_key_is_false_only_with_auth_off_outside_production(tmp_path):
    off = _load(tmp_path, env={"RUNTIME_REQUIRE_AUTH": "false"})
    assert zc.needs_api_key(off, env={"ZNYX_ENVIRONMENT": "development"}) is False
    assert zc.needs_api_key(off, env={"ENVIRONMENT": "test"}) is False
    on = _load(tmp_path, env={"RUNTIME_REQUIRE_AUTH": "true"})
    assert zc.needs_api_key(on, env={"ZNYX_ENVIRONMENT": "development"}) is True
    unset = _load(tmp_path)
    assert zc.needs_api_key(unset, env={"ZNYX_ENVIRONMENT": "development"}) is True


def test_needs_api_key_reads_os_environ_when_no_env_is_given(tmp_path, monkeypatch):
    off = _load(tmp_path, env={"RUNTIME_REQUIRE_AUTH": "false"})
    for var in ("ZNYX_ENV", "ZNYX_ENVIRONMENT", "ENVIRONMENT", "GUARDRAILS_ENVIRONMENT"):
        monkeypatch.delenv(var, raising=False)
    assert zc.needs_api_key(off) is True
    monkeypatch.setenv("ZNYX_ENVIRONMENT", "development")
    assert zc.needs_api_key(off) is False


def test_rotate_never_replaces_a_configured_key(tmp_path):
    """A key the operator configured (auth.api_key / RUNTIME_API_KEY) is theirs: rotating
    the generated key file must not swap in a new key behind it."""
    from znyx.config import ensure_api_key, load_settings
    key_file = tmp_path / "api-key"
    settings = load_settings(env={"RUNTIME_API_KEY": "zx_operator_key", "ZNYX_API_KEY_FILE": str(key_file)},
                             cwd=tmp_path)
    key, path, created = ensure_api_key(settings, rotate=True)
    assert (key, created) == ("zx_operator_key", False)
    assert not key_file.exists()



# -- image defaults (ZNYX_DEFAULTS) ------------------------------------------------

def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_image_defaults_sit_below_the_config_file_and_the_environment(tmp_path):
    """The container image listens on 0.0.0.0 by default, but a mounted znyx.yaml (and
    the environment) must win: an image must never override what the operator set."""
    from znyx.config import load_settings
    defaults = _write(tmp_path / "defaults.yaml", "server: {host: 0.0.0.0, port: 8080}\nmode: local\n"
                                                  "ml: {enabled: true}\n")
    user = _write(tmp_path / "znyx.yaml", "mode: managed\ncontrol_plane: {url: https://cp, token: t}\n"
                                          "server: {port: 9090}\n")
    settings = load_settings(str(user), env={"ZNYX_DEFAULTS": str(defaults), "ZNYX_ML": "false"},
                             cwd=tmp_path)
    assert settings["mode"] == "managed" and settings.sources["mode"] == "file"
    assert settings["server.port"] == 9090
    assert settings["server.host"] == "0.0.0.0" and settings.sources["server.host"] == "image defaults"
    assert settings["ml.enabled"] is False and settings.sources["ml.enabled"] == "env ZNYX_ML"


def test_image_defaults_alone_apply_over_built_in_defaults(tmp_path):
    from znyx.config import load_settings
    defaults = _write(tmp_path / "defaults.yaml", "server: {host: 0.0.0.0}\n")
    settings = load_settings(env={"ZNYX_DEFAULTS": str(defaults)}, cwd=tmp_path)
    assert settings["server.host"] == "0.0.0.0"
    assert settings["server.port"] == 8080 and settings.sources["server.port"] == "default"


def test_image_defaults_are_checked_like_a_config_file(tmp_path):
    import pytest
    from znyx.config import ConfigError, load_settings
    with pytest.raises(ConfigError, match="missing file"):
        load_settings(env={"ZNYX_DEFAULTS": str(tmp_path / "nope.yaml")}, cwd=tmp_path)
    bad = _write(tmp_path / "bad.yaml", "server: {prot: 1}\n")
    with pytest.raises(ConfigError, match="did you mean server.port"):
        load_settings(env={"ZNYX_DEFAULTS": str(bad)}, cwd=tmp_path)
