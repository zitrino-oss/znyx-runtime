"""Phase 5, part 1: shadow-AI discovery (``znyx discover``) and the AI services catalogue."""
import gzip
import json

import pytest

from znyx import cli, discover
from znyx_core.ai_services import match_host


@pytest.mark.parametrize("host,service", [
    ("api.openai.com", "OpenAI API"), ("API.OpenAI.com.", "OpenAI API"), ("api.openai.com:443", "OpenAI API"),
    ("myco.openai.azure.com", "Azure OpenAI"), ("chatgpt.com", "ChatGPT"), ("cdn.chatgpt.com", "ChatGPT"),
    ("claude.ai", "Claude"), ("api.anthropic.com", "Anthropic API"),
    ("bedrock-runtime.eu-west-1.amazonaws.com", "Amazon Bedrock"), ("api.perplexity.ai", "Perplexity API"),
    ("www.perplexity.ai", "Perplexity"), ("api-inference.huggingface.co", "Hugging Face Inference"),
    ("huggingface.co", "Hugging Face"), ("europe-west4-aiplatform.googleapis.com", "Vertex AI"),
    ("openai.com", None), ("notclaude.ai", None), ("s3.amazonaws.com", None), ("", None), (None, None),
])
def test_hosts_are_matched_to_services(host, service):
    found = match_host(host)
    assert (found.name if found else None) == service


def _run(tmp_path, name, text, *extra, compress=False):
    path = tmp_path / name
    if compress:
        path.write_bytes(gzip.compress(text.encode()))
    else:
        path.write_text(text)
    out = tmp_path / "report.json"
    with pytest.raises(SystemExit) as exc:
        cli.main(["discover", str(path), "-o", str(out), *extra])
    assert exc.value.code == 0
    return json.loads(out.read_text())


SQUID = """1729500000.123    45 10.0.0.5 TCP_TUNNEL/200 3456 CONNECT chatgpt.com:443 alice HIER_DIRECT/1.2.3.4 -
1729500100.000    12 10.0.0.6 TCP_TUNNEL/200 120 CONNECT chatgpt.com:443 bob HIER_DIRECT/1.2.3.4 -
1729500200.000    12 10.0.0.6 TCP_TUNNEL/200 120 CONNECT chatgpt.com:443 bob HIER_DIRECT/1.2.3.4 -
1729500300.000    30 10.0.0.9 TCP_MISS/200 999 POST https://api.openai.com/v1/chat/completions - HIER_DIRECT/5.6.7.8 application/json
1729500400.000    30 10.0.0.2 TCP_TUNNEL/200 10 CONNECT intranet.example:443 carol HIER_DIRECT/9.9.9.9 -
"""


def test_squid_logs_are_read_and_users_pseudonymised(tmp_path):
    report = _run(tmp_path, "access.log", SQUID)
    assert report["source"] == "access.log" and report["lines_read"] == 5 and report["users_pseudonymised"]
    chat, api = report["findings"]
    assert (chat["service"], chat["users"], chat["requests"]) == ("ChatGPT", 2, 3)
    assert all(u["user"].startswith("u_") for u in chat["top_users"])
    assert "alice" not in json.dumps(report) and "bob" not in json.dumps(report)
    assert (api["service"], api["users"], api["category"]) == ("OpenAI API", 1, "api")    # the client as the user
    assert report["period_start"].startswith("2024-10-21") and chat["first_seen"] <= chat["last_seen"]


def test_names_kept_on_request_and_the_gateway_left_out(tmp_path):
    report = _run(tmp_path, "access.log", SQUID, "--keep-user-names", "--exclude-source", "10.0.0.9")
    assert [f["service"] for f in report["findings"]] == ["ChatGPT"]
    assert {u["user"] for u in report["findings"][0]["top_users"]} == {"alice", "bob"}
    assert report["users_pseudonymised"] is False


def test_the_same_user_is_the_same_pseudonym_across_reports(tmp_path):
    a = _run(tmp_path, "a.log", SQUID)
    b = _run(tmp_path, "b.log", SQUID)
    assert a["findings"][0]["top_users"] == b["findings"][0]["top_users"]


def test_w3c_logs(tmp_path):
    text = ("#Software: proxy\n#Fields: date time c-ip cs-username cs-host cs-uri-stem sc-status\n"
            "2026-10-21 09:00:00 10.1.1.1 dave claude.ai /chat 200\n"
            "2026-10-21 09:05:00 10.1.1.1 dave example.org / 200\n")
    report = _run(tmp_path, "proxy.log", text)
    assert [(f["service"], f["users"]) for f in report["findings"]] == [("Claude", 1)]
    assert report["findings"][0]["first_seen"] == "2026-10-21T09:00:00Z"


def test_zeek_tsv_and_json(tmp_path):
    tsv = ("#separator \\x09\n#fields\tts\tuid\tid.orig_h\tquery\n"
           "1729500000.1\tC1\t10.2.0.1\tgemini.google.com\n1729500001.1\tC2\t10.2.0.2\tgemini.google.com\n")
    report = _run(tmp_path, "dns.log", tsv)
    assert [(f["service"], f["users"], f["requests"]) for f in report["findings"]] == [("Gemini", 2, 2)]
    lines = "\n".join(json.dumps(r) for r in (
        {"ts": 1729500000.5, "id.orig_h": "10.2.0.3", "server_name": "api.mistral.ai"},
        {"ts": 1729500000.6, "id": {"orig_h": "10.2.0.3"}, "server_name": "api.mistral.ai"}))
    report = _run(tmp_path, "ssl.json", lines + "\n")
    assert [(f["service"], f["users"]) for f in report["findings"]] == [("Mistral API", 1)]


def test_csv_with_a_column_mapping_and_gzip(tmp_path):
    text = "when,who,where\n2026-10-20T10:00:00Z,erin,https://grok.com/chat\n2026-10-20T11:00:00Z,frank,grok.com\n"
    report = _run(tmp_path, "export.csv.gz", text, "--host-column", "where", "--user-column", "who",
                  "--time-column", "when", "--source", "swg", compress=True)
    assert report["source"] == "swg"
    assert [(f["service"], f["users"], f["requests"]) for f in report["findings"]] == [("Grok", 2, 2)]
    assert report["period_end"] == "2026-10-20T11:00:00Z"


def test_upload_needs_where_and_who(tmp_path, monkeypatch, capsys):
    for name in ("ZNYX_CONTROL_PLANE_URL", "ZNYX_ORG_ID", "ZNYX_CONSOLE_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "access.log"
    path.write_text(SQUID)
    with pytest.raises(SystemExit) as exc:
        cli.main(["discover", str(path), "--upload"])
    assert exc.value.code == 2 and "--control-plane" in capsys.readouterr().err


def test_upload_sends_only_the_aggregates(tmp_path, monkeypatch):
    sent = {}

    def fake_upload(report, control_plane, org_id, token, timeout=30.0):
        sent.update(report=report, url=control_plane, org=org_id, token=token)
        return {"id": "r1", "source": report["source"], "findings": len(report["findings"])}

    monkeypatch.setattr(discover, "upload", fake_upload)
    path = tmp_path / "access.log"
    path.write_text(SQUID)
    with pytest.raises(SystemExit) as exc:
        cli.main(["discover", str(path), "--upload", "--control-plane", "https://console.example",
                  "--org", "o1", "--token", "t"])
    assert exc.value.code == 0 and sent["org"] == "o1"
    assert "CONNECT" not in json.dumps(sent["report"]) and "intranet" not in json.dumps(sent["report"])


def test_a_missing_file_is_reported(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["discover", str(tmp_path / "nope.log")])
    assert exc.value.code == 2 and "no such file" in capsys.readouterr().err


def test_hostile_logs_never_stop_a_run_nor_leak_a_user_name(tmp_path):
    csv_text = ('host,user\n"chatgpt.com,alice\n' + "claude.ai,bob\n" + "x" * 200_000 + ",carol\n"
                + "claude.ai,dave\n")
    report = _run(tmp_path, "bad.csv", csv_text)
    assert [(f["service"], f["users"]) for f in report["findings"]] == [("Claude", 2)]
    deep = "{" * 5000 + "}" * 5000
    report = _run(tmp_path, "deep.json", deep + "\n" + json.dumps({"host": "grok.com", "user": "x"}) + "\n")
    assert [f["service"] for f in report["findings"]] == ["Grok"]
    report = _run(tmp_path, "z.log", "#separator \\\n#fields\tts\tquery\n1729500000\tgemini.google.com\n")
    assert [f["service"] for f in report["findings"]] == ["Gemini"]
    report = _run(tmp_path, "s.log", "1729500000.1 1 10.0.0.1 TCP/200 1 GET alice.smith@acme.openai.azure.com/x - H -\n")
    assert [f["host"] for f in report["findings"]] == ["acme.openai.azure.com"] and "alice" not in json.dumps(report)


@pytest.mark.parametrize("value,host", [
    ("https://user:pw@api.openai.com/v1", "api.openai.com"), ("bob@claude.ai:443", "claude.ai"),
    ("claude.ai/chat?x=1", "claude.ai"), ("<script>", None), ("a b.com", None), ("-", None), ("x" * 300, None),
])
def test_only_plain_host_names_are_read(value, host):
    assert discover._host_of(value) == host
