// Znyx local dashboard. Reads /v1/local/summary every few seconds. Every value is
// written with textContent, never as HTML.
(function () {
  "use strict";

  var KEY = "znyx_api_key";
  var REFRESH_MS = 5000;
  var timer = null;
  var $ = function (id) { return document.getElementById(id); };

  // A key in the URL fragment (#key=...) is taken once and removed from the address
  // bar. The fragment is never sent to the server.
  (function takeKeyFromFragment() {
    var m = /(?:^|[#&])key=([^&]+)/.exec(location.hash || "");
    if (m) {
      try { sessionStorage.setItem(KEY, decodeURIComponent(m[1])); } catch (e) { /* private mode */ }
      history.replaceState(null, "", location.pathname + location.search);
    }
  })();

  function key() {
    try { return sessionStorage.getItem(KEY) || ""; } catch (e) { return ""; }
  }

  function headers(extra) {
    var h = extra || {};
    var k = key();
    if (k) { h["X-API-Key"] = k; }
    return h;
  }

  function el(tag, text, cls) {
    var node = document.createElement(tag);
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    if (cls) { node.className = cls; }
    return node;
  }

  function clear(node) { while (node.firstChild) { node.removeChild(node.firstChild); } }

  function fmt(n) { return n === null || n === undefined ? "-" : Number(n).toLocaleString(); }

  function pct(part, whole) {
    if (!whole) { return "0%"; }
    var p = (100 * part) / whole;
    return (p < 1 && p > 0 ? p.toFixed(1) : Math.round(p)) + "%";
  }

  function duration(s) {
    if (s === null || s === undefined) { return "-"; }
    s = Math.round(s);
    if (s < 60) { return s + "s"; }
    if (s < 3600) { return Math.floor(s / 60) + "m"; }
    if (s < 86400) { return Math.floor(s / 3600) + "h " + Math.floor((s % 3600) / 60) + "m"; }
    return Math.floor(s / 86400) + "d " + Math.floor((s % 86400) / 3600) + "h";
  }

  function decisionClass(d) { return "pill d-" + String(d || "").toLowerCase(); }

  function showSignin(message) {
    $("content").hidden = true;
    $("signin").hidden = false;
    $("signout").hidden = true;
    $("signin-error").textContent = message || "";
    $("key").focus();
  }

  function render(data) {
    $("signin").hidden = true;
    $("content").hidden = false;
    $("signout").hidden = !key();

    var meta = $("meta");
    clear(meta);
    meta.appendChild(el("span", "v" + data.version, "tag"));
    meta.appendChild(el("span", data.mode === "managed" ? "managed" : "local policy", "tag"));
    meta.appendChild(el("span", data.workers.count + (data.workers.count === 1 ? " worker" : " workers"), "tag"));
    meta.appendChild(el("span", "up " + duration(data.uptime_seconds), "tag"));
    if (data.airgapped) { meta.appendChild(el("span", "air-gapped", "tag strong")); }

    var total = data.evaluations || 0;
    var d = data.decisions || {};
    var blocked = d.BLOCK || 0;
    var changed = (d.REDACT || 0) + (d.TRANSFORM || 0);
    $("s-evals").textContent = fmt(total);
    $("s-evals-sub").textContent = Object.keys(data.contexts || {}).map(function (c) {
      return c + " " + fmt(data.contexts[c]);
    }).join(", ");
    $("s-blocked").textContent = fmt(blocked);
    $("s-blocked-sub").textContent = pct(blocked, total) + " of evaluations";
    $("s-changed").textContent = fmt(changed);
    $("s-changed-sub").textContent = pct(changed, total) + " of evaluations";
    var lat = data.latency_ms || {};
    $("s-p95").textContent = lat.p95 === null || lat.p95 === undefined ? "-" : "≤ " + lat.p95 + " ms";
    $("s-lat-sub").textContent = lat.mean === null || lat.mean === undefined ? "" :
      "mean " + lat.mean + " ms, p50 ≤ " + lat.p50 + " ms";

    var bars = $("decisions");
    clear(bars);
    ["ALLOW", "WARN", "TRANSFORM", "REDACT", "BLOCK"].forEach(function (name) {
      var n = d[name] || 0;
      var row = el("div", null, "bar-row");
      row.appendChild(el("span", name, "bar-label"));
      var track = el("span", null, "bar-track");
      var fill = el("span", null, "bar-fill d-" + name.toLowerCase());
      fill.style.width = total ? Math.max(n ? 2 : 0, (100 * n) / total) + "%" : "0";
      track.appendChild(fill);
      row.appendChild(track);
      row.appendChild(el("span", fmt(n), "bar-value"));
      bars.appendChild(row);
    });

    var rules = $("rules");
    clear(rules);
    (data.top_rules || []).forEach(function (r) {
      var tr = el("tr");
      tr.appendChild(el("td", r.rule, "mono"));
      tr.appendChild(el("td", fmt(r.hits), "num"));
      rules.appendChild(tr);
    });
    $("rules-empty").hidden = (data.top_rules || []).length > 0;

    var recent = $("recent");
    clear(recent);
    (data.recent || []).forEach(function (e) {
      var tr = el("tr");
      var t = e.time ? new Date(e.time) : null;
      tr.appendChild(el("td", t && !isNaN(t) ? t.toLocaleTimeString() : "-", "mono"));
      tr.appendChild(el("td", e.context || "-"));
      var cell = el("td");
      cell.appendChild(el("span", e.decision || "-", decisionClass(e.decision)));
      tr.appendChild(cell);
      tr.appendChild(el("td", e.risk_score === null || e.risk_score === undefined ? "-" : e.risk_score, "num"));
      tr.appendChild(el("td", (e.rules || []).join(", ") || "-", "mono small"));
      tr.appendChild(el("td", e.latency_ms === null || e.latency_ms === undefined ? "-" : e.latency_ms, "num"));
      tr.appendChild(el("td", e.app_id || "-", "small"));
      recent.appendChild(tr);
    });
    $("recent-empty").hidden = (data.recent || []).length > 0;

    var workers = $("workers");
    clear(workers);
    var procs = (data.workers && data.workers.processes) || [];
    procs.forEach(function (w) {
      var tr = el("tr");
      tr.appendChild(el("td", "worker " + w.index));
      tr.appendChild(el("td", w.pid, "num mono"));
      tr.appendChild(el("td", duration(w.uptime_seconds), "num"));
      tr.appendChild(el("td", fmt(w.requests), "num"));
      var age = el("td", w.age_seconds === null || w.age_seconds === undefined ? "-" : w.age_seconds + "s ago", "num");
      if (w.age_seconds > 15) { age.className += " warn"; }
      tr.appendChild(age);
      workers.appendChild(tr);
    });
    $("workers-empty").hidden = procs.length > 0;

    var policy = $("policy");
    clear(policy);
    var p = data.policy || {};
    var rows = [["Mode", data.mode], ["State store", data.state_backend]];
    if (p.policy_path) { rows.push(["Policy file", p.policy_path]); }
    if (p.bundle_id) { rows.push(["Bundle", p.bundle_id]); }
    if (p.policy_hash) { rows.push(["Policy hash", String(p.policy_hash).slice(0, 16)]); }
    if (p.environment) { rows.push(["Environment", p.environment]); }
    (p.environments || []).forEach(function (e) {
      rows.push([String(e.project || "-") + " / " + String(e.environment || "-"),
                 String(e.policy_hash || "").slice(0, 16)]);
    });
    if (p.missing) { rows.push(["Not yet published", String(p.missing)]); }
    rows.forEach(function (r) {
      policy.appendChild(el("dt", r[0]));
      policy.appendChild(el("dd", r[1] === undefined || r[1] === null ? "-" : r[1], "mono"));
    });

    var models = $("models");
    clear(models);
    var ms = data.model_server;
    if (!ms) {
      models.appendChild(el("p", "Not configured. Rules only. Start with znyx serve --ml to add model-backed checks.", "hint"));
    } else if (!ms.reachable) {
      models.appendChild(el("p", "Configured at " + ms.url + " but not reachable. Checks fall back to rules.", "warn"));
    } else if (!ms.models.length) {
      models.appendChild(el("p", "Running, no models loaded yet.", "hint"));
    } else {
      var tbl = el("table", null, "table");
      var body = el("tbody");
      ms.models.forEach(function (m) {
        var tr = el("tr");
        tr.appendChild(el("td", m.task));
        tr.appendChild(el("td", m.model_id || m.runner, "mono small"));
        tr.appendChild(el("td", m.available ? "ready" : (m.detail || "unavailable"), m.available ? "" : "warn"));
        body.appendChild(tr);
      });
      tbl.appendChild(body);
      models.appendChild(tbl);
    }

    renderGateway(data.gateway || {});

    $("updated").textContent = "Updated " + new Date().toLocaleTimeString() + ". Refreshes every 5 seconds.";
  }

  function usd(n) {
    if (n === null || n === undefined) { return "-"; }
    return "$" + (n < 1 ? Number(n).toFixed(4) : Number(n).toFixed(2));
  }

  function renderGateway(gw) {
    $("gateway-card").hidden = !gw.enabled;
    if (!gw.enabled) { return; }
    $("gw-providers").textContent = "Providers: " + (gw.providers || []).join(", ") + ". " +
      fmt(gw.keys) + (gw.keys === 1 ? " app key" : " app keys") + ". Configured from " + gw.source + ".";
    $("gw-calls").textContent = fmt(gw.calls);
    $("gw-outcomes").textContent = Object.keys(gw.outcomes || {}).map(function (k) {
      return k.replace("_", " ") + " " + fmt(gw.outcomes[k]);
    }).join(", ");
    var tokens = gw.tokens || {};
    $("gw-tokens").textContent = fmt((tokens.input || 0) + (tokens.output || 0));
    $("gw-tokens-sub").textContent = "in " + fmt(tokens.input || 0) + ", out " + fmt(tokens.output || 0);
    $("gw-cost").textContent = usd(gw.cost_usd);
    var guard = gw.guard_ms || {}, ttft = gw.ttft_ms || {};
    $("gw-guard").textContent = guard.p50 === null || guard.p50 === undefined ? "-" : "\u2264 " + guard.p50 + " ms";
    $("gw-ttft").textContent = ttft.p50 === null || ttft.p50 === undefined ? "" :
      "time to first token p50 \u2264 " + ttft.p50 + " ms";
    var models = $("gw-models");
    clear(models);
    (gw.models || []).forEach(function (m) {
      var tr = el("tr");
      tr.appendChild(el("td", m.model, "mono small"));
      tr.appendChild(el("td", fmt(m.calls), "num"));
      tr.appendChild(el("td", fmt(m.tokens), "num"));
      tr.appendChild(el("td", usd(m.cost_usd), "num"));
      models.appendChild(tr);
    });
    var recent = $("gw-recent");
    clear(recent);
    (gw.recent || []).slice(0, 20).forEach(function (c) {
      var tr = el("tr");
      tr.appendChild(el("td", c.app_id || c.key_id || "-", "small"));
      tr.appendChild(el("td", c.response_model || c.model || "-", "mono small"));
      var cell = el("td");
      var decision = c.outcome === "ok" ? "ALLOW" : (c.outcome.indexOf("blocked") === 0 ? "BLOCK" : "WARN");
      cell.appendChild(el("span", c.outcome.replace("_", " "), decisionClass(decision)));
      tr.appendChild(cell);
      tr.appendChild(el("td", fmt((c.input_tokens || 0) + (c.output_tokens || 0)), "num"));
      tr.appendChild(el("td", c.latency_ms, "num"));
      recent.appendChild(tr);
    });
  }

  function load() {
    return fetch("/v1/local/summary", { headers: headers(), cache: "no-store" }).then(function (resp) {
      if (resp.status === 401) { showSignin(key() ? "That key was not accepted." : ""); return; }
      if (resp.status === 404) { showSignin("The dashboard is switched off on this instance."); return; }
      if (!resp.ok) {
        return resp.json().catch(function () { return {}; }).then(function (body) {
          showSignin((body && body.detail) || ("The server answered " + resp.status + "."));
        });
      }
      return resp.json().then(render);
    }).catch(function () {
      $("updated").textContent = "Cannot reach the server. Retrying.";
    });
  }

  function schedule() {
    if (timer) { clearInterval(timer); }
    timer = setInterval(function () { if (!document.hidden && $("signin").hidden) { load(); } }, REFRESH_MS);
  }

  $("keyform").addEventListener("submit", function (ev) {
    ev.preventDefault();
    try { sessionStorage.setItem(KEY, $("key").value.trim()); } catch (e) { /* private mode */ }
    $("key").value = "";
    load();
  });

  $("signout").addEventListener("click", function () {
    try { sessionStorage.removeItem(KEY); } catch (e) { /* private mode */ }
    showSignin("");
  });

  function updateCurl() {
    var ctx = $("try-context").value;
    var text = $("try-text").value;
    var body = JSON.stringify({ request_id: "try-1", tenant_id: "local", app_id: "dashboard", text: text });
    $("curl").textContent = "curl -s " + location.origin + "/v1/evaluate/" + ctx +
      " \\\n  -H 'Content-Type: application/json' \\\n  -H \"X-API-Key: $(cat ~/.znyx/api-key)\" \\\n  -d '" +
      body.replace(/'/g, "'\\''") + "'";
  }
  $("try-text").addEventListener("input", updateCurl);
  $("try-context").addEventListener("change", updateCurl);

  $("tryform").addEventListener("submit", function (ev) {
    ev.preventDefault();
    var ctx = $("try-context").value === "output" ? "output" : "input";
    var out = $("try-result");
    out.hidden = false;
    clear(out);
    out.appendChild(el("p", "Checking...", "hint"));
    fetch("/v1/evaluate/" + ctx, {
      method: "POST",
      headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({ request_id: "dashboard-" + Date.now(), tenant_id: "local",
                             app_id: "dashboard", text: $("try-text").value })
    }).then(function (resp) {
      return resp.json().catch(function () { return {}; }).then(function (body) {
        clear(out);
        if (!resp.ok) { out.appendChild(el("p", body.detail || ("Error " + resp.status), "warn")); return; }
        var head = el("div", null, "try-head");
        head.appendChild(el("span", body.decision, decisionClass(body.decision)));
        head.appendChild(el("span", "risk " + body.risk_score, "muted"));
        out.appendChild(head);
        var hits = body.rule_hits || [];
        if (hits.length) {
          var ul = el("ul", null, "hits");
          hits.forEach(function (h) {
            var li = el("li");
            li.appendChild(el("span", h.rule_id, "mono"));
            if (h.message) { li.appendChild(el("span", " " + h.message, "muted")); }
            ul.appendChild(li);
          });
          out.appendChild(ul);
        } else {
          out.appendChild(el("p", "No rule fired.", "hint"));
        }
        if (body.sanitized_text) {
          out.appendChild(el("div", "Text after the check", "label"));
          out.appendChild(el("pre", body.sanitized_text, "sanitized"));
        }
        load();
      });
    }).catch(function () {
      clear(out);
      out.appendChild(el("p", "Cannot reach the server.", "warn"));
    });
  });

  updateCurl();
  load();
  schedule();
})();
