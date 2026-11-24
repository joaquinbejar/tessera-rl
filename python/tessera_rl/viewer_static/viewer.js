// SPDX-License-Identifier: MIT OR Apache-2.0
// The Tessera Viewer front end (ADR-0083 item 6, ADR-0085): vanilla
// JavaScript, Canvas 2D, no external requests. Every request carries the
// session token from this page's URL.
"use strict";

(function () {
  var token = new URLSearchParams(location.search).get("token") || "";
  var $ = function (id) { return document.getElementById(id); };
  var canvas = $("canvas");
  var ctx = canvas.getContext("2d");
  var timeline = $("timeline");
  var instanceSelect = $("instance");
  var meta = null;
  var pending = 0;

  function api(path, params) {
    var q = new URLSearchParams(params || {});
    q.set("token", token);
    return fetch(path + "?" + q.toString(), { cache: "no-store", credentials: "omit", referrerPolicy: "no-referrer" })
      .then(function (r) {
        if (!r.ok) { throw new Error(path + ": HTTP " + r.status); }
        return r.json();
      });
  }

  function showError(err) {
    var el = $("error");
    el.textContent = String(err && err.message ? err.message : err);
    el.hidden = false;
  }

  function color(rgba) {
    return "rgba(" + rgba[0] + "," + rgba[1] + "," + rgba[2] + "," + (rgba[3] / 255) + ")";
  }

  function draw(prims) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    prims.forEach(function (p) {
      var kind = p[0];
      if (kind === "rect") {
        ctx.fillStyle = color(p[5]);
        ctx.fillRect(p[1], p[2], p[3] - p[1], p[4] - p[2]);
      } else if (kind === "circle") {
        ctx.fillStyle = color(p[4]);
        ctx.beginPath();
        ctx.arc(p[1], p[2], p[3], 0, 2 * Math.PI);
        ctx.fill();
      } else if (kind === "line") {
        ctx.strokeStyle = color(p[6]);
        ctx.lineWidth = p[5];
        ctx.beginPath();
        ctx.moveTo(p[1], p[2]);
        ctx.lineTo(p[3], p[4]);
        ctx.stroke();
      } else if (kind === "polygon") {
        var pts = p[1];
        if (pts.length < 2) { return; }
        ctx.fillStyle = color(p[2]);
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (var j = 1; j < pts.length; j++) { ctx.lineTo(pts[j][0], pts[j][1]); }
        ctx.closePath();
        ctx.fill();
      }
    });
  }

  function cell(row, text) {
    var td = document.createElement("td");
    td.textContent = text;
    row.appendChild(td);
  }

  function fmt(v) {
    if (v === null) { return "nan"; }
    if (typeof v === "number" && !Number.isInteger(v)) { return v.toPrecision(5); }
    return String(v);
  }

  function showStep(s) {
    var body = $("agents").querySelector("tbody");
    body.textContent = "";
    for (var a = 0; a < s.rewards.length; a++) {
      var row = document.createElement("tr");
      cell(row, String(a));
      cell(row, fmt(s.rewards[a]));
      cell(row, s.alive[a] ? "yes" : "no");
      cell(row, s.acting[a] ? "yes" : "no");
      cell(row, s.agent_terminated[a] ? "yes" : "no");
      cell(row, s.agent_truncated[a] ? "yes" : "no");
      cell(row, s.masks ? s.masks[a].map(function (m) { return m ? "1" : "0"; }).join("") : "-");
      body.appendChild(row);
    }
    $("episode").textContent = "Episode terminated: " + (s.episode_terminated ? "yes" : "no") +
      "; truncated: " + (s.episode_truncated ? "yes" : "no");
    var obs = $("observations");
    obs.textContent = "";
    Object.keys(s.observations).forEach(function (name) {
      var leaf = s.observations[name];
      var div = document.createElement("div");
      div.className = "leaf";
      var head = document.createElement("div");
      head.textContent = name + " ";
      var shape = document.createElement("span");
      shape.className = "shape";
      shape.textContent = leaf.dtype + "[" + leaf.shape.join(", ") + "]" + (leaf.truncated ? " (too large to show)" : "");
      head.appendChild(shape);
      div.appendChild(head);
      if (!leaf.truncated) {
        var code = document.createElement("code");
        code.textContent = leaf.values.map(fmt).join(" ");
        div.appendChild(code);
      }
      obs.appendChild(div);
    });
  }

  function load() {
    if (!meta || meta.calls === 0) { return; }
    var i = Number(timeline.value);
    var params = { i: String(i), instance: instanceSelect.value };
    var ticket = ++pending;
    $("position").textContent = "call " + i + " / " + (meta.calls - 1);
    var frame = meta.canvas ? api("/frame", params) : Promise.resolve(null);
    Promise.all([frame, api("/step", params)]).then(function (res) {
      if (ticket !== pending) { return; }
      if (res[0] && res[0].prims) { draw(res[0].prims); }
      showStep(res[1]);
      $("error").hidden = true;
    }).catch(showError);
  }

  function move(delta) {
    var v = Math.min(Math.max(Number(timeline.value) + delta, 0), Number(timeline.max));
    timeline.value = String(v);
    load();
  }

  api("/meta").then(function (m) {
    meta = m;
    $("env").textContent = (m.env || "") + "  " + m.num_envs + " instance(s), " + m.agents + " agent(s), " + m.calls + " call(s)";
    if (m.canvas) {
      canvas.width = m.canvas[0];
      canvas.height = m.canvas[1];
    } else {
      canvas.hidden = true;
      $("no-canvas").hidden = false;
    }
    for (var k = 0; k < m.num_envs; k++) {
      var opt = document.createElement("option");
      opt.value = String(k);
      opt.textContent = String(k);
      instanceSelect.appendChild(opt);
    }
    timeline.max = String(Math.max(m.calls - 1, 0));
    timeline.value = "0";
    load();
  }).catch(showError);

  timeline.addEventListener("input", load);
  instanceSelect.addEventListener("change", load);
  $("prev").addEventListener("click", function () { move(-1); });
  $("next").addEventListener("click", function () { move(1); });
  document.addEventListener("keydown", function (e) {
    if (e.target === timeline || e.target === instanceSelect) { return; }
    if (e.key === "ArrowLeft") { move(-1); } else if (e.key === "ArrowRight") { move(1); }
  });
})();
