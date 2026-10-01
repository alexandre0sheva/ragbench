/* RAGBench report behavior: theme, sortable leaderboard, column picker, heatmap metric toggle, tooltips, copy. Everything works without it except these
   conveniences: every value is in the page's tables. No network, no external resources; data is inserted with textContent only. */
(function () {
  "use strict";
  var root = document.documentElement;
  function store(key, value) {
    try {
      if (value === undefined) return window.localStorage.getItem(key);
      window.localStorage.setItem(key, value);
    } catch (e) { /* private mode or blocked storage: the report works without remembering */ }
    return null;
  }

  // ---- theme: auto (follow the OS) -> light -> dark ----
  var themeButton = document.getElementById("theme-toggle");
  function applyTheme(mode) {
    if (mode === "light" || mode === "dark") root.setAttribute("data-theme", mode); else root.removeAttribute("data-theme");
    if (themeButton) { themeButton.textContent = "Theme: " + (mode === "light" || mode === "dark" ? mode : "auto"); themeButton.setAttribute("aria-pressed", mode === "light" || mode === "dark" ? "true" : "false"); }
  }
  var saved = store("ragbench-report-theme");
  applyTheme(saved === "light" || saved === "dark" ? saved : "auto");
  if (themeButton) themeButton.addEventListener("click", function () {
    var now = root.getAttribute("data-theme") || "auto";
    var next = now === "auto" ? "light" : now === "light" ? "dark" : "auto";
    applyTheme(next); store("ragbench-report-theme", next);
  });

  // ---- leaderboard: sort ----
  var table = document.getElementById("leaderboard");
  if (table) {
    var tbody = table.tBodies[0];
    Array.prototype.forEach.call(table.querySelectorAll("th button[data-sort]"), function (button) {
      button.addEventListener("click", function () {
        var th = button.closest("th");
        var index = Array.prototype.indexOf.call(th.parentNode.children, th);
        var numeric = button.getAttribute("data-kind") === "num";
        var ascending = th.getAttribute("aria-sort") !== "ascending";
        Array.prototype.forEach.call(th.parentNode.children, function (other) { other.removeAttribute("aria-sort"); });
        th.setAttribute("aria-sort", ascending ? "ascending" : "descending");
        var rows = Array.prototype.slice.call(tbody.rows);
        rows.sort(function (a, b) {
          var av = a.cells[index], bv = b.cells[index];
          if (numeric) {
            var x = parseFloat(av.getAttribute("data-value")), y = parseFloat(bv.getAttribute("data-value"));
            var xn = isNaN(x), yn = isNaN(y);
            if (xn || yn) return xn === yn ? 0 : xn ? 1 : -1; // missing values sink whichever way it is sorted
            return (x - y) * (ascending ? 1 : -1);
          }
          return av.textContent.trim().toLowerCase().localeCompare(bv.textContent.trim().toLowerCase()) * (ascending ? 1 : -1);
        });
        rows.forEach(function (row) { tbody.appendChild(row); });
      });
    });

    // ---- leaderboard: column picker ----
    var boxes = document.querySelectorAll("#column-picker input[type=checkbox]");
    var remembered = null;
    try { remembered = JSON.parse(store("ragbench-report-columns") || "null"); } catch (e) { remembered = null; }
    function showColumns() {
      var visible = {};
      Array.prototype.forEach.call(boxes, function (box) { visible[box.value] = box.checked; });
      Array.prototype.forEach.call(table.querySelectorAll("[data-col]"), function (cell) { cell.hidden = !visible[cell.getAttribute("data-col")]; });
    }
    Array.prototype.forEach.call(boxes, function (box) {
      if (remembered && Object.prototype.hasOwnProperty.call(remembered, box.value)) box.checked = !!remembered[box.value];
      box.addEventListener("change", function () {
        showColumns();
        var state = {};
        Array.prototype.forEach.call(boxes, function (b) { state[b.value] = b.checked; });
        store("ragbench-report-columns", JSON.stringify(state));
      });
    });
    showColumns();
  }

  // ---- heatmap metric toggle ----
  var toggles = document.querySelectorAll("[data-heat-button]");
  Array.prototype.forEach.call(toggles, function (button) {
    button.addEventListener("click", function () {
      var which = button.getAttribute("data-heat-button");
      Array.prototype.forEach.call(toggles, function (other) { other.setAttribute("aria-pressed", other === button ? "true" : "false"); });
      Array.prototype.forEach.call(document.querySelectorAll("[data-heat]"), function (panel) { panel.hidden = panel.getAttribute("data-heat") !== which; });
    });
  });

  // ---- tooltips on chart marks (hover and keyboard focus) ----
  var tip = document.createElement("div");
  tip.className = "tip"; tip.setAttribute("role", "tooltip"); tip.id = "chart-tip";
  document.body.appendChild(tip);
  Array.prototype.forEach.call(document.querySelectorAll(".chart .mark > title"), function (node) { node.parentNode.removeChild(node); });
  function showTip(text, x, y) {
    tip.textContent = text; tip.classList.add("on");
    var box = tip.getBoundingClientRect();
    var left = Math.min(Math.max(8, x + 14), window.innerWidth - box.width - 8);
    var top = y + 18 + box.height > window.innerHeight ? y - box.height - 12 : y + 18;
    tip.style.left = left + "px"; tip.style.top = Math.max(8, top) + "px";
  }
  function hideTip() { tip.classList.remove("on"); }
  document.addEventListener("pointermove", function (event) {
    var mark = event.target.closest ? event.target.closest(".chart .mark[data-tip]") : null;
    if (mark) showTip(mark.getAttribute("data-tip"), event.clientX, event.clientY); else hideTip();
  });
  document.addEventListener("focusin", function (event) {
    var mark = event.target.closest ? event.target.closest(".chart .mark[data-tip]") : null;
    if (!mark) return;
    var box = mark.getBoundingClientRect();
    showTip(mark.getAttribute("data-tip"), box.left + box.width / 2, box.top + box.height / 2);
  });
  document.addEventListener("focusout", hideTip);
  document.addEventListener("keydown", function (event) { if (event.key === "Escape") hideTip(); });
  window.addEventListener("scroll", hideTip, true);

  // ---- copy winner.yaml ----
  var copy = document.getElementById("copy-yaml");
  var source = document.getElementById("winner-yaml");
  if (copy && source) copy.addEventListener("click", function () {
    var done = function (ok) { copy.textContent = ok ? "Copied" : "Press Ctrl/Cmd+C"; window.setTimeout(function () { copy.textContent = "Copy"; }, 1800); };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(source.textContent).then(function () { done(true); }, function () { selectText(); done(false); });
    } else { selectText(); done(false); }
    function selectText() { var range = document.createRange(); range.selectNodeContents(source); var sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range); }
  });

  // ---- printing: open the table views so the numbers are on paper ----
  window.addEventListener("beforeprint", function () {
    Array.prototype.forEach.call(document.querySelectorAll("details.tableview, details.more"), function (d) { d.setAttribute("open", ""); });
  });
})();

/* Question explorer: filter, search and expand every question, systems side by side. Data comes from the embedded #questions-data (and, when the page
   is served over http and the embed was cut down, from report_questions.json). Everything is inserted with textContent: answers and contexts are
   untrusted text. */
(function () {
  "use strict";
  var holder = document.getElementById("questions-data");
  var table = document.getElementById("q-table");
  if (!holder || !table) return;
  var data;
  try { data = JSON.parse(holder.textContent); } catch (e) { return; }
  var tbody = table.tBodies[0];
  var PAGE = 50;
  var shown = PAGE, focusIndex = 0, openId = null, visible = [];
  var JUDGE_LABELS = { correctness: "Correctness", faithfulness: "Faithfulness", completeness: "Completeness", relevance: "Relevance", citation_quality: "Citations" };
  var MARK_LABELS = { relevant: "labeled relevant", judged_good: "likely relevant (unlabeled)" };

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }
  function num(value, digits) { return value === null || value === undefined ? "—" : Number(value).toFixed(digits === undefined ? 2 : digits); }
  function money(value) { return value === null || value === undefined ? "—" : value === 0 ? "$0" : value >= 1 ? "$" + value.toFixed(2) : "$" + value.toFixed(5); }
  function failureName(key) { return (data.failure_names && data.failure_names[key]) || String(key).replace(/_/g, " "); }
  function runFailure(run) { return run.error ? "run_error" : run.failure; }
  function bucket(score) { return Math.max(0, Math.min(6, Math.round((score / 5) * 6))); }
  function metricValue(run, metric) {
    if (run.error) return null;
    if (metric === "score") return run.score;
    return run.judge ? run.judge[metric] : null;
  }
  function systemIndex(name) { return data.systems.indexOf(name); }

  // ---- filters ----
  var search = $("q-search"), category = $("q-category"), system = $("q-system"), failure = $("q-failure");
  var metric = $("q-metric"), below = $("q-below"), disagree = $("q-disagree"), count = $("q-count");
  function fillOptions() {
    var cats = {}, fails = {};
    data.questions.forEach(function (q) {
      cats[q.cat] = (cats[q.cat] || 0) + 1;
      q.runs.forEach(function (run) { fails[runFailure(run)] = true; });
    });
    function reset(select, first) {
      var keep = select.value;
      while (select.options.length > 1) select.remove(1);
      select.options[0].textContent = first;
      return keep;
    }
    var keepCat = reset(category, "All"), keepSys = reset(system, "All systems"), keepFail = reset(failure, "Any");
    Object.keys(cats).sort(function (a, b) { return cats[b] - cats[a] || (a < b ? -1 : 1); }).forEach(function (c) { category.add(new Option(c + " (" + cats[c] + ")", c)); });
    data.systems.forEach(function (name) { system.add(new Option(name, name)); });
    Object.keys(fails).sort().forEach(function (f) { failure.add(new Option(failureName(f), f)); });
    category.value = keepCat; system.value = keepSys; failure.value = keepFail;
    disagree.parentNode.title = "The best and worst answer scores for this question are at least " + data.disagree_spread + " points apart";
  }
  function haystack(q) {
    if (q._hay === undefined) q._hay = [q.id, q.q, q.cat, q.ref || ""].concat(q.runs.map(function (r) { return r.answer; })).join("\n").toLowerCase();
    return q._hay;
  }
  function filtered() {
    var text = search.value.trim().toLowerCase();
    var limit = below.value === "" ? null : parseFloat(below.value);
    var only = system.value === "" ? -1 : systemIndex(system.value);
    return data.questions.filter(function (q) {
      if (category.value && q.cat !== category.value) return false;
      var runs = only < 0 ? q.runs : q.runs.filter(function (r) { return r.s === only; });
      if (!runs.length) return false;
      if (failure.value && !runs.some(function (r) { return runFailure(r) === failure.value; })) return false;
      if (limit !== null && !isNaN(limit) && !runs.some(function (r) { var v = metricValue(r, metric.value); return v !== null && v !== undefined && v < limit; })) return false;
      if (disagree.checked && !(q.spread !== null && q.spread >= data.disagree_spread)) return false;
      return !text || haystack(q).indexOf(text) >= 0;
    });
  }

  // ---- the table ----
  function chip(run) {
    var name = data.systems[run.s];
    var node = el("span", "q-chip " + (run.error ? "err" : run.score === null ? "na" : "sc-" + bucket(run.score)) + (runFailure(run) !== "no_failure" ? " failed" : ""), run.error ? "err" : num(run.score, 1));
    var label = name + ": " + (run.error ? "error" : "score " + num(run.score, 1)) + (runFailure(run) !== "no_failure" ? " (" + failureName(runFailure(run)) + ")" : "");
    node.title = label; node.setAttribute("aria-label", label);
    return node;
  }
  function rowFor(q, index) {
    var tr = el("tr", "q-row");
    tr.tabIndex = index === focusIndex ? 0 : -1;
    tr.setAttribute("data-id", q.id); tr.setAttribute("data-index", String(index));
    tr.setAttribute("aria-expanded", q.id === openId ? "true" : "false");
    tr.appendChild(el("td", "mono q-id", q.id));
    var text = el("td", "q-text"); text.appendChild(el("span", null, q.q)); tr.appendChild(text);
    tr.appendChild(el("td", "q-cat", q.cat));
    var chips = el("td", "q-chips");
    q.runs.slice().sort(function (a, b) { return a.s - b.s; }).forEach(function (run) { chips.appendChild(chip(run)); });
    tr.appendChild(chips);
    tr.appendChild(el("td", "n q-spread" + (q.spread !== null && q.spread >= data.disagree_spread ? " wide" : ""), num(q.spread, 1)));
    return tr;
  }
  function render() {
    visible = filtered();
    var matching = visible.length;
    visible = visible.slice(0, shown);
    if (focusIndex >= visible.length) focusIndex = 0;
    while (tbody.firstChild) tbody.removeChild(tbody.firstChild);
    visible.forEach(function (q, index) {
      tbody.appendChild(rowFor(q, index));
      if (q.id === openId) tbody.appendChild(detailRow(q));
    });
    if (!visible.length) {
      var tr = el("tr"), td = el("td", "empty", "No question matches these filters."); td.colSpan = 5; tr.appendChild(td); tbody.appendChild(tr);
    }
    var more = $("q-more");
    more.hidden = matching <= visible.length;
    more.textContent = "Show " + Math.min(PAGE, matching - visible.length) + " more";
    count.textContent = matching + " of " + data.questions.length + " questions match" + (data.total > data.questions.length ? " (" + data.total + " in the run; see the note above)" : "") + ".";
  }

  // ---- one question, systems side by side ----
  function section(title, open) {
    var d = el("details", "q-sec"); if (open) d.open = true;
    d.appendChild(el("summary", null, title));
    return d;
  }
  function contextList(run) {
    var list = el("ol", "q-ctx");
    run.ctx.forEach(function (c) {
      var li = el("li", "ctx " + (c.mark === "relevant" ? "relevant" : c.mark === "judged_good" ? "good" : "plain") + (c.in ? "" : " out"));
      var head = el("div", "ctx-head");
      head.appendChild(el("span", "mono", "#" + c.rank + " " + c.doc));
      if (c.mark) head.appendChild(el("span", "badge", MARK_LABELS[c.mark]));
      if (!c.in) head.appendChild(el("span", "badge", "not shown to the generator"));
      var where = [c.page !== undefined ? "page " + c.page : "", c.heading || ""].filter(Boolean).join(" · ");
      if (where) head.appendChild(el("span", "muted", where));
      head.appendChild(el("span", "muted", "score " + num(c.score, 3)));
      li.appendChild(head);
      if (c.text) li.appendChild(el("p", "ctx-text", c.text));
      list.appendChild(li);
    });
    return list;
  }
  function traceList(run) {
    var total = run.steps.reduce(function (sum, s) { return sum + (s.ms || 0); }, 0) || 1;
    var list = el("ol", "q-trace");
    run.steps.forEach(function (s) {
      var li = el("li", "step k-" + s.kind);
      var head = el("div", "step-head");
      head.appendChild(el("span", "kind", s.kind));
      head.appendChild(el("span", "mono", s.name));
      head.appendChild(el("span", "muted", num(s.ms, 1) + " ms" + (s.cost ? " · " + money(s.cost) : "") + (s.tokens ? " · " + s.tokens + " tokens" : "")));
      li.appendChild(head);
      var bar = el("span", "step-bar"); bar.style.width = Math.max(2, Math.round((100 * (s.ms || 0)) / total)) + "%"; bar.setAttribute("aria-hidden", "true");
      li.appendChild(bar);
      if (s.in) { var i = el("p", "step-io"); i.appendChild(el("b", null, "in ")); i.appendChild(document.createTextNode(s.in)); li.appendChild(i); }
      if (s.out) { var o = el("p", "step-io"); o.appendChild(el("b", null, "out ")); o.appendChild(document.createTextNode(s.out)); li.appendChild(o); }
      if (s.meta) {
        var keys = Object.keys(s.meta);
        if (keys.length) li.appendChild(el("p", "step-meta muted", keys.map(function (k) { return k + "=" + s.meta[k]; }).join("  ")));
      }
      list.appendChild(li);
    });
    return list;
  }
  function runCard(q, run) {
    var card = el("article", "q-card");
    card.setAttribute("aria-label", data.systems[run.s]);
    var head = el("header", "q-card-head");
    head.appendChild(el("h4", null, data.systems[run.s]));
    head.appendChild(chip(run));
    if (runFailure(run) !== "no_failure") head.appendChild(el("span", "badge failed", failureName(runFailure(run))));
    if (run.route) head.appendChild(el("span", "badge", "route: " + run.route));
    if (run.refused) head.appendChild(el("span", "badge", "refused"));
    card.appendChild(head);
    card.appendChild(el("p", "muted q-facts", num(run.latency_ms, 0) + " ms · " + money(run.cost) + " per question"));
    if (run.error) { card.appendChild(el("p", "q-error", "This question raised an error: " + run.error)); return card; }
    var answer = section("Answer", true);
    answer.appendChild(el("p", "q-answer", run.answer || "(empty answer)"));
    card.appendChild(answer);
    if (run.judge) {
      var judge = section("Judge", true), grid = el("dl", "q-judge");
      Object.keys(JUDGE_LABELS).forEach(function (key) {
        var cell = el("div"); cell.appendChild(el("dt", null, JUDGE_LABELS[key])); cell.appendChild(el("dd", null, num(run.judge[key], 1))); grid.appendChild(cell);
      });
      judge.appendChild(grid);
      if (run.reasoning) judge.appendChild(el("p", "q-reason", run.reasoning));
      if (run.judge_by) judge.appendChild(el("p", "muted", "judged by: " + run.judge_by));
      card.appendChild(judge);
    }
    if (run.ctx) {
      var ctx = section("Retrieved contexts (" + run.ctx.length + ")", true);
      ctx.appendChild(run.ctx.length ? contextList(run) : el("p", "muted", "This system retrieved nothing."));
      card.appendChild(ctx);
    }
    if (run.steps) {
      var trace = section("Trace (" + run.steps.length + " steps)", true);
      if (run.agent) trace.appendChild(el("p", "muted", Object.keys(run.agent).map(function (k) { return k + "=" + run.agent[k]; }).join("  ")));
      trace.appendChild(run.steps.length ? traceList(run) : el("p", "muted", "No steps were recorded."));
      card.appendChild(trace);
    } else {
      card.appendChild(el("p", "muted", "Contexts and trace are not embedded in this copy of the page."));
    }
    return card;
  }
  function detailRow(q) {
    var tr = el("tr", "q-detail-row"), td = el("td"); td.colSpan = 5;
    var box = el("div", "q-detail");
    box.appendChild(el("h3", null, q.q));
    var facts = el("dl", "q-ref");
    function fact(label, value) { if (!value) return; var d = el("div"); d.appendChild(el("dt", null, label)); d.appendChild(el("dd", null, value)); facts.appendChild(d); }
    fact("Reference answer", q.unanswerable ? "(unanswerable: the right behavior is to say so)" : q.ref);
    fact("Relevant documents", q.rel.length ? q.rel.join(", ") : "none labeled");
    fact("Category", q.cat + (q.diff ? " · " + q.diff : ""));
    fact("Needs tools", q.tools ? q.tools.join(", ") : "");
    fact("Routing hint", q.hint);
    box.appendChild(facts);
    var grid = el("div", "q-systems");
    q.runs.slice().sort(function (a, b) { return a.s - b.s; }).forEach(function (run) { grid.appendChild(runCard(q, run)); });
    grid.tabIndex = 0; grid.setAttribute("role", "region"); grid.setAttribute("aria-label", "Systems side by side");
    box.appendChild(grid);
    td.appendChild(box); tr.appendChild(td);
    return tr;
  }
  function rowAt(index) { return tbody.querySelector('.q-row[data-index="' + index + '"]'); }
  function setOpen(id) {
    var index = visible.findIndex(function (q) { return q.id === id; });
    openId = openId === id ? null : id;
    if (index >= 0) focusIndex = index;
    render();
    var row = rowAt(focusIndex);
    if (row) row.focus();
  }
  function moveFocus(to) {
    if (to >= visible.length && $("q-more").hidden === false) { shown += PAGE; render(); }
    to = Math.max(0, Math.min(to, visible.length - 1));
    var old = rowAt(focusIndex), next = rowAt(to);
    if (old) old.tabIndex = -1;
    if (next) { next.tabIndex = 0; next.focus(); focusIndex = to; }
  }

  tbody.addEventListener("click", function (event) {
    var row = event.target.closest ? event.target.closest(".q-row") : null;
    if (row) setOpen(row.getAttribute("data-id"));
  });
  tbody.addEventListener("keydown", function (event) {
    var row = event.target.classList && event.target.classList.contains("q-row") ? event.target : null;
    if (row) {
      var index = parseInt(row.getAttribute("data-index"), 10);
      if (event.key === "ArrowDown") { event.preventDefault(); moveFocus(index + 1); }
      else if (event.key === "ArrowUp") { event.preventDefault(); moveFocus(index - 1); }
      else if (event.key === "Home") { event.preventDefault(); moveFocus(0); }
      else if (event.key === "End") { event.preventDefault(); moveFocus(visible.length - 1); }
      else if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setOpen(row.getAttribute("data-id")); }
      else if (event.key === "Escape" && openId) { event.preventDefault(); setOpen(openId); }
    } else if (event.key === "Escape" && openId) {
      event.preventDefault(); setOpen(openId); // close it, and put focus back on its row
    }
  });
  document.addEventListener("keydown", function (event) {
    if (event.key !== "/" || event.ctrlKey || event.metaKey || event.altKey) return;
    var tag = event.target && event.target.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || (event.target && event.target.isContentEditable)) return;
    event.preventDefault(); search.focus(); search.select();
  });
  function refresh() { shown = PAGE; focusIndex = 0; render(); }
  [search, below].forEach(function (input) { input.addEventListener("input", refresh); });
  [category, system, failure, metric, disagree].forEach(function (input) { input.addEventListener("change", refresh); });
  $("q-reset").addEventListener("click", function () {
    search.value = ""; below.value = ""; category.value = ""; system.value = ""; failure.value = ""; metric.value = "score"; disagree.checked = false; openId = null; refresh();
  });
  $("q-more").addEventListener("click", function () { shown += PAGE; render(); });

  // ---- a page that holds a reduced copy says so, and loads the complete data when it can ----
  var note = $("q-note"), noteText = $("q-note-text");
  function reducedNote() {
    var parts = [];
    if (data.questions.length < data.total) parts.push(data.questions.length + " of " + data.total + " questions");
    if (data.detail !== "full") parts.push("reduced detail (" + (data.detail === "scores" ? "scores only" : "shorter text, no trace previews") + ")");
    return "To stay under the size cap this page holds " + parts.join(" with ") + ". ";
  }
  if (data.file) {
    note.hidden = false;
    var http = /^https?:$/.test(window.location.protocol), fileName = data.file;
    noteText.textContent = reducedNote() + (http ? "Loading everything from " + data.file + "…" : "Serve this folder over http (for example `python -m http.server`) to load all of it from " + data.file + ", or read per_question_results.jsonl.");
    if (http && window.fetch) {
      window.fetch(data.file).then(function (r) { if (!r.ok) throw new Error(String(r.status)); return r.json(); }).then(function (full) {
        if (!full || !Array.isArray(full.questions) || !Array.isArray(full.systems)) throw new Error("unexpected data");
        data = full; fillOptions(); render(); note.hidden = true; // everything is here now: nothing left to warn about
      }).catch(function () { noteText.textContent = reducedNote() + "The complete data (" + fileName + ") could not be loaded."; });
    }
  }
  fillOptions();
  render();
})();
