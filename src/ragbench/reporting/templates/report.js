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
