// Appendix D: the messy ledger and its list of problems highlight each other.
//
// Markup: <div class="tidy-link" markdown> wraps two Markdown tables, the
// ledger first and the problem list second. PROBLEMS[k] lists the ledger
// cells [row, column] for problem row k; row 0 is the ledger's header row.
// Without JavaScript the two tables stay as they are.
(function () {
  "use strict";

  const ALL = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  const row = (r, cols) => (cols || ALL).map((c) => [r, c]);

  const PROBLEMS = [
    [...row(0), ...row(1)],                    // 2 行の列名
    [[3, 0]],                                  // 「〃」
    [[0, 1], [0, 4]],                          // ID がない
    [[2, 1], [3, 1], [4, 1]],                  // 日付の混在
    [[2, 4], [2, 6], [4, 4]],                  // 1 列に複数の変数
    [[2, 5], [2, 7], [4, 5]],                  // 単位入りの数
    row(1, [4, 5, 6, 7]),                      // 横に並べた病変
    [[2, 8]],                                  // 1 セルに複数の値
    [[2, 3], [3, 3]],                          // 表記ゆれ
    row(3, [4, 5, 6, 7, 8]),                   // 欠測の混在
    row(5),                                    // 合計の行
    [...row(2), ...row(3), ...row(4)],         // 2 種類の観測が同じ行
  ];

  function cellsOf(table) {
    return Array.from(table.querySelectorAll("tr")).map((tr) => Array.from(tr.children));
  }

  function setup(root) {
    if (root.dataset.ready) return;
    const tables = root.querySelectorAll("table");
    if (tables.length < 2) return;
    root.dataset.ready = "1";
    tables[0].dataset.tidy = "ledger";
    const ledger = cellsOf(tables[0]);
    const problems = Array.from(tables[1].querySelectorAll("tbody tr"));
    const hits = ledger.map((r) => r.map(() => []));   // problems that contain each cell

    PROBLEMS.forEach((cells, k) => {
      for (const [r, c] of cells) if (ledger[r] && ledger[r][c]) hits[r][c].push(k);
    });
    // Most specific problem first, so a click on a cell picks the narrowest match.
    for (const r of hits) for (const list of r) list.sort((a, b) => PROBLEMS[a].length - PROBLEMS[b].length);

    let current = -1;
    let last = null;

    function show(k, from) {
      current = k;
      root.querySelectorAll(".tidy-on").forEach((n) => n.classList.remove("tidy-on"));
      if (k < 0) return;
      for (const [r, c] of PROBLEMS[k]) if (ledger[r] && ledger[r][c]) ledger[r][c].classList.add("tidy-on");
      problems[k].classList.add("tidy-on");
      const [r0, c0] = PROBLEMS[k][0];
      const target = from === "ledger" ? problems[k] : ledger[r0][c0];
      if (target) target.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" });
    }

    ledger.forEach((r, i) =>
      r.forEach((cell, j) => {
        const list = hits[i][j];
        if (!list.length) return;
        cell.classList.add("tidy-hot");
        cell.addEventListener("click", () => {
          // A second click on the same cell moves to the next problem that contains it.
          const at = list.indexOf(current);
          const next = last === cell && at >= 0 ? list[(at + 1) % list.length] : list[0];
          last = cell;
          show(next === current && list.length === 1 ? -1 : next, "ledger");
        });
      })
    );
    problems.forEach((tr, k) => {
      tr.classList.add("tidy-hot");
      tr.addEventListener("click", () => {
        last = null;
        show(current === k ? -1 : k, "problems");
      });
    });

    const hint = document.createElement("p");
    hint.className = "tidy-hint";
    hint.textContent = "台帳のセルか、下の表の行をクリックすると、対応する場所に色がつきます。同じセルをもう一度クリックすると、次の崩れ方に移ります。";
    (tables[0].closest(".md-typeset__scrollwrap") || tables[0]).before(hint);
  }

  function boot(scope) {
    for (const root of (scope || document).querySelectorAll(".tidy-link")) setup(root);
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(({ body }) => boot(body));
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => boot(document));
  } else {
    boot(document);
  }
})();
