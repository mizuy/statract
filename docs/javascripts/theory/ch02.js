// Interactive figures, see docs/javascripts/theory-widgets.js.
(function () {
  "use strict";
  const { COLOR, el, svgEl, pct, signed, controls, slider, buttons, readout, panel, newSvg, loadCsv, register } = window.SX;

  // ---------- 2.1 confounding and exchangeability ----------

  // One binary confounder: large lesion (20 mm or more). Group sizes and the
  // share of large lesions are those of the 3000 patients. The risks in the
  // two strata are chosen so that the mean Y(0) and Y(1) of each group equal
  // ch2_by_group.csv; with them the crude difference, ATE and ATT are those
  // of the text (+0.8, −2.1, −3.7 points).
  const plus = (x, d) => (x > 0 && Number(x.toFixed(d)) !== 0 ? "+" : "") + signed(x, d);

  async function confoundingWidget(root) {
    const rows = await loadCsv("ch2_by_group.csv");
    const g1 = rows.find((r) => r.clip === 1);
    const g0 = rows.find((r) => r.clip === 0);
    const n1 = g1.len;
    const n0 = g0.len;
    const Q1 = 581 / 927;
    const Q0 = 207 / 2073;
    // Solve q r_large + (1 − q) r_small = mean, for both groups.
    const solve = (m1, m0) => {
      const gap = (m1 - m0) / (Q1 - Q0);
      const small = m0 - Q0 * gap;
      return [small, small + gap];
    };
    const [r0s, r0l] = solve(g1.y0, g0.y0);
    const [r1s, r1l] = solve(g1.y1, g0.y1);
    const RR = { small: r1s / r0s, large: r1l / r0l };
    const book = { q1: Q1, q0: Q0, large: r0l };
    const state = Object.assign({}, book);

    const ctl = controls(root);
    const pre = buttons(ctl, "例", [
      { label: "本のデータに合わせた値", onClick: () => set(book) },
      { label: "両群で同じ割合（ランダム化に近い）", onClick: () => set({ q1: 0.3, q0: 0.3, large: r0l }) },
    ]);
    pre.select(0);
    const mk = (label, key, max) =>
      slider(ctl, {
        label,
        min: 0,
        max,
        step: 0.001,
        value: state[key],
        format: (v) => pct(v, key === "large" ? 1 : 0),
        onInput: (v) => {
          state[key] = v;
          pre.select(-1);
          draw();
        },
      });
    const s1 = mk("クリップ群の大きな病変", "q1", 1);
    const s0 = mk("非クリップ群の大きな病変", "q0", 1);
    const sl = mk("大きな病変の出血リスク", "large", 0.3);
    function set(v) {
      Object.assign(state, v);
      s1.set(v.q1);
      s0.set(v.q0);
      sl.set(v.large);
      draw();
    }

    const svg = newSvg(root, 640, 300, "群ごとの Y(0) と Y(1)");
    const P = panel(svg, {
      left: 60,
      top: 24,
      width: 380,
      height: 220,
      x: [0, 2],
      y: [0, 0.32],
      yticks: [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
      yfmt: (t) => Math.round(100 * t) + "%",
      ylab: "出血割合",
      ylabOffset: 40,
    });
    svgEl("text", { x: P.sx(0.5), y: 262, "text-anchor": "middle", class: "sx-label", text: `クリップ群（${n1} 人）` }, svg);
    svgEl("text", { x: P.sx(1.5), y: 262, "text-anchor": "middle", class: "sx-label", text: `非クリップ群（${n0} 人）` }, svg);
    const key = svgEl("g", {}, svg);
    const keyItem = (y, fill, opacity, label) => {
      svgEl("rect", { x: 460, y: y - 9, width: 14, height: 12, fill, "fill-opacity": opacity }, key);
      svgEl("text", { x: 480, y: y + 2, text: label }, key);
    };
    keyItem(40, COLOR.gray, 1, "Y(0)：クリップなしなら");
    keyItem(60, COLOR.blue, 1, "Y(1)：クリップありなら");
    svgEl("text", { x: 460, y: 84, class: "sx-note", text: "薄い棒は起きなかった方（反事実）" }, key);
    const side = svgEl("g", {}, svg);
    const table = readout(root);

    function draw() {
      const { q1, q0, large } = state;
      const small = r0s;
      const y0 = (q) => q * large + (1 - q) * small;
      const y1 = (q) => q * large * RR.large + (1 - q) * small * RR.small;
      const m = { y0c: y0(q1), y1c: y1(q1), y0n: y0(q0), y1n: y1(q0) };
      const crude = m.y1c - m.y0n;
      const n = n1 + n0;
      const qAll = (n1 * q1 + n0 * q0) / n;
      // Standardized: stratum differences averaged over the whole sample.
      const dSmall = small * (RR.small - 1);
      const dLarge = large * (RR.large - 1);
      const std = qAll * dLarge + (1 - qAll) * dSmall;
      const ate = (n1 * (m.y1c - m.y0c) + n0 * (m.y1n - m.y0n)) / n;
      const att = m.y1c - m.y0c;

      P.clear();
      const bar = (x, v, fill, observed) => {
        const w = 0.32;
        svgEl(
          "rect",
          { x: P.sx(x - w / 2), y: P.sy(v), width: P.sx(w) - P.sx(0), height: P.sy(0) - P.sy(v), fill, "fill-opacity": observed ? 1 : 0.3 },
          P.layer
        );
        svgEl("text", { x: P.sx(x), y: P.sy(v) - 4, "text-anchor": "middle", class: "sx-note", text: pct(v) }, P.over);
      };
      bar(0.3, m.y0c, COLOR.gray, false);
      bar(0.7, m.y1c, COLOR.blue, true);
      bar(1.3, m.y0n, COLOR.gray, true);
      bar(1.7, m.y1n, COLOR.blue, false);
      // Observed comparison: solid bars.
      const yObs = Math.max(m.y1c, m.y0n) + 0.036;
      P.path([[0.7, m.y1c + 0.024], [0.7, yObs], [1.3, yObs], [1.3, m.y0n + 0.024]], {
        stroke: COLOR.ink,
        "stroke-width": 1,
        "stroke-dasharray": "3 3",
      });
      svgEl("text", { x: P.sx(1), y: P.sy(yObs) - 4, "text-anchor": "middle", class: "sx-note", text: `観察された差 ${plus(100 * crude, 1)} ポイント` }, P.over);

      side.textContent = "";
      const lines = [
        `大きな病変の割合：クリップ群 ${pct(q1, 0)}、非クリップ群 ${pct(q0, 0)}`,
        `クリップなしの出血リスク：小さな病変 ${pct(small)}、大きな病変 ${pct(large)}`,
        `クリップの効果（固定）：リスク比 ${RR.small.toFixed(2)}（小）、${RR.large.toFixed(2)}（大）`,
      ];
      lines.forEach((t, i) => svgEl("text", { x: 460, y: 120 + 34 * i, class: "sx-note", text: t.split("：")[0] + "：" }, side));
      lines.forEach((t, i) => svgEl("text", { x: 460, y: 135 + 34 * i, class: "sx-note", text: t.split("：")[1] }, side));

      const gap = m.y0c - m.y0n;
      table([
        ["もともとの差 Y(0)", `クリップ群 ${pct(m.y0c)}、非クリップ群 ${pct(m.y0n)}（差 ${plus(100 * gap, 1)} ポイント）`],
        ["観察された差（粗）", `${plus(100 * crude, 1)} ポイント`],
        ["本当の ATE", `${signed(100 * ate, 1)} ポイント（ATT ${signed(100 * att, 1)} ポイント）`],
        ["層で分けて標準化した差", `${signed(100 * std, 1)} ポイント`],
        ["観察された差と ATT のずれ", `${plus(100 * (crude - att), 1)} ポイント（＝ もともとの差）`],
      ]);
    }
    draw();
  }

  // ---------- 2.2 DAG and the backdoor criterion ----------

  const NODES = {
    age: { label: "年齢", x: 50, y: 125 },
    ht: { label: "高血圧", x: 140, y: 25 },
    at: { label: "抗血栓薬", x: 230, y: 125 },
    size: { label: "病変径", x: 230, y: 205 },
    prox: { label: "近位結腸", x: 230, y: 280 },
    clip: { label: "クリップ", x: 390, y: 205 },
    bleed: { label: "遅発性出血", x: 540, y: 205 },
  };
  const EDGES = [
    ["age", "at"],
    ["age", "ht"],
    ["at", "ht"],
    ["age", "bleed"],
    ["at", "clip"],
    ["at", "bleed"],
    ["size", "clip"],
    ["size", "bleed"],
    ["prox", "clip"],
    ["prox", "bleed"],
    ["clip", "bleed"],
  ];
  const COVARIATES = ["age", "at", "ht", "size", "prox"];

  const has = (a, b) => EDGES.some(([u, v]) => u === a && v === b);
  const children = (a) => EDGES.filter((e) => e[0] === a).map((e) => e[1]);
  function descendants(a) {
    const out = new Set();
    const stack = [a];
    while (stack.length) {
      for (const c of children(stack.pop())) {
        if (!out.has(c)) {
          out.add(c);
          stack.push(c);
        }
      }
    }
    return out;
  }

  // All paths between clip and bleed in the undirected skeleton.
  function allPaths() {
    const nbr = (a) => EDGES.flatMap(([u, v]) => (u === a ? [v] : v === a ? [u] : []));
    const out = [];
    const walk = (path) => {
      const last = path[path.length - 1];
      if (last === "bleed") {
        out.push(path.slice());
        return;
      }
      for (const n of nbr(last)) {
        if (path.includes(n)) continue;
        path.push(n);
        walk(path);
        path.pop();
      }
    };
    walk(["clip"]);
    return out.sort((a, b) => a.length - b.length);
  }

  function pathText(p) {
    let s = NODES[p[0]].label;
    for (let i = 1; i < p.length; i++) s += (has(p[i - 1], p[i]) ? " → " : " ← ") + NODES[p[i]].label;
    return s;
  }

  // Is the path open given the adjustment set? Returns [open, reason].
  function judge(p, adj) {
    const blocks = [];
    const opens = [];
    for (let i = 1; i < p.length - 1; i++) {
      const z = p[i];
      const collider = has(p[i - 1], z) && has(p[i + 1], z);
      const name = NODES[z].label;
      if (collider) {
        const desc = [...descendants(z)].filter((d) => adj.has(d));
        if (adj.has(z)) opens.push(`合流点の${name}を調整したので開く`);
        else if (desc.length) opens.push(`合流点${name}の子孫（${desc.map((d) => NODES[d].label).join("、")}）を調整したので開く`);
        else blocks.push(`合流点の${name}を調整していないので閉じる`);
      } else if (adj.has(z)) {
        const fork = has(z, p[i - 1]) && has(z, p[i + 1]);
        blocks.push(`${fork ? "分岐" : "連鎖"}の${name}を調整したので閉じる`);
      }
    }
    if (blocks.length) return [false, blocks.join("、")];
    if (opens.length) return [true, opens.join("、")];
    const mids = p.slice(1, -1).map((z) => NODES[z].label).join("、");
    return [true, `途中の${mids}を調整していない`];
  }

  function dagWidget(root) {
    const adj = new Set(["at", "size", "prox"]);
    const paths = allPaths();
    const causal = paths.filter((p) => p.every((z, i) => i === 0 || has(p[i - 1], z)));
    const backdoor = paths.filter((p) => has(p[1], p[0]));

    const ctl = controls(root);
    const pre = buttons(ctl, "調整する変数", [
      { label: "本文の組（抗血栓薬、病変径、近位結腸）", onClick: () => set(["at", "size", "prox"]) },
      { label: "なし", onClick: () => set([]) },
      { label: "全部", onClick: () => set(COVARIATES) },
      { label: "年齢と高血圧だけ", onClick: () => set(["age", "ht"]) },
    ]);
    pre.select(0);
    el("p", { class: "sx-note", text: "箱をクリックすると、その変数を調整する・しないが切りかわります。" }, ctl);
    function set(list) {
      adj.clear();
      list.forEach((v) => adj.add(v));
      draw();
    }

    const svg = newSvg(root, 600, 305, "クリップと遅発性出血の DAG");
    const defs = svgEl("defs", {}, svg);
    const marker = (id, color) => {
      const m = svgEl("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerUnits: "userSpaceOnUse", markerWidth: 11, markerHeight: 11, orient: "auto-start-reverse" }, defs);
      svgEl("path", { d: "M0,0 L10,5 L0,10 z", fill: color }, m);
    };
    const uid = Math.random().toString(36).slice(2);
    marker("sx-ar-g-" + uid, COLOR.gray);
    marker("sx-ar-r-" + uid, COLOR.red);
    marker("sx-ar-b-" + uid, COLOR.blue);
    const edgeLayer = svgEl("g", {}, svg);
    const nodeLayer = svgEl("g", {}, svg);
    const boxW = (k) => 18 + 15 * NODES[k].label.length;
    const boxH = 30;

    // Where the segment from a's centre toward b leaves a's box.
    function exitPoint(a, tx, ty) {
      const n = NODES[a];
      const dx = tx - n.x;
      const dy = ty - n.y;
      const hw = boxW(a) / 2 + 3;
      const hh = boxH / 2 + 3;
      const t = Math.min(Math.abs(hw / (dx || 1e-9)), Math.abs(hh / (dy || 1e-9)));
      return [n.x + dx * t, n.y + dy * t];
    }
    // Curved routes for the edges that would cross other boxes.
    const VIA = {
      "age>bleed": [300, 70],
      "at>bleed": [400, 128],
      "size>bleed": [460, 248],
      "prox>bleed": [460, 280],
    };

    function draw() {
      const states = backdoor.map((p) => judge(p, adj));
      const hot = new Set();
      backdoor.forEach((p, i) => {
        if (!states[i][0]) return;
        for (let j = 1; j < p.length; j++) hot.add([p[j - 1], p[j]].sort().join(">"));
      });
      edgeLayer.textContent = "";
      for (const [a, b] of EDGES) {
        const k = a + ">" + b;
        const isHot = hot.has([a, b].sort().join(">"));
        const isCausal = a === "clip" && b === "bleed";
        const color = isCausal ? COLOR.blue : isHot ? COLOR.red : COLOR.gray;
        const name = isCausal ? "b" : isHot ? "r" : "g";
        const via = VIA[k];
        const [x1, y1] = exitPoint(a, via ? via[0] : NODES[b].x, via ? via[1] : NODES[b].y);
        const [x2, y2] = exitPoint(b, via ? via[0] : NODES[a].x, via ? via[1] : NODES[a].y);
        const d = via ? `M${x1},${y1} Q${2 * via[0] - (x1 + x2) / 2},${2 * via[1] - (y1 + y2) / 2} ${x2},${y2}` : `M${x1},${y1} L${x2},${y2}`;
        svgEl(
          "path",
          { d, fill: "none", stroke: color, "stroke-width": isHot || isCausal ? 2.4 : 1.4, "marker-end": `url(#sx-ar-${name}-${uid})` },
          edgeLayer
        );
      }

      nodeLayer.textContent = "";
      for (const [k, n] of Object.entries(NODES)) {
        const g = svgEl("g", {}, nodeLayer);
        const clickable = COVARIATES.includes(k);
        const on = adj.has(k);
        const fill = k === "clip" ? COLOR.blue : k === "bleed" ? COLOR.red : on ? COLOR.ink : "none";
        svgEl(
          "rect",
          {
            x: n.x - boxW(k) / 2,
            y: n.y - boxH / 2,
            width: boxW(k),
            height: boxH,
            rx: 5,
            fill: fill === "none" ? "var(--md-default-bg-color, #fff)" : fill,
            stroke: fill === "none" ? COLOR.gray : fill,
            "stroke-width": 1.5,
          },
          g
        );
        const label = svgEl("text", { x: n.x, y: n.y + 4.5, "text-anchor": "middle", text: n.label, "font-size": 15 }, g);
        if (fill !== "none") label.setAttribute("style", "fill: #fff");
        if (clickable) {
          g.setAttribute("tabindex", "0");
          g.setAttribute("role", "button");
          g.setAttribute("aria-pressed", on ? "true" : "false");
          g.setAttribute("aria-label", n.label + (on ? "（調整する）" : "（調整しない）"));
          g.style.cursor = "pointer";
          const toggle = () => {
            if (adj.has(k)) adj.delete(k);
            else adj.add(k);
            pre.select(-1);
            draw();
          };
          g.addEventListener("click", toggle);
          g.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              toggle();
            }
          });
        }
      }

      const anyOpen = states.some((s) => s[0]);
      const clipDesc = descendants("clip");
      const bad = [...adj].filter((v) => clipDesc.has(v));
      const ok = !anyOpen && bad.length === 0;
      const rows = backdoor.map((p, i) => [`経路 ${i + 1}`, `${pathText(p)}：${states[i][0] ? "開いている" : "閉じている"}（${states[i][1]}）`]);
      const chosen = adj.size ? "{" + COVARIATES.filter((v) => adj.has(v)).map((v) => NODES[v].label).join("、") + "}" : "調整なし";
      rows.push([
        "バックドア基準",
        `${chosen}：${ok ? "満たす。" : "満たさない。"}${anyOpen ? "開いたバックドア経路が残っています。" : "バックドア経路はすべて閉じています。"}${causal.map(pathText).join("、")} はいつも開いていて、これが知りたい効果です。`,
      ]);
      table(rows);
    }
    const table = readout(root);
    draw();
  }

  register("confounding", confoundingWidget);
  register("dag-backdoor", dagWidget);
})();
