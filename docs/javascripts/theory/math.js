// Interactive figures, see docs/javascripts/theory-widgets.js.
(function () {
  "use strict";
  const { COLOR, el, svgEl, pct, range, signed, expit, logit, rng, normCdf, controls, slider, buttons, readout, panel, newSvg, legend } = window.SX;

  const trim = (x, d) => String(Number(x.toFixed(d)));
  // Round ticks from 0 to about max (the last tick may stop short of max).
  function niceTicks(max) {
    const raw = max / 4;
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const m = raw / p;
    const step = (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
    const out = [];
    for (let t = 0; t <= max + 1e-12; t += step) out.push(Math.round(t / step) * step);
    return out;
  }
  const Z975 = 1.959963984540054;

  // ---------- A1 odds, odds ratio and risk ratio ----------

  function oddsWidget(root) {
    // Default: bleeding with and without antithrombotics in the 3000 patients (A2).
    const state = { p1: 51 / 616, p0: 85 / 2384 };
    const ctl = controls(root);
    const s1 = slider(ctl, {
      label: "群 1 の出血の確率",
      min: 0.001,
      max: 0.99,
      step: 0.001,
      value: state.p1,
      format: (v) => pct(v),
      onInput: (v) => {
        state.p1 = v;
        pre.select(-1);
        draw();
      },
    });
    const s0 = slider(ctl, {
      label: "群 0 の出血の確率",
      min: 0.001,
      max: 0.99,
      step: 0.001,
      value: state.p0,
      format: (v) => pct(v),
      onInput: (v) => {
        state.p0 = v;
        pre.select(-1);
        draw();
      },
    });
    const pre = buttons(ctl, "例", [
      { label: "抗血栓薬あり / なし（8.3%、3.6%）", onClick: () => set(51 / 616, 85 / 2384) },
      { label: "まれ（2%、1%）", onClick: () => set(0.02, 0.01) },
      { label: "多い（60%、30%）", onClick: () => set(0.6, 0.3) },
    ]);
    pre.select(0);
    function set(a, b) {
      state.p1 = a;
      state.p0 = b;
      s1.set(a);
      s0.set(b);
      draw();
    }

    const svg = newSvg(root, 680, 320, "確率とオッズ、オッズ比とリスク比");
    const L = panel(svg, {
      left: 50,
      top: 26,
      width: 250,
      height: 230,
      x: [0, 1],
      y: [0, 4],
      xticks: [0, 0.2, 0.4, 0.6, 0.8, 1],
      yticks: [0, 1, 2, 3, 4],
      xfmt: (t) => Math.round(100 * t) + "%",
      xlab: "確率 p",
      ylab: "オッズ p / (1 − p)",
      ylabOffset: 32,
      title: "確率とオッズ（赤：群 1、青：群 0）",
    });
    const ps = [];
    for (let i = 0; i <= 400; i++) ps.push((0.99 * i) / 400);
    L.path(ps.map((p) => [p, p]), { stroke: COLOR.gray, "stroke-width": 1, "stroke-dasharray": "4 3" });
    L.path(ps.map((p) => [p, p / (1 - p)]), { stroke: COLOR.ink, "stroke-width": 2 });
    svgEl("text", { x: L.sx(0.62), y: L.sy(0.5), class: "sx-note", fill: COLOR.gray, text: "オッズ = p の線" }, svg);

    const rightLayer = svgEl("g", {}, svg);
    const table = readout(root);

    function draw() {
      const { p1, p0 } = state;
      const o1 = p1 / (1 - p1);
      const o0 = p0 / (1 - p0);
      const rr = p1 / p0;
      const or = o1 / o0;

      L.over.textContent = "";
      [[p1, o1, COLOR.red, "1"], [p0, o0, COLOR.blue, "0"]].forEach(([p, o, c, name]) => {
        if (o <= 4) {
          svgEl("circle", { cx: L.sx(p), cy: L.sy(o), r: 5, fill: c, stroke: "#fff", "stroke-width": 1 }, L.over);
        } else {
          svgEl("text", { x: L.sx(p), y: L.sy(4) + 12, "text-anchor": "middle", class: "sx-note", fill: c, text: "↑ 群 " + name }, L.over);
        }
      });

      // Right: with this risk ratio held fixed, the odds ratio as group 0's risk grows.
      rightLayer.textContent = "";
      const pmax = Math.min(0.99, 0.99 / Math.max(rr, 1));
      const curve = [];
      for (let i = 1; i <= 300; i++) {
        const q = (pmax * i) / 300;
        const a = rr * q;
        curve.push([q, (a / (1 - a)) / (q / (1 - q))]);
      }
      const cmin = Math.min(...curve.map((c) => c[1]));
      const cmax = Math.max(...curve.map((c) => c[1]));
      const ylo = Math.max(0.005, Math.min(rr, or, cmin) / 1.3);
      const yhi = Math.max(Math.min(cmax, Math.max(rr, or) * 4), Math.max(rr, or) * 1.3);
      const logTicks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        .filter((t) => t >= ylo && t <= yhi && Math.abs(Math.log(t / rr)) > 0.25)
        .concat([rr]);
      const xs = niceTicks(pmax);
      const R = panel(rightLayer, {
        left: 400,
        top: 26,
        width: 260,
        height: 230,
        x: [0, pmax],
        y: [Math.log(ylo), Math.log(yhi)],
        xticks: xs,
        yticks: logTicks.map(Math.log),
        xfmt: (t) => trim(100 * t, 1) + "%",
        yfmt: (t) => trim(Math.exp(t), 2),
        xlab: "群 0 の確率（リスク比はそのまま）",
        ylabOffset: 34,
        title: `リスク比 ${rr.toFixed(2)} のときのオッズ比`,
      });
      R.path([[0, Math.log(rr)], [pmax, Math.log(rr)]], { stroke: COLOR.gray, "stroke-width": 1.5, "stroke-dasharray": "5 4" });
      R.path(curve.map(([q, v]) => [q, Math.log(v)]), { stroke: COLOR.ink, "stroke-width": 2 });
      svgEl("circle", { cx: R.sx(p0), cy: R.sy(Math.log(or)), r: 5, fill: COLOR.red, stroke: "#fff", "stroke-width": 1 }, R.over);
      svgEl("text", { x: R.sx(0) + 6, y: R.sy(Math.log(rr)) + (or >= rr ? 14 : -6), class: "sx-note", fill: COLOR.gray, text: "リスク比" }, R.over);

      const lab = (v) => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2));
      table([
        ["リスク差", (100 * (p1 - p0) >= 0 ? "" : "−") + Math.abs(100 * (p1 - p0)).toFixed(1) + " ポイント"],
        ["リスク比", lab(rr)],
        ["オッズ（群 1、群 0）", `${o1.toFixed(4)}、${o0.toFixed(4)}`],
        ["オッズ比", `${lab(or)}（リスク比との違い ${signed(100 * (or / rr - 1), 0)}%）`],
        ["対数オッズ（群 1、群 0）", `${signed(logit(p1), 2)}、${signed(logit(p0), 2)}（差 ${signed(Math.log(or), 2)} = log オッズ比）`],
      ]);
    }
    draw();
  }

  // ---------- A2 positive predictive value ----------

  function ppvWidget(root) {
    const state = { prev: 0.044, sens: 0.52, spec: 0.73 };
    const ctl = controls(root);
    const mk = (key, label, min, max, step, digits) =>
      slider(ctl, {
        label,
        min,
        max,
        step,
        value: state[key],
        format: (v) => pct(v, digits),
        onInput: (v) => {
          state[key] = v;
          draw();
        },
      });
    mk("prev", "出血割合（有病割合）", 0.001, 0.5, 0.001, 1);
    mk("sens", "感度", 0.01, 1, 0.01, 0);
    mk("spec", "特異度", 0.01, 1, 0.01, 0);

    const COLS = 50;
    const ROWS = 20;
    const CELL = 9.6;
    const svg = newSvg(root, 680, 236, "1000 人の内訳");
    const gx = 20;
    const gy = 30;
    svgEl("text", { x: gx, y: 18, class: "sx-title", text: "1000 人に当てはめると（1 マスが 1 人）" }, svg);
    const grid = svgEl("g", {}, svg);
    const cells = [];
    for (let i = 0; i < COLS * ROWS; i++) {
      const c = Math.floor(i / ROWS);
      const r = i % ROWS;
      cells.push(svgEl("rect", { x: gx + c * CELL, y: gy + r * CELL, width: CELL - 1.4, height: CELL - 1.4, rx: 1 }, grid));
    }
    const KINDS = [
      { key: "tp", label: "出血する・陽性", fill: COLOR.red, op: 1 },
      { key: "fn", label: "出血する・陰性", fill: COLOR.red, op: 0.3 },
      { key: "fp", label: "出血しない・陽性", fill: COLOR.blue, op: 1 },
      { key: "tn", label: "出血しない・陰性", fill: COLOR.blue, op: 0.18 },
    ];
    const lx = gx + COLS * CELL + 18;
    const legendText = KINDS.map((k, i) => {
      svgEl("rect", { x: lx, y: gy + i * 24, width: 12, height: 12, rx: 1, fill: k.fill, "fill-opacity": k.op }, svg);
      return svgEl("text", { x: lx + 18, y: gy + i * 24 + 10, text: k.label }, svg);
    });
    const big = svgEl("text", { x: lx, y: gy + 4 * 24 + 22, class: "sx-title" }, svg);
    const big2 = svgEl("text", { x: lx, y: gy + 4 * 24 + 40, class: "sx-note" }, svg);
    const table = readout(root);

    function draw() {
      const { prev, sens, spec } = state;
      const exact = {
        tp: 1000 * prev * sens,
        fn: 1000 * prev * (1 - sens),
        fp: 1000 * (1 - prev) * (1 - spec),
        tn: 1000 * (1 - prev) * spec,
      };
      // Round the counts so that they add up to 1000 within each disease group.
      const d = Math.round(1000 * prev);
      const tp = Math.round(d * sens);
      const fp = Math.round((1000 - d) * (1 - spec));
      const n = { tp, fn: d - tp, fp, tn: 1000 - d - fp };
      let i = 0;
      KINDS.forEach((k, j) => {
        for (let m = 0; m < n[k.key]; m++, i++) {
          cells[i].setAttribute("fill", k.fill);
          cells[i].setAttribute("fill-opacity", k.op);
        }
        legendText[j].textContent = `${k.label} ${n[k.key]} 人`;
      });
      const ppv = exact.tp / (exact.tp + exact.fp);
      const npv = exact.tn / (exact.tn + exact.fn);
      big.textContent = `陽性 ${n.tp + n.fp} 人のうち出血 ${n.tp} 人`;
      big2.textContent = `陽性的中率 ${pct(ppv)}`;
      table([
        ["1000 人のうち出血する人", `${d} 人`],
        ["陽性と判定される人", `${n.tp + n.fp} 人（出血する ${n.tp} 人 + 出血しない ${n.fp} 人）`],
        ["陽性的中率 P(出血 | 陽性)", pct(ppv)],
        ["陰性的中率 P(出血しない | 陰性)", pct(npv)],
      ]);
    }
    draw();
  }

  // ---------- A3 standard error and 1.96 ----------

  function samplingWidget(root) {
    const SIZES = [20, 50, 100, 300, 1000, 3000];
    const state = { n: 3000, p: 0.045, draws: [] };
    let gen = rng(3);
    const ctl = controls(root);
    slider(ctl, {
      label: "人数 n",
      min: 0,
      max: SIZES.length - 1,
      step: 1,
      value: SIZES.indexOf(state.n),
      toValue: (i) => SIZES[i],
      format: (v) => v + " 人",
      onInput: (v) => {
        state.n = v;
        reset();
      },
    });
    slider(ctl, {
      label: "本当の出血割合 p",
      min: 0.005,
      max: 0.5,
      step: 0.005,
      value: state.p,
      format: (v) => pct(v),
      onInput: (v) => {
        state.p = v;
        reset();
      },
    });
    buttons(ctl, "データを", [
      { label: "1000 回取り直す", onClick: () => more(1000) },
      { label: "はじめから", onClick: () => reset() },
    ]);

    function more(k) {
      const { n, p } = state;
      for (let j = 0; j < k; j++) {
        let y = 0;
        for (let i = 0; i < n; i++) if (gen.uniform() < p) y++;
        state.draws.push(y);
      }
      draw();
    }
    function reset() {
      gen = rng(3);
      state.draws = [];
      more(1000);
    }

    const svg = newSvg(root, 640, 320, "標本の出血割合のばらつき");
    const plotLayer = svgEl("g", {}, svg);
    legend(svg, 480, 40, [
      { label: "正規分布", attrs: { stroke: COLOR.red, "stroke-width": 2 } },
      { label: "p ± 1.96 × 標準誤差", attrs: { stroke: COLOR.ink, "stroke-width": 1.5, "stroke-dasharray": "4 3" } },
    ]);
    const table = readout(root);

    function draw() {
      const { n, p, draws } = state;
      const se = Math.sqrt((p * (1 - p)) / n);
      const lo = Math.max(0, p - 4.2 * se);
      const hi = Math.min(1, p + 4.2 * se);
      // Bins on whole numbers of patients.
      const cLo = Math.floor(lo * n);
      const cHi = Math.ceil(hi * n);
      const width = Math.max(1, Math.ceil((cHi - cLo + 1) / 40));
      const nb = Math.ceil((cHi - cLo + 1) / width);
      const counts = new Array(nb).fill(0);
      let inside = 0;
      for (const y of draws) {
        const b = Math.floor((y - cLo) / width);
        if (b >= 0 && b < nb) counts[b]++;
        if (Math.abs(y / n - p) <= Z975 * se + 1e-12) inside++;
      }
      const m = draws.reduce((s, v) => s + v / n, 0) / draws.length;
      const sd = Math.sqrt(draws.reduce((s, v) => s + (v / n - m) ** 2, 0) / (draws.length - 1));
      const binW = width / n;
      // Normal curve scaled to the histogram (count per bin).
      const nd = (x) => (draws.length * binW * Math.exp(-0.5 * ((x - p) / se) ** 2)) / (se * Math.sqrt(2 * Math.PI));
      const ymax = 1.15 * Math.max(nd(p), ...counts);
      const xLo = (cLo - 0.5) / n;
      const xHi = (cLo + nb * width - 0.5) / n;
      const stepPct = (xHi - xLo) * 100 > 12 ? 5 : (xHi - xLo) * 100 > 5 ? 2 : (xHi - xLo) * 100 > 2.4 ? 1 : 0.5;
      const xticks = [];
      for (let t = Math.ceil((100 * xLo) / stepPct) * stepPct; t <= 100 * xHi + 1e-9; t += stepPct) xticks.push(t / 100);

      plotLayer.textContent = "";
      const P = panel(plotLayer, {
        left: 56,
        top: 24,
        width: 400,
        height: 240,
        x: [xLo, xHi],
        y: [0, ymax],
        xticks,
        yticks: [],
        xfmt: (t) => trim(100 * t, 1) + "%",
        xlab: `${n} 人の標本の出血割合`,
        ylab: "回数",
        ylabOffset: 20,
      });
      counts.forEach((c, b) => {
        if (!c) return;
        const x0 = (cLo + b * width - 0.5) / n;
        const x1 = x0 + binW;
        svgEl("rect", { x: P.sx(x0) + 0.5, y: P.sy(c), width: Math.max(0.5, P.sx(x1) - P.sx(x0) - 1), height: P.sy(0) - P.sy(c), fill: COLOR.blue, "fill-opacity": 0.55 }, P.layer);
      });
      const xs = [];
      for (let i = 0; i <= 300; i++) xs.push(xLo + ((xHi - xLo) * i) / 300);
      P.path(xs.map((x) => [x, nd(x)]), { stroke: COLOR.red, "stroke-width": 2 });
      for (const x of [p - Z975 * se, p + Z975 * se]) {
        svgEl("line", { x1: P.sx(x), x2: P.sx(x), y1: P.sy(0), y2: P.sy(ymax), stroke: COLOR.ink, "stroke-width": 1.5, "stroke-dasharray": "4 3" }, P.layer);
      }

      table([
        ["標準誤差 √(p(1 − p)/n)", pct(se, 2)],
        ["取り直した回数", `${draws.length} 回`],
        ["取り直した割合の標準偏差", pct(sd, 2)],
        ["p ± 1.96 × 標準誤差に入った割合", `${pct(inside / draws.length)}（${range(Math.max(0, p - Z975 * se), p + Z975 * se, 2)}）`],
      ]);
    }
    reset();
  }

  // ---------- A4 Newton's method ----------

  function newtonWidget(root) {
    // Log-odds parametrization, as in the table of A4: 136 bleeds in 3000.
    const n = 3000;
    const d = 136;
    const ll = (t) => d * t - n * (t > 0 ? t + Math.log1p(Math.exp(-t)) : Math.log1p(Math.exp(t)));
    const g = (t) => d - n * expit(t);
    const h = (t) => -n * expit(t) * (1 - expit(t));
    const state = { start: 0, path: [0] };

    const ctl = controls(root);
    slider(ctl, {
      label: "始める値 θ₀",
      min: -6,
      max: 1.5,
      step: 0.1,
      value: state.start,
      format: (v) => signed(v, 1),
      onInput: (v) => {
        state.start = v;
        state.path = [v];
        draw();
      },
    });
    buttons(ctl, "ニュートン法を", [
      { label: "1 回進める", onClick: () => step() },
      { label: "はじめから", onClick: () => { state.path = [state.start]; draw(); } },
    ]);
    function step() {
      const t = state.path[state.path.length - 1];
      if (!isFinite(t) || Math.abs(t) > 50 || state.path.length > 40) return;
      state.path.push(t - g(t) / h(t));
      draw();
    }

    const X0 = -6.5;
    const X1 = 2;
    const Y0 = -4500;
    const Y1 = -300;
    const svg = newSvg(root, 640, 320, "ニュートン法で対数尤度の頂上を探す");
    const P = panel(svg, {
      left: 66,
      top: 24,
      width: 420,
      height: 250,
      x: [X0, X1],
      y: [Y0, Y1],
      xticks: [-6, -5, -4, -3, -2, -1, 0, 1, 2],
      yticks: [-4000, -3000, -2000, -1000],
      xfmt: (t) => signed(t, 0),
      yfmt: (t) => signed(t, 0),
      xlab: "対数オッズ θ",
      ylab: "対数尤度",
      ylabOffset: 52,
    });
    const ts = [];
    for (let i = 0; i <= 500; i++) ts.push(X0 + ((X1 - X0) * i) / 500);
    P.path(ts.map((t) => [t, ll(t)]), { stroke: COLOR.blue, "stroke-width": 2.5 });
    const top = Math.log(d / (n - d));
    svgEl("line", { x1: P.sx(top), x2: P.sx(top), y1: P.sy(Y1), y2: P.sy(Y0), class: "sx-guide" }, P.layer);
    legend(svg, 500, 40, [
      { label: "対数尤度", attrs: { stroke: COLOR.blue, "stroke-width": 2.5 } },
      { label: "今の場所での放物線", attrs: { stroke: COLOR.orange, "stroke-width": 1.8, "stroke-dasharray": "5 4" } },
    ]);
    const msg = svgEl("text", { x: 500, y: 90, class: "sx-note" }, svg);
    const tableBox = el("div", {}, root);
    const table = readout(tableBox);

    function draw() {
      P.over.textContent = "";
      const path = state.path;
      const t = path[path.length - 1];
      const ok = isFinite(t) && Math.abs(t) <= 50;
      msg.textContent = "";
      if (ok) {
        const a = ll(t);
        const b = g(t);
        const c = h(t);
        const q = ts.map((x) => [x, a + b * (x - t) + 0.5 * c * (x - t) ** 2]);
        P.path(q, { stroke: COLOR.orange, "stroke-width": 1.8, "stroke-dasharray": "5 4" }, P.over);
        const next = t - b / c;
        if (next >= X0 && next <= X1) {
          const qn = a + b * (next - t) + 0.5 * c * (next - t) ** 2;
          if (qn <= Y1 && qn >= Y0) {
            svgEl("circle", { cx: P.sx(next), cy: P.sy(qn), r: 4, fill: "none", stroke: COLOR.orange, "stroke-width": 1.8 }, P.over);
          }
          svgEl("line", { x1: P.sx(next), x2: P.sx(next), y1: P.sy(Math.min(Y1, Math.max(Y0, qn))), y2: P.sy(Math.max(Y0, ll(next))), stroke: COLOR.orange, "stroke-width": 1, "stroke-dasharray": "2 3" }, P.over);
        } else {
          msg.textContent = "次の値は図の外に跳びます";
        }
      } else {
        msg.textContent = "θ が発散しました";
      }
      path.forEach((x, i) => {
        if (!isFinite(x) || x < X0 || x > X1) return;
        const y = ll(x);
        if (y < Y0) return;
        svgEl("circle", { cx: P.sx(x), cy: P.sy(y), r: i === path.length - 1 ? 5.5 : 4, fill: i === path.length - 1 ? COLOR.red : COLOR.ink, stroke: "#fff", "stroke-width": 1 }, P.over);
        svgEl("text", { x: P.sx(x), y: P.sy(y) - 9, "text-anchor": "middle", class: "sx-note", text: String(i) }, P.over);
      });

      const fmtLl = (v) => signed(v, Math.abs(v - ll(top)) < 0.05 ? 3 : 1);
      const fmtP = (p) => (p < 0.0001 ? "0.00%" : pct(p, Math.abs(p - d / n) < 0.003 ? 2 : 1));
      const shown = path.map((x, i) => [i, x]).slice(-6);
      table(
        shown.map(([i, x]) => [
          "反復 " + i,
          isFinite(x) && Math.abs(x) <= 50
            ? `θ = ${signed(x, Math.abs(x) < 1e-9 ? 0 : 3)}、p = ${fmtP(expit(x))}、対数尤度 ${fmtLl(ll(x))}`
            : "発散",
        ])
      );
    }
    draw();
  }

  // ---------- A5 confidence ellipse and SE of a combination ----------

  function ellipseWidget(root) {
    // Logistic regression of the 3000 patients with the seven predictors of
    // chapter 7 and clip:large (size 20 mm or more); computed with fit_glm.
    const EST = [-0.6547, -0.2073];
    const state = { se1: 0.3263, se2: 0.4129, r: -0.684, sign: 1 };
    const ctl = controls(root);
    const mk = (key, label, min, max, step, fmt) =>
      slider(ctl, {
        label,
        min,
        max,
        step,
        value: state[key],
        format: fmt,
        onInput: (v) => {
          state[key] = v;
          draw();
        },
      });
    mk("se1", "クリップの係数の標準誤差", 0.05, 1, 0.01, (v) => v.toFixed(2));
    mk("se2", "交互作用の係数の標準誤差", 0.05, 1, 0.01, (v) => v.toFixed(2));
    mk("r", "二つの係数の相関", -0.95, 0.95, 0.01, (v) => signed(v, 2));
    buttons(ctl, "組み合わせ", [
      { label: "和（20 mm 以上でのクリップの効果）", onClick: () => { state.sign = 1; draw(); } },
      { label: "差", onClick: () => { state.sign = -1; draw(); } },
    ]).select(0);

    const W = 3.4;
    const svg = newSvg(root, 640, 360, "二つの係数の信頼楕円");
    const P = panel(svg, {
      left: 62,
      top: 20,
      width: 300,
      height: 300,
      x: [EST[0] - W / 2, EST[0] + W / 2],
      y: [EST[1] - W / 2, EST[1] + W / 2],
      xticks: [-2, -1.5, -1, -0.5, 0, 0.5, 1],
      yticks: [-1.5, -1, -0.5, 0, 0.5, 1, 1.5],
      xfmt: (t) => signed(t, 1),
      yfmt: (t) => signed(t, 1),
      xlab: "クリップの係数",
      ylab: "交互作用の係数",
      ylabOffset: 42,
    });
    legend(svg, 385, 40, [
      { label: "信頼楕円（1.96 × 標準誤差）", attrs: { stroke: COLOR.blue, "stroke-width": 2.2 } },
      { label: "組み合わせの 95% 信頼区間", attrs: { stroke: COLOR.red, "stroke-width": 1.5, "stroke-dasharray": "5 4" } },
      { label: "共分散を無視した区間", attrs: { stroke: COLOR.gray, "stroke-width": 1.2, "stroke-dasharray": "2 3" } },
    ]);
    const n1 = svgEl("text", { x: 385, y: 110, class: "sx-note" }, svg);
    const n2 = svgEl("text", { x: 385, y: 126, class: "sx-note" }, svg);
    const table = readout(root);

    function draw() {
      const { se1, se2, r, sign } = state;
      const v11 = se1 * se1;
      const v22 = se2 * se2;
      const v12 = r * se1 * se2;
      // Ellipse of radius 1.96 standard errors: its tangent lines in any
      // direction give the 95% interval of that combination of the two coefficients.
      const c = Z975;
      const l21 = v12 / se1;
      const l22 = Math.sqrt(Math.max(v22 - l21 * l21, 0));
      const pts = [];
      for (let i = 0; i <= 240; i++) {
        const a = (2 * Math.PI * i) / 240;
        const u = c * Math.cos(a);
        const w = c * Math.sin(a);
        pts.push([EST[0] + se1 * u, EST[1] + l21 * u + l22 * w]);
      }
      P.clear();
      svgEl("line", { x1: P.sx(0), x2: P.sx(0), y1: P.sy(EST[1] + W / 2), y2: P.sy(EST[1] - W / 2), class: "sx-guide" }, P.layer);
      svgEl("line", { x1: P.sx(EST[0] - W / 2), x2: P.sx(EST[0] + W / 2), y1: P.sy(0), y2: P.sy(0), class: "sx-guide" }, P.layer);
      P.path(pts, { stroke: COLOR.blue, "stroke-width": 2.2, fill: COLOR.blue, "fill-opacity": 0.08 });
      svgEl("circle", { cx: P.sx(EST[0]), cy: P.sy(EST[1]), r: 4, fill: COLOR.ink }, P.layer);

      const est = EST[0] + sign * EST[1];
      const varC = v11 + v22 + 2 * sign * v12;
      const se = Math.sqrt(Math.max(varC, 0));
      const seNaive = Math.sqrt(v11 + v22);
      // Lines b = sign × (k − a), for the ends of the intervals.
      const line = (k, attrs) => {
        const xa = EST[0] - W;
        const xb = EST[0] + W;
        P.path([[xa, sign * (k - xa)], [xb, sign * (k - xb)]], attrs);
      };
      for (const s of [-1, 1]) {
        line(est + s * Z975 * seNaive, { stroke: COLOR.gray, "stroke-width": 1.2, "stroke-dasharray": "2 3" });
        line(est + s * Z975 * se, { stroke: COLOR.red, "stroke-width": 1.5, "stroke-dasharray": "5 4" });
      }
      n1.textContent = sign > 0 ? "斜めの線の上では、和が一定です。" : "斜めの線の上では、差が一定です。";
      n2.textContent = "区間の線は楕円に接します。";

      const what = sign > 0 ? "和" : "差";
      const ci = (s) => `${Math.exp(est - Z975 * s).toFixed(2)}–${Math.exp(est + Z975 * s).toFixed(2)}`;
      table([
        ["V₁₁、V₂₂、V₁₂", `${v11.toFixed(3)}、${v22.toFixed(3)}、${signed(v12, 3)}`],
        [`${what}の分散 V₁₁ + V₂₂ ${sign > 0 ? "+" : "−"} 2V₁₂`, varC.toFixed(3)],
        [`${what}の標準誤差`, `${se.toFixed(3)}（共分散を無視すると ${seNaive.toFixed(3)}）`],
        [`${what}（オッズ比）と 95% 信頼区間`, `${Math.exp(est).toFixed(2)}（${ci(se)}、共分散を無視すると ${ci(seNaive)}）`],
      ]);
    }
    draw();
  }

  window.SX.register("odds-rr", oddsWidget);
  window.SX.register("ppv", ppvWidget);
  window.SX.register("sampling", samplingWidget);
  window.SX.register("newton", newtonWidget);
  window.SX.register("ellipse", ellipseWidget);
})();
