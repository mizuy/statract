// Interactive figures, see docs/javascripts/theory-widgets.js.
(function () {
  "use strict";
  const { COLOR, svgEl, rng, controls, slider, readout, panel, newSvg, legend, register } = window.SX;

  // ---------- 11.3 gradient boosting with stumps ----------

  function boostingWidget(root) {
    // A small made-up data set: one continuous predictor, a smooth truth and noise.
    const truth = (x) => Math.sin(x / 1.3) + 0.15 * x;
    const SD = 0.6;
    const N = 60;
    const R = rng(1);
    const xs = [];
    const ys = [];
    for (let i = 0; i < N; i++) {
      const x = 10 * R.uniform();
      xs.push(x);
      ys.push(truth(x) + SD * R.normal());
    }
    const order = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b]);
    const sx = order.map((i) => xs[i]);
    const sy = order.map((i) => ys[i]);
    const grid = [];
    for (let i = 0; i <= 500; i++) grid.push((10 * i) / 500);

    const MAXT = 500;
    // Boosting path for one learning rate: each tree is a stump fitted to the residuals.
    function boost(lr) {
      const F0 = sy.reduce((s, v) => s + v, 0) / N;
      const F = new Array(N).fill(F0);
      const G = grid.map(() => F0);
      const stumps = [];
      const trainErr = [mse(sy, F)];
      const newErr = [newError(G)];
      for (let m = 0; m < MAXT; m++) {
        const r = sy.map((y, i) => y - F[i]);
        const total = r.reduce((s, v) => s + v, 0);
        let best = 1;
        let bestScore = -Infinity;
        let left = 0;
        for (let k = 1; k < N; k++) {
          left += r[k - 1];
          if (sx[k] === sx[k - 1]) continue;
          const right = total - left;
          const score = (left * left) / k + (right * right) / (N - k);
          if (score > bestScore) {
            bestScore = score;
            best = k;
          }
        }
        let L = 0;
        for (let k = 0; k < best; k++) L += r[k];
        const Rm = (total - L) / (N - best);
        L /= best;
        const cut = (sx[best - 1] + sx[best]) / 2;
        const stump = { cut, left: lr * L, right: lr * Rm };
        stumps.push(stump);
        for (let k = 0; k < N; k++) F[k] += k < best ? stump.left : stump.right;
        for (let i = 0; i < grid.length; i++) G[i] += grid[i] < cut ? stump.left : stump.right;
        trainErr.push(mse(sy, F));
        newErr.push(newError(G));
      }
      return { F0, stumps, trainErr, newErr };
    }
    function mse(y, f) {
      let s = 0;
      for (let i = 0; i < y.length; i++) s += (y[i] - f[i]) ** 2;
      return s / y.length;
    }
    // Expected squared error for a new patient: noise plus the distance from the truth.
    function newError(G) {
      let s = 0;
      for (let i = 0; i < grid.length; i++) s += (truth(grid[i]) - G[i]) ** 2;
      return SD * SD + s / grid.length;
    }

    const TREES = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 500];
    const RATES = [0.03, 0.1, 0.3, 1];
    const state = { m: 40, lr: 0.3 };
    const cache = {};
    const pathFor = (lr) => cache[lr] || (cache[lr] = boost(lr));

    const ctl = controls(root);
    slider(ctl, {
      label: "木の数",
      min: 0,
      max: TREES.length - 1,
      step: 1,
      value: TREES.indexOf(state.m),
      toValue: (i) => TREES[i],
      format: (v) => v + " 本",
      onInput: (v) => {
        state.m = v;
        draw();
      },
    });
    slider(ctl, {
      label: "学習率（1 本の木を足す割合）",
      min: 0,
      max: RATES.length - 1,
      step: 1,
      value: RATES.indexOf(state.lr),
      toValue: (i) => RATES[i],
      format: (v) => String(v),
      onInput: (v) => {
        state.lr = v;
        draw();
      },
    });

    const first = pathFor(state.lr);
    const YMAX = Math.ceil(10 * 1.1 * Math.max(first.trainErr[0], first.newErr[0])) / 10;
    const yt = [];
    for (let t = 0; t <= YMAX + 1e-9; t += YMAX > 1.2 ? 0.4 : 0.3) yt.push(Math.round(10 * t) / 10);

    const svg = newSvg(root, 660, 345, "勾配ブースティングの予測と誤差");
    const P = panel(svg, {
      left: 44,
      top: 24,
      width: 330,
      height: 240,
      x: [0, 10],
      y: [-1, 4],
      xticks: [0, 2, 4, 6, 8, 10],
      yticks: [-1, 0, 1, 2, 3, 4],
      xfmt: (t) => String(t),
      yfmt: (t) => (t < 0 ? "−" + -t : String(t)),
      xlab: "x（例として作った説明変数）",
      ylab: "y",
      ylabOffset: 30,
    });
    const E = panel(svg, {
      left: 450,
      top: 24,
      width: 190,
      height: 240,
      x: [0, Math.log10(MAXT + 1)],
      y: [0, YMAX],
      xticks: [0, 10, 100, 500].map((v) => Math.log10(v + 1)),
      yticks: yt,
      xfmt: (t) => String(Math.round(Math.pow(10, t) - 1)),
      yfmt: (t) => t.toFixed(1),
      xlab: "木の数",
      ylab: "誤差（2 乗の平均）",
      ylabOffset: 36,
    });
    // Data points and the truth stay fixed.
    const fixed = svgEl("g", {}, P.layer);
    for (let i = 0; i < N; i++) {
      svgEl("circle", { cx: P.sx(xs[i]), cy: P.sy(ys[i]), r: 3, fill: COLOR.gray, "fill-opacity": 0.75 }, fixed);
    }
    P.path(grid.map((x) => [x, truth(x)]), { stroke: COLOR.ink, "stroke-width": 1.5, "stroke-dasharray": "5 4" }, fixed);
    const fitLayer = svgEl("g", {}, P.layer);

    legend(svg, 46, 326, [{ label: "本当の関係", attrs: { stroke: COLOR.ink, "stroke-width": 1.5, "stroke-dasharray": "5 4" } }]);
    legend(svg, 170, 326, [{ label: "ブースティングの予測", attrs: { stroke: COLOR.blue, "stroke-width": 2.2 } }]);
    legend(svg, 400, 326, [{ label: "作ったデータ", attrs: { stroke: COLOR.gray, "stroke-width": 2 } }]);
    legend(svg, 530, 326, [{ label: "新しいデータ", attrs: { stroke: COLOR.red, "stroke-width": 2 } }]);
    const table = readout(root);
    const lx = (m) => Math.log10(m + 1);

    function draw() {
      const path = pathFor(state.lr);
      const m = state.m;
      // Prediction after m trees, as a step function on the grid.
      const G = grid.map(() => path.F0);
      for (let t = 0; t < m; t++) {
        const s = path.stumps[t];
        for (let i = 0; i < grid.length; i++) G[i] += grid[i] < s.cut ? s.left : s.right;
      }
      fitLayer.textContent = "";
      P.path(grid.map((x, i) => [x, G[i]]), { stroke: COLOR.blue, "stroke-width": 2.2 }, fitLayer);

      E.clear();
      const ms = [];
      for (let t = 0; t <= MAXT; t++) ms.push(t);
      E.path([[0, SD * SD], [lx(MAXT), SD * SD]], { stroke: COLOR.gray, "stroke-width": 0.8, "stroke-dasharray": "2 3" });
      E.path(ms.map((t) => [lx(t), path.trainErr[t]]), { stroke: COLOR.gray, "stroke-width": 2 });
      E.path(ms.map((t) => [lx(t), path.newErr[t]]), { stroke: COLOR.red, "stroke-width": 2 });
      E.path([[lx(m), 0], [lx(m), YMAX]], { stroke: COLOR.blue, "stroke-width": 1.5 });
      svgEl("text", { x: E.sx(lx(MAXT)) - 2, y: E.sy(SD * SD) + 13, "text-anchor": "end", class: "sx-note", text: "偶然のぶれ" }, E.over);

      let bestM = 0;
      for (let t = 1; t <= MAXT; t++) if (path.newErr[t] < path.newErr[bestM]) bestM = t;
      table([
        ["木の数、学習率", `${m} 本、${state.lr}`],
        ["誤差（作ったデータ）", path.trainErr[m].toFixed(3)],
        ["誤差（新しいデータ）", path.newErr[m].toFixed(3)],
        ["新しいデータで最も良い木の数", `${bestM} 本（誤差 ${path.newErr[bestM].toFixed(3)}）`],
        ["偶然のぶれ（これより小さくはできない）", (SD * SD).toFixed(3)],
      ]);
    }
    draw();
  }

  register("boosting", boostingWidget);
})();
