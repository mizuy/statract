// Interactive figures, see docs/javascripts/theory-widgets.js.
(function () {
  "use strict";
  const { COLOR, svgEl, rng, controls, slider, buttons, readout, panel, newSvg, legend, register } = window.SX;

  // ---------- the running example, made as in examples/theory_bleeding/build.py ----------

  const expit1 = (x) => 1 / (1 + Math.exp(-x));

  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

  function trueLogit(age, at, size, prox, clip) {
    const clipEffect = size >= 20 ? -1.0 : -0.2;
    return -3.6 + (0.15 * (age - 70)) / 10 + 0.9 * at + (0.55 * (size - 15)) / 10 + 0.6 * prox + clipEffect * clip;
  }

  function simulate(n, R) {
    const cols = ["age", "male", "at", "htn", "size", "prox", "clip"];
    const d = { n, p: new Float64Array(n) };
    for (const c of cols) d[c] = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const age = Math.round(clamp(68 + 10 * R.normal(), 30, 95));
      const male = R.uniform() < 0.6 ? 1 : 0;
      const at = R.uniform() < expit1(-1.6 + 0.08 * (age - 68)) ? 1 : 0;
      const htn = R.uniform() < expit1(-0.4 + 0.09 * (age - 68) + 1.8 * at) ? 1 : 0;
      const size = Math.round(clamp(Math.exp(Math.log(14) + 0.55 * R.normal()), 5, 80));
      const prox = R.uniform() < 0.55 ? 1 : 0;
      const clip = R.uniform() < expit1(-2.4 + 1.8 * (size >= 20) + 0.08 * (size - 15) + 1.6 * at + 0.8 * prox) ? 1 : 0;
      d.age[i] = age;
      d.male[i] = male;
      d.at[i] = at;
      d.htn[i] = htn;
      d.size[i] = size;
      d.prox[i] = prox;
      d.clip[i] = clip;
      d.p[i] = expit1(trueLogit(age, at, size, prox, clip));
    }
    return d;
  }

  function coin(p, R) {
    const y = new Uint8Array(p.length);
    for (let i = 0; i < p.length; i++) y[i] = R.uniform() < p[i] ? 1 : 0;
    return y;
  }

  function design7(d, i) {
    return [d.age[i], d.male[i], d.at[i], d.htn[i], d.size[i], d.prox[i], d.clip[i]];
  }

  function solveSpd(A, b) {
    const k = b.length;
    const L = A.map((r) => r.slice());
    for (let j = 0; j < k; j++) {
      for (let m = 0; m < j; m++) L[j][j] -= L[j][m] * L[j][m];
      L[j][j] = Math.sqrt(Math.max(L[j][j], 1e-12));
      for (let i = j + 1; i < k; i++) {
        for (let m = 0; m < j; m++) L[i][j] -= L[i][m] * L[j][m];
        L[i][j] /= L[j][j];
      }
    }
    const z = new Array(k).fill(0);
    for (let i = 0; i < k; i++) {
      let s = b[i];
      for (let m = 0; m < i; m++) s -= L[i][m] * z[m];
      z[i] = s / L[i][i];
    }
    const x = new Array(k).fill(0);
    for (let i = k - 1; i >= 0; i--) {
      let s = z[i];
      for (let m = i + 1; m < k; m++) s -= L[m][i] * x[m];
      x[i] = s / L[i][i];
    }
    return x;
  }

  function fitLogistic(X, y, offset) {
    const n = X.length;
    const k = X[0].length + 1;
    let beta = new Array(k).fill(0);
    let ybar = 0;
    for (let i = 0; i < n; i++) ybar += y[i];
    beta[0] = Math.log(Math.max(ybar, 0.5) / Math.max(n - ybar, 0.5));
    for (let it = 0; it < 30; it++) {
      const H = [];
      for (let a = 0; a < k; a++) H.push(new Array(k).fill(0));
      const g = new Array(k).fill(0);
      for (let i = 0; i < n; i++) {
        const xi = X[i];
        let eta = beta[0] + (offset ? offset[i] : 0);
        for (let a = 1; a < k; a++) eta += beta[a] * xi[a - 1];
        const p = expit1(eta);
        const w = p * (1 - p);
        const r = y[i] - p;
        g[0] += r;
        H[0][0] += w;
        for (let a = 1; a < k; a++) {
          const xa = xi[a - 1];
          g[a] += r * xa;
          H[a][0] += w * xa;
          for (let b = 1; b <= a; b++) H[a][b] += w * xa * xi[b - 1];
        }
      }
      for (let a = 0; a < k; a++) for (let b = a + 1; b < k; b++) H[a][b] = H[b][a];
      const step = solveSpd(H, g);
      let size = 0;
      for (let a = 0; a < k; a++) {
        beta[a] += step[a];
        size = Math.max(size, Math.abs(step[a]));
      }
      if (size < 1e-8) break;
    }
    return beta;
  }

  function linpred(beta, X) {
    return X.map((xi) => {
      let eta = beta[0];
      for (let a = 1; a < beta.length; a++) eta += beta[a] * xi[a - 1];
      return eta;
    });
  }

  function auc(y, score) {
    const n = y.length;
    const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => score[a] - score[b]);
    let rankSum = 0;
    let pos = 0;
    let i = 0;
    while (i < n) {
      let j = i;
      while (j + 1 < n && score[idx[j + 1]] === score[idx[i]]) j++;
      const rank = (i + j) / 2 + 1;
      for (let m = i; m <= j; m++) if (y[idx[m]]) {
        rankSum += rank;
        pos++;
      }
      i = j + 1;
    }
    const neg = n - pos;
    return (rankSum - (pos * (pos + 1)) / 2) / (pos * neg);
  }

  function calSlope(y, lp) {
    return fitLogistic(Array.from(lp, (v) => [v]), y)[1];
  }

  // ---------- 8.1 / 8.2 apparent and new-patient performance ----------

  function overfittingWidget(root) {
    const KMAX = 30;
    // New patients: 50,000, with 30 lab values unrelated to bleeding.
    const test = simulate(50000, rng(13));
    const yt = coin(test.p, rng(1013));
    const labsT = labs(test.n, rng(1014));
    const best = auc(yt, test.p);
    // Development data of each size (seeds chosen so that the example looks like the book's table).
    const DEV = { 400: 9370, 2000: 380 };
    const cache = {};

    function labs(n, R) {
      const out = [];
      for (let i = 0; i < n; i++) {
        const r = [];
        for (let j = 0; j < KMAX; j++) r.push(R.normal());
        out.push(r);
      }
      return out;
    }

    function prepare(n) {
      if (cache[n]) return cache[n];
      const s = DEV[n];
      const d = simulate(n, rng(s));
      const y = coin(d.p, rng(s + 1));
      const lab = labs(n, rng(s + 2));
      let events = 0;
      for (let i = 0; i < n; i++) events += y[i];
      const rows = [];
      for (let i = 0; i < n; i++) rows.push(design7(d, i));
      cache[n] = { n, y, lab, rows, events, res: [] };
      return cache[n];
    }

    // Fit the model with 7 + k predictors and measure it on the same and on new patients.
    function evaluate(dev, k) {
      const X = dev.rows.map((r, i) => r.concat(dev.lab[i].slice(0, k)));
      const beta = fitLogistic(X, dev.y);
      const lpDev = linpred(beta, X);
      const lpNew = new Float64Array(test.n);
      for (let i = 0; i < test.n; i++) {
        const x = design7(test, i);
        let e = beta[0];
        for (let j = 0; j < 7; j++) e += beta[j + 1] * x[j];
        for (let j = 0; j < k; j++) e += beta[j + 8] * labsT[i][j];
        lpNew[i] = e;
      }
      return { apparent: auc(dev.y, lpDev), fresh: auc(yt, lpNew), slope: calSlope(yt, lpNew) };
    }

    const state = { n: 400, k: 10 };
    let job = 0;
    const ctl = controls(root);
    buttons(ctl, "作る人数", [
      { label: "400 人", onClick: () => setN(400) },
      { label: "2000 人", onClick: () => setN(2000) },
    ]).select(0);
    slider(ctl, {
      label: "足した関係のない検査値",
      min: 0,
      max: KMAX,
      step: 1,
      value: state.k,
      format: (v) => v + " 個",
      onInput: (v) => {
        state.k = v;
        draw();
      },
    });
    function setN(n) {
      state.n = n;
      run();
    }

    const svg = newSvg(root, 660, 330, "変数の数と、作ったデータと新しい患者での成績");
    const A = panel(svg, {
      left: 50,
      top: 26,
      width: 300,
      height: 230,
      x: [0, KMAX],
      y: [0.5, 0.9],
      xticks: [0, 5, 10, 15, 20, 25, 30],
      yticks: [0.5, 0.6, 0.7, 0.8, 0.9],
      yfmt: (t) => t.toFixed(1),
      xlab: "足した関係のない検査値の数",
      ylab: "C 統計量",
      ylabOffset: 36,
    });
    const B = panel(svg, {
      left: 430,
      top: 26,
      width: 210,
      height: 230,
      x: [0, KMAX],
      y: [0, 1.2],
      xticks: [0, 10, 20, 30],
      yticks: [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2],
      yfmt: (t) => t.toFixed(1),
      xlab: "足した関係のない検査値の数",
      ylab: "較正の傾き（新しい患者）",
      ylabOffset: 34,
    });
    legend(svg, 52, 312, [{ label: "作ったデータ（見かけ）", attrs: { stroke: COLOR.gray, "stroke-width": 2.2 } }]);
    legend(svg, 230, 312, [{ label: "新しい患者 5 万人", attrs: { stroke: COLOR.blue, "stroke-width": 2.2 } }]);
    legend(svg, 400, 312, [{ label: `本当のモデル（${best.toFixed(2)}）`, attrs: { stroke: COLOR.red, "stroke-width": 1.5, "stroke-dasharray": "2 3" } }]);
    const table = readout(root);

    // Compute the 31 models a few at a time and draw as they come in.
    function run() {
      const dev = prepare(state.n);
      const my = ++job;
      const step = () => {
        if (my !== job) return;
        if (dev.res.length <= KMAX) dev.res.push(evaluate(dev, dev.res.length));
        draw();
        if (dev.res.length <= KMAX) setTimeout(step, 0);
      };
      step();
    }

    function draw() {
      const dev = prepare(state.n);
      const res = dev.res;
      const k = state.k;
      A.clear();
      B.clear();
      A.path([[0, best], [KMAX, best]], { stroke: COLOR.red, "stroke-width": 1.5, "stroke-dasharray": "2 3" });
      A.path([[0, 0.5], [KMAX, 0.5]], { stroke: COLOR.gray, "stroke-width": 0.8 });
      B.path([[0, 1], [KMAX, 1]], { stroke: COLOR.gray, "stroke-width": 1, "stroke-dasharray": "4 3" });
      A.path(res.map((r, j) => [j, r.apparent]), { stroke: COLOR.gray, "stroke-width": 2.2 });
      A.path(res.map((r, j) => [j, r.fresh]), { stroke: COLOR.blue, "stroke-width": 2.2 });
      B.path(res.map((r, j) => [j, r.slope]), { stroke: COLOR.blue, "stroke-width": 2.2 });
      for (const P of [A, B]) {
        svgEl("line", { x1: P.sx(k), x2: P.sx(k), y1: P.sy(P === A ? 0.9 : 1.2), y2: P.sy(P === A ? 0.5 : 0), class: "sx-guide" }, P.layer);
      }
      const r = res[k];
      if (r) {
        svgEl("circle", { cx: A.sx(k), cy: A.sy(r.apparent), r: 5, fill: COLOR.gray, stroke: "#fff", "stroke-width": 1.5 }, A.over);
        svgEl("circle", { cx: A.sx(k), cy: A.sy(r.fresh), r: 5, fill: COLOR.blue, stroke: "#fff", "stroke-width": 1.5 }, A.over);
        svgEl("circle", { cx: B.sx(k), cy: B.sy(Math.min(r.slope, 1.2)), r: 5, fill: COLOR.blue, stroke: "#fff", "stroke-width": 1.5 }, B.over);
      }
      const p = 7 + k;
      const wait = "計算中";
      table([
        ["作ったデータ", `${dev.n} 人（出血 ${dev.events} 人）`],
        ["係数の数", `${p}（七つの変数 + 検査値 ${k} 個）、1 係数あたりの出血数 ${(dev.events / p).toFixed(1)}`],
        ["C 統計量（作ったデータ）", r ? r.apparent.toFixed(2) : wait],
        ["C 統計量（新しい患者）", r ? `${r.fresh.toFixed(2)}（本当のモデルでは ${best.toFixed(2)}）` : wait],
        ["較正の傾き（新しい患者）", r ? r.slope.toFixed(2) : wait],
      ]);
    }
    run();
  }

  register("overfitting", overfittingWidget);
})();
