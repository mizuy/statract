// Interactive figures, see docs/javascripts/theory-widgets.js.
(function () {
  "use strict";
  const { COLOR, svgEl, pct, signed, rng, bisect, loadCsv, controls, slider, buttons, readout, panel, newSvg, legend, register } = window.SX;

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

  // SX.loadCsv splits on every comma; ch9_metrics.csv has quoted names with commas.
  async function loadQuotedCsv(name) {
    const config = document.getElementById("__config");
    let base = ".";
    try {
      base = (config && JSON.parse(config.textContent).base) || ".";
    } catch (e) {
      base = ".";
    }
    const url = new URL("examples/assets/theory_bleeding/" + name, new URL(base.replace(/\/?$/, "/"), window.location.href));
    const response = await fetch(url);
    if (!response.ok) throw new Error("cannot load " + url);
    const split = (line) => (line.match(/("([^"]|"")*"|[^,]*)(,|$)/g) || []).map((c) => c.replace(/,$/, "").replace(/^"|"$/g, "").replace(/""/g, '"'));
    const lines = (await response.text()).trim().split(/\r?\n/);
    const header = split(lines[0]);
    return lines.slice(1).map((line) => {
      const cells = split(line);
      const row = {};
      header.forEach((h, i) => {
        const v = cells[i];
        row[h] = v !== "" && !isNaN(Number(v)) ? Number(v) : v;
      });
      return row;
    });
  }

  // New patients for chapter 9: 50,000 made the same way as the book's data, with
  // the chapter 7 model's predictions. The intercept is set so that the mean
  // predicted risk equals the book's (ch9_metrics.csv). Shared by both widgets.
  const TEST_SEED = 190;
  let shared = null;
  function newPatients() {
    if (shared) return shared;
    shared = (async () => {
      const model = await loadCsv("ch7_model.csv");
      const metrics = await loadQuotedCsv("ch9_metrics.csv");
      const b = model.map((r) => Math.log(r.exp_estimate));
      const meanPred = metrics.find((r) => /^Seven predictors, same/.test(r.model)).mean_predicted;
      const d = simulate(50000, rng(TEST_SEED));
      const lp0 = new Float64Array(d.n);
      for (let i = 0; i < d.n; i++) {
        const x = design7(d, i);
        let e = 0;
        for (let j = 0; j < 7; j++) e += b[j] * x[j];
        lp0[i] = e;
      }
      const meanAt = (a) => {
        let s = 0;
        for (let i = 0; i < d.n; i++) s += expit1(a + lp0[i]);
        return s / d.n - meanPred;
      };
      const a = bisect(meanAt, -10, 0);
      const lp = lp0.map((e) => a + e);
      const y = coin(d.p, rng(1000 + TEST_SEED));
      // Another hospital: same patients and effects, baseline log odds 0.8 higher.
      const R = rng(2000 + TEST_SEED);
      const yOther = new Uint8Array(d.n);
      for (let i = 0; i < d.n; i++) {
        const risk = expit1(trueLogit(d.age[i], d.at[i], d.size[i], d.prox[i], d.clip[i]) + 0.8);
        yOther[i] = R.uniform() < risk ? 1 : 0;
      }
      return { lp, y, yOther, metrics };
    })();
    return shared;
  }

  // ---------- 9.1 threshold, sensitivity and specificity, ROC curve ----------

  async function rocThresholdWidget(root) {
    const { lp, y } = await newPatients();
    const n = y.length;
    const p = Array.from(lp, expit1);
    const pos = [];
    const neg = [];
    for (let i = 0; i < n; i++) (y[i] ? pos : neg).push(p[i]);
    pos.sort((a, b) => a - b);
    neg.sort((a, b) => a - b);
    // Number of values >= t in a sorted array.
    const atLeast = (arr, t) => {
      let lo = 0;
      let hi = arr.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (arr[mid] < t) lo = mid + 1;
        else hi = mid;
      }
      return arr.length - lo;
    };
    const cstat = auc(y, lp);

    const TH = [];
    for (let v = 5; v <= 300; v += 5) TH.push(v / 1000);
    const state = { t: 0.05 };
    const ctl = controls(root);
    const tSlider = slider(ctl, {
      label: "閾値（この確率以上を陽性）",
      min: 0,
      max: TH.length - 1,
      step: 1,
      value: TH.indexOf(state.t),
      toValue: (i) => TH[i],
      format: (v) => pct(v, 1),
      onInput: (v) => {
        state.t = v;
        pre.select([0.02, 0.05, 0.1, 0.2].indexOf(v));
        draw();
      },
    });
    const pre = buttons(
      ctl,
      "表の閾値",
      [0.02, 0.05, 0.1, 0.2].map((t) => ({
        label: pct(t, 0),
        onClick: () => {
          state.t = t;
          tSlider.set(TH.indexOf(t));
          draw();
        },
      }))
    );
    pre.select(1);

    const svg = newSvg(root, 660, 330, "予測確率の分布と ROC 曲線");
    const LX0 = Math.log(0.005);
    const LX1 = Math.log(0.4);
    const A = panel(svg, {
      left: 44,
      top: 24,
      width: 320,
      height: 240,
      x: [LX0, LX1],
      y: [0, 1],
      xticks: [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4].map(Math.log),
      xfmt: (t) => {
        const v = 100 * Math.exp(t);
        return (v < 1 ? v.toFixed(1) : Math.round(v)) + "%";
      },
      xlab: "予測確率（対数目盛）",
      ylab: "人の割合（それぞれの群で）",
      ylabOffset: 18,
    });
    const B = panel(svg, {
      left: 430,
      top: 24,
      width: 210,
      height: 210,
      x: [0, 1],
      y: [0, 1],
      xticks: [0, 0.2, 0.4, 0.6, 0.8, 1],
      yticks: [0, 0.2, 0.4, 0.6, 0.8, 1],
      xfmt: (t) => t.toFixed(1),
      yfmt: (t) => t.toFixed(1),
      xlab: "1 − 特異度",
      ylab: "感度",
      ylabOffset: 32,
    });
    // Densities of predicted risk on a log scale, for each group.
    const BINS = 60;
    const w = (LX1 - LX0) / BINS;
    const dens = (arr) => {
      const h = new Array(BINS).fill(0);
      for (const v of arr) {
        const j = Math.floor((Math.log(v) - LX0) / w);
        if (j >= 0 && j < BINS) h[j]++;
      }
      return h.map((c) => c / arr.length);
    };
    const hPos = dens(pos);
    const hNeg = dens(neg);
    const top = 1.1 * Math.max(...hPos, ...hNeg);
    const scale = (v) => v / top;
    const stepPts = (h) => {
      const pts = [[LX0, 0]];
      h.forEach((v, j) => {
        pts.push([LX0 + j * w, scale(v)]);
        pts.push([LX0 + (j + 1) * w, scale(v)]);
      });
      pts.push([LX1, 0]);
      return pts;
    };
    const fixedA = svgEl("g", {}, A.layer);
    A.path(stepPts(hNeg), { stroke: COLOR.blue, "stroke-width": 1.8, fill: COLOR.blue, "fill-opacity": 0.12 }, fixedA);
    A.path(stepPts(hPos), { stroke: COLOR.red, "stroke-width": 1.8, fill: COLOR.red, "fill-opacity": 0.12 }, fixedA);
    // ROC curve over all thresholds.
    const roc = [[1, 1]];
    for (let i = 0; i <= 400; i++) {
      const t = Math.exp(Math.log(0.001) + ((Math.log(0.9) - Math.log(0.001)) * i) / 400);
      roc.push([atLeast(neg, t) / neg.length, atLeast(pos, t) / pos.length]);
    }
    roc.push([0, 0]);
    B.path([[0, 0], [1, 1]], { stroke: COLOR.gray, "stroke-width": 1, "stroke-dasharray": "4 3" });
    B.path(roc, { stroke: COLOR.ink, "stroke-width": 2 });
    const dynA = svgEl("g", {}, A.layer);
    legend(svg, 50, 312, [{ label: `出血した人（${pos.length.toLocaleString("en-US")} 人）`, attrs: { stroke: COLOR.red, "stroke-width": 2 } }]);
    legend(svg, 230, 312, [{ label: `出血しなかった人（${neg.length.toLocaleString("en-US")} 人）`, attrs: { stroke: COLOR.blue, "stroke-width": 2 } }]);
    svgEl("text", { x: B.sx(0.97), y: B.sy(0.06), "text-anchor": "end", class: "sx-note", text: `C 統計量 ${cstat.toFixed(2)}` }, B.over);
    const dynB = svgEl("g", {}, B.over);
    const table = readout(root);
    const num = (v) => v.toLocaleString("en-US");

    function draw() {
      const t = state.t;
      const tp = atLeast(pos, t);
      const fp = atLeast(neg, t);
      const fn = pos.length - tp;
      const tn = neg.length - fp;
      const sens = tp / pos.length;
      const spec = tn / neg.length;
      dynA.textContent = "";
      A.over.textContent = "";
      svgEl("rect", { x: A.sx(Math.log(t)), y: A.sy(1), width: A.sx(LX1) - A.sx(Math.log(t)), height: A.sy(0) - A.sy(1), fill: COLOR.ink, "fill-opacity": 0.06 }, dynA);
      svgEl("line", { x1: A.sx(Math.log(t)), x2: A.sx(Math.log(t)), y1: A.sy(0), y2: A.sy(1), stroke: COLOR.ink, "stroke-width": 1.5 }, A.over);
      svgEl("text", { x: A.sx(Math.log(t)) + 5, y: A.sy(1) + 14, class: "sx-note", text: "陽性 →" }, A.over);
      dynB.textContent = "";
      svgEl("circle", { cx: B.sx(1 - spec), cy: B.sy(sens), r: 6, fill: COLOR.orange, stroke: "#fff", "stroke-width": 1.5 }, dynB);
      const ppv = tp + fp ? tp / (tp + fp) : NaN;
      const npv = tn + fn ? tn / (tn + fn) : NaN;
      table([
        ["閾値", `${pct(t, 1)}（判定が陽性の人 ${pct((tp + fp) / n, (tp + fp) / n < 0.01 ? 1 : 0)}）`],
        ["判定が陽性", `出血した ${num(tp)} 人（真陽性）、出血しなかった ${num(fp)} 人（偽陽性）`],
        ["判定が陰性", `出血した ${num(fn)} 人（偽陰性）、出血しなかった ${num(tn)} 人（真陰性）`],
        ["感度、特異度", `${pct(sens, 0)}、${pct(spec, spec > 0.99 ? 1 : 0)}`],
        ["陽性的中率、陰性的中率", `${pct(ppv, 1)}、${pct(npv, 1)}`],
        ["C 統計量（閾値によらない）", cstat.toFixed(2)],
      ]);
    }
    draw();
  }

  // ---------- 9.2 calibration intercept and slope ----------

  async function calibrationShiftWidget(root) {
    const { lp, y, yOther, metrics } = await newPatients();
    const n = y.length;
    const scenes = [
      { name: "同じ集団", y, row: metrics.find((r) => /^Seven predictors, same/.test(r.model)) },
      { name: "リスクの高い病院", y: yOther, row: metrics.find((r) => /higher risk/.test(r.model)) },
    ];
    // Deciles of the predicted risk (the order does not change with a and b > 0).
    const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => lp[i] - lp[j]);
    const group = new Uint8Array(n);
    order.forEach((i, r) => (group[i] = Math.floor((10 * r) / n)));

    const state = { scene: 0, a: 0, b: 1 };
    const ctl = controls(root);
    buttons(ctl, "場面", scenes.map((s, k) => ({ label: s.name, onClick: () => ((state.scene = k), draw()) }))).select(0);
    const aSlider = slider(ctl, {
      label: "対数オッズに足す値",
      min: -1.5,
      max: 1.5,
      step: 0.01,
      value: state.a,
      format: (v) => signed(v, 2),
      onInput: (v) => {
        state.a = v;
        fix.select(-1);
        draw();
      },
    });
    const bSlider = slider(ctl, {
      label: "対数オッズに掛ける値",
      min: 0.3,
      max: 2,
      step: 0.01,
      value: state.b,
      format: (v) => v.toFixed(2),
      onInput: (v) => {
        state.b = v;
        fix.select(-1);
        draw();
      },
    });
    const fix = buttons(ctl, "予測を", [
      { label: "そのまま", onClick: () => setAB(0, 1) },
      { label: "切片だけ直す", onClick: () => setAB(Math.round(100 * citl(scenes[state.scene].y, lp, 1, 0)) / 100, 1) },
    ]);
    fix.select(0);
    function setAB(a, b) {
      state.a = a;
      state.b = b;
      aSlider.set(a);
      bSlider.set(b);
      draw();
    }

    const svg = newSvg(root, 640, 330, "較正の図");
    const LIM = 0.25;
    const P = panel(svg, {
      left: 60,
      top: 20,
      width: 260,
      height: 260,
      x: [0, LIM],
      y: [0, LIM],
      xticks: [0, 0.05, 0.1, 0.15, 0.2, 0.25],
      yticks: [0, 0.05, 0.1, 0.15, 0.2, 0.25],
      xfmt: (t) => Math.round(100 * t) + "%",
      yfmt: (t) => Math.round(100 * t) + "%",
      xlab: "予測確率（10 等分した各グループの平均）",
      ylab: "実際の出血割合",
      ylabOffset: 42,
    });
    P.path([[0, 0], [LIM, LIM]], { stroke: COLOR.gray, "stroke-width": 1.2, "stroke-dasharray": "5 4" }, svgEl("g", {}, P.layer));
    const dyn = svgEl("g", {}, P.layer);
    legend(svg, 350, 40, [
      { label: "直した予測", attrs: { stroke: COLOR.blue, "stroke-width": 2.2 } },
      { label: "元の予測", attrs: { stroke: COLOR.gray, "stroke-width": 1.5 } },
      { label: "対角線（理想）", attrs: { stroke: COLOR.gray, "stroke-width": 1.2, "stroke-dasharray": "5 4" } },
    ]);
    const note = svgEl("text", { x: 350, y: 120, class: "sx-note" }, svg);
    const table = readout(root);

    // Calibration intercept: with slope fixed at 1, the shift that makes the mean right.
    function citl(yy, eta, b, a) {
      let k = 0;
      for (let i = 0; i < n; i++) k += yy[i];
      return bisect((c) => {
        let s = 0;
        for (let i = 0; i < n; i++) s += expit1(c + a + b * eta[i]);
        return s - k;
      }, -5, 5);
    }

    function draw() {
      const { a, b } = state;
      const scene = scenes[state.scene];
      const yy = scene.y;
      const eta = new Float64Array(n);
      for (let i = 0; i < n; i++) eta[i] = a + b * lp[i];
      const sumP = new Float64Array(10);
      const sumP0 = new Float64Array(10);
      const sumY = new Float64Array(10);
      const cnt = new Float64Array(10);
      let meanP = 0;
      let obs = 0;
      for (let i = 0; i < n; i++) {
        const g = group[i];
        const pi = expit1(eta[i]);
        sumP[g] += pi;
        sumP0[g] += expit1(lp[i]);
        sumY[g] += yy[i];
        cnt[g]++;
        meanP += pi;
        obs += yy[i];
      }
      meanP /= n;
      obs /= n;
      const pts = [];
      const pts0 = [];
      for (let g = 0; g < 10; g++) {
        pts.push([sumP[g] / cnt[g], sumY[g] / cnt[g]]);
        pts0.push([sumP0[g] / cnt[g], sumY[g] / cnt[g]]);
      }
      dyn.textContent = "";
      P.over.textContent = "";
      P.path(pts0, { stroke: COLOR.gray, "stroke-width": 1.5 }, dyn);
      for (const q of pts0) svgEl("circle", { cx: P.sx(q[0]), cy: P.sy(q[1]), r: 3, fill: COLOR.gray }, dyn);
      P.path(pts, { stroke: COLOR.blue, "stroke-width": 2.2 }, dyn);
      for (const q of pts) svgEl("circle", { cx: P.sx(q[0]), cy: P.sy(q[1]), r: 4.5, fill: COLOR.blue, stroke: "#fff", "stroke-width": 1 }, dyn);
      const off = pts.filter((q) => q[0] > LIM).length;
      note.textContent = off ? `${off} 個の点が図の右にはみ出しています` : "";

      const intercept = citl(yy, eta, 1, 0);
      const slope = calSlope(yy, eta);
      const c = auc(yy, eta);
      table([
        ["場面", `${scene.name}（実際の出血割合 ${pct(obs, 1)}）`],
        ["直した予測", `${b.toFixed(2)} × 元の対数オッズ ${a < 0 ? "−" : "+"} ${Math.abs(a).toFixed(2)}`],
        ["予測確率の平均", `${pct(meanP, 1)}（実際 ${pct(obs, 1)}）`],
        ["較正の切片、傾き", `${signed(intercept, 2)}、${slope.toFixed(2)}`],
        ["C 統計量", `${c.toFixed(2)}（順位が変わらないので、切片と傾きを動かしても同じ）`],
      ]);
    }
    draw();
  }

  register("roc-threshold", rocThresholdWidget);
  register("calibration-shift", calibrationShiftWidget);
})();
