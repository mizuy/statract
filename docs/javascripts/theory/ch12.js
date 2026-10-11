// Interactive figures, see docs/javascripts/theory-widgets.js.
(function () {
  "use strict";
  const { COLOR, el, svgEl, rng, loadCsv, controls, readout, panel, newSvg, legend, register } = window.SX;

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

  // ---------- 12.3 flipping the coins again ----------

  async function replicatesWidget(root) {
    const [ref] = await loadCsv("ch12_replicates.csv");
    // Chapter 7 estimate in the book's data (odds ratio of antithrombotic, with 95% CI).
    const ch7 = (await loadCsv("ch7_model.csv")).find((r) => r.term === "Antithrombotic");
    const trueOr = Math.exp(0.9);

    // 3000 patients made the same way as the book's data, with their true risks.
    const d = simulate(3000, rng(1438));
    const X = [];
    for (let i = 0; i < d.n; i++) X.push(design7(d, i));
    let expected = 0;
    for (let i = 0; i < d.n; i++) expected += d.p[i];
    const R = rng(12);
    const counts = [];
    const ors = [];
    let busy = false;

    function once() {
      const y = coin(d.p, R);
      let k = 0;
      for (let i = 0; i < y.length; i++) k += y[i];
      counts.push(k);
      ors.push(Math.exp(fitLogistic(X, y)[3]));
    }
    // Many draws in small chunks, so the page stays responsive.
    function run(times) {
      if (busy) return;
      busy = true;
      let left = times;
      const step = () => {
        const m = Math.min(left, 25);
        for (let j = 0; j < m; j++) once();
        left -= m;
        draw();
        if (left > 0) requestAnimationFrame(step);
        else busy = false;
      };
      step();
    }

    const ctl = controls(root);
    const wrap = el("div", { class: "sx-buttons", role: "group", "aria-label": "硬貨を投げ直す" }, ctl);
    el("span", { class: "sx-buttons__label", text: "硬貨を投げ直す" }, wrap);
    for (const [label, fn] of [
      ["1 回", () => run(1)],
      ["100 回", () => run(100)],
      ["1000 回", () => run(1000)],
      ["最初から", () => {
        if (busy) return;
        counts.length = 0;
        ors.length = 0;
        draw();
      }],
    ]) {
      const b = el("button", { type: "button", class: "md-button sx-button", text: label }, wrap);
      b.addEventListener("click", fn);
    }

    const svg = newSvg(root, 660, 330, "硬貨を投げ直したときの出血の人数と、抗血栓薬のオッズ比");
    const C0 = 90;
    const C1 = 180;
    const A = panel(svg, {
      left: 40,
      top: 28,
      width: 270,
      height: 230,
      x: [C0, C1],
      y: [0, 1],
      xticks: [90, 110, 130, 150, 170],
      xlab: "出血した人数（3000 人中）",
      title: "出血の人数",
    });
    const OR0 = Math.log(1);
    const OR1 = Math.log(6);
    const B = panel(svg, {
      left: 370,
      top: 28,
      width: 270,
      height: 230,
      x: [OR0, OR1],
      y: [0, 1],
      xticks: [1, 1.5, 2, 3, 4, 6].map(Math.log),
      xfmt: (t) => String(Math.round(10 * Math.exp(t)) / 10),
      xlab: "抗血栓薬のオッズ比（対数目盛）",
      title: "抗血栓薬のオッズ比の推定",
    });
    legend(svg, 40, 312, [{ label: "本の 3000 人のデータ", attrs: { stroke: COLOR.red, "stroke-width": 2 } }]);
    legend(svg, 230, 312, [{ label: "本当の値（期待値）", attrs: { stroke: COLOR.ink, "stroke-width": 1.5, "stroke-dasharray": "5 4" } }]);
    legend(svg, 420, 312, [{ label: "最新の 1 回", attrs: { stroke: COLOR.blue, "stroke-width": 3 } }]);
    const table = readout(root);

    function hist(P, values, lo, hi, bins, last) {
      const w = (hi - lo) / bins;
      const h = new Array(bins).fill(0);
      for (const v of values) {
        const j = Math.floor((v - lo) / w);
        if (j >= 0 && j < bins) h[j]++;
      }
      const top = Math.max(1, ...h);
      for (let j = 0; j < bins; j++) {
        if (!h[j]) continue;
        const x0 = P.sx(lo + j * w);
        const x1 = P.sx(lo + (j + 1) * w);
        const y = P.sy((0.92 * h[j]) / top);
        svgEl("rect", { x: x0 + 0.5, y, width: Math.max(0.5, x1 - x0 - 1), height: P.sy(0) - y, fill: COLOR.gray, "fill-opacity": 0.55 }, P.layer);
      }
      if (last !== undefined) {
        svgEl("line", { x1: P.sx(last), x2: P.sx(last), y1: P.sy(0), y2: P.sy(0.97), stroke: COLOR.blue, "stroke-width": 3 }, P.layer);
      }
    }
    const vline = (P, x, attrs) => svgEl("line", Object.assign({ x1: P.sx(x), x2: P.sx(x), y1: P.sy(0), y2: P.sy(1) }, attrs), P.layer);
    const quant = (v, q) => {
      const s = v.slice().sort((a, b) => a - b);
      const pos = q * (s.length - 1);
      const i = Math.floor(pos);
      return s[i] + (s[Math.min(i + 1, s.length - 1)] - s[i]) * (pos - i);
    };

    function draw() {
      const n = counts.length;
      A.clear();
      B.clear();
      hist(A, counts, C0, C1, 45, n ? counts[n - 1] : undefined);
      hist(B, ors.map(Math.log), OR0, OR1, 45, n ? Math.log(ors[n - 1]) : undefined);
      vline(A, expected, { stroke: COLOR.ink, "stroke-width": 1.5, "stroke-dasharray": "5 4" });
      vline(A, ref.observed, { stroke: COLOR.red, "stroke-width": 2 });
      vline(B, Math.log(trueOr), { stroke: COLOR.ink, "stroke-width": 1.5, "stroke-dasharray": "5 4" });
      vline(B, Math.log(ch7.exp_estimate), { stroke: COLOR.red, "stroke-width": 2 });
      if (!n) {
        for (const P of [A, B]) {
          svgEl("text", { x: P.sx((P === A ? C0 + C1 : OR0 + OR1) / 2), y: P.sy(0.5), "text-anchor": "middle", class: "sx-note", text: "上のボタンで硬貨を投げ直します" }, P.over);
        }
      }
      const enough = n >= 40;
      const or2 = (v) => v.toFixed(2);
      table([
        ["投げ直した回数", n + " 回"],
        ["最新の 1 回", n ? `出血 ${counts[n - 1]} 人、抗血栓薬のオッズ比 ${or2(ors[n - 1])}` : "まだありません"],
        [
          "出血の人数",
          enough
            ? `平均 ${Math.round(counts.reduce((s, v) => s + v, 0) / n)} 人、95% の範囲 ${Math.round(quant(counts, 0.025))}–${Math.round(quant(counts, 0.975))} 人`
            : "40 回以上で表示します",
        ],
        ["抗血栓薬のオッズ比", enough ? `中央値 ${or2(quant(ors, 0.5))}、95% の範囲 ${or2(quant(ors, 0.025))}–${or2(quant(ors, 0.975))}` : "40 回以上で表示します"],
        ["本（5000 回）の出血の人数", `平均 ${Math.round(ref.expected)} 人、95% の範囲 ${ref.sim_low}–${ref.sim_high} 人`],
        ["本当のオッズ比と第 7 章の推定", `本当は ${or2(trueOr)}。第 7 章では ${or2(ch7.exp_estimate)}（95% 信頼区間 ${or2(ch7.exp_conf_low)}–${or2(ch7.exp_conf_high)}）`],
      ]);
    }
    draw();
  }

  register("replicates", replicatesWidget);
})();
