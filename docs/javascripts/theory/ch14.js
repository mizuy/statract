// Interactive figures, see docs/javascripts/theory-widgets.js.
(function () {
  "use strict";
  const { COLOR, svgEl, pct, range, expit, logit, rng, loadCsv, controls, slider, buttons, readout, panel, newSvg } = window.SX;

  // ---------- 14.3 ICC and MOR ----------

  const Z75 = 0.6744897501960817; // standard normal quantile at 0.75
  const Z975 = 1.959963984540054;
  const LOGISTIC_VAR = (Math.PI * Math.PI) / 3;

  async function iccMorWidget(root) {
    const rows = await loadCsv("ch14_models.csv");
    const glmm = rows.find((r) => String(r.model).indexOf("GLMM") >= 0);
    const sigmaHat = glmm.hospital_sd;
    const state = { sigma: sigmaHat, center: 0.055, pair: null };
    const draw2 = rng(14);

    const ctl = controls(root);
    const sSlider = slider(ctl, {
      label: "施設の効果の標準偏差 σ",
      min: 0,
      max: 1.5,
      step: 0.01,
      value: Math.round(100 * sigmaHat) / 100,
      format: (v) => v.toFixed(2),
      onInput: (v) => {
        state.sigma = v;
        pre.select(-1);
        draw();
      },
    });
    slider(ctl, {
      label: "標準的な施設の出血割合",
      min: 0.01,
      max: 0.3,
      step: 0.005,
      value: state.center,
      format: (v) => pct(v),
      onInput: (v) => {
        state.center = v;
        draw();
      },
    });
    const pre = buttons(ctl, "σ を", [
      { label: `データから推定（${sigmaHat.toFixed(2)}）`, onClick: () => setSigma(sigmaHat) },
      { label: "0（施設差なし）", onClick: () => setSigma(0) },
      { label: "1", onClick: () => setSigma(1) },
    ]);
    pre.select(0);
    buttons(ctl, "施設を", [{ label: "ランダムに二つ選ぶ", onClick: () => pick() }]);
    function setSigma(v) {
      state.sigma = v;
      sSlider.set(Math.round(100 * v) / 100);
      draw();
    }
    function pick() {
      state.pair = [draw2.normal(), draw2.normal()];
      draw();
    }

    const svg = newSvg(root, 680, 330, "施設の出血割合の分布と、二つの施設のオッズ比");
    const plotLayer = svgEl("g", {}, svg);
    const table = readout(root);

    function draw() {
      const { sigma, center, pair } = state;
      const mu = logit(center);
      const icc = (sigma * sigma) / (sigma * sigma + LOGISTIC_VAR);
      const mor = Math.exp(Math.SQRT2 * sigma * Z75);
      const lo = expit(mu - Z975 * sigma);
      const hi = expit(mu + Z975 * sigma);

      // Left: density of hospital bleeding rates, logit-normal around the center.
      const xmax = Math.min(1, Math.max(0.1, Math.ceil(expit(mu + 3 * sigma) / 0.05) * 0.05));
      const xs = [];
      for (let i = 1; i < 500; i++) xs.push((xmax * i) / 500);
      const dens = sigma > 0.005
        ? xs.map((p) => {
            const z = (logit(p) - mu) / sigma;
            return [p, Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI) * p * (1 - p))];
          })
        : [];
      const top = dens.length ? 1.15 * Math.max(...dens.map((d) => d[1])) : 1;
      const step = xmax <= 0.1 ? 0.02 : xmax <= 0.3 ? 0.05 : xmax <= 0.6 ? 0.1 : 0.2;
      const xticks = [];
      for (let t = 0; t <= xmax + 1e-9; t += step) xticks.push(Math.round(t * 1000) / 1000);

      plotLayer.textContent = "";
      const L = panel(plotLayer, {
        left: 30,
        top: 26,
        width: 290,
        height: 240,
        x: [0, xmax],
        y: [0, top],
        xticks,
        xfmt: (t) => Math.round(100 * t) + "%",
        xlab: "施設の出血割合（同じ背景の患者）",
        title: "施設の出血割合の分布",
      });
      if (dens.length) {
        L.path(dens.filter((d) => d[0] >= lo && d[0] <= hi).concat([[hi, 0], [lo, 0]]), { fill: COLOR.blue, "fill-opacity": 0.12, stroke: "none" });
        L.path(dens, { stroke: COLOR.blue, "stroke-width": 2.2 });
      }
      svgEl("line", { x1: L.sx(center), x2: L.sx(center), y1: L.sy(0), y2: L.sy(top), class: "sx-guide" }, L.layer);
      if (dens.length) {
        svgEl("text", { x: L.sx(center) + 4, y: L.sy(top) + 12, class: "sx-note", text: "中心" }, L.over);
      }
      if (pair) {
        pair.forEach((z, k) => {
          const p = expit(mu + sigma * z);
          const col = k ? COLOR.red : COLOR.orange;
          svgEl("line", { x1: L.sx(p), x2: L.sx(p), y1: L.sy(0), y2: L.sy(top * 0.55), stroke: col, "stroke-width": 2 }, L.layer);
          svgEl("text", { x: L.sx(p), y: L.sy(top * 0.55) - 4, "text-anchor": "middle", class: "sx-note", fill: col, text: k ? "B" : "A" }, L.layer);
        });
      }

      // Right: distribution of the odds ratio of the riskier to the safer of two hospitals.
      const rmax = 6;
      const R = panel(plotLayer, {
        left: 390,
        top: 26,
        width: 270,
        height: 240,
        x: [1, rmax],
        y: [0, 1],
        xticks: [1, 2, 3, 4, 5, 6],
        xlab: "高いほうの施設のオッズ ÷ 低いほうの施設のオッズ",
        title: "ランダムな二つの施設のオッズ比",
      });
      if (sigma > 0.005) {
        const s2 = Math.SQRT2 * sigma;
        const rs = [];
        for (let i = 0; i <= 500; i++) rs.push(1 + ((rmax - 1) * i) / 500);
        const f = rs.map((r) => {
          const z = Math.log(r) / s2;
          return [r, (2 * Math.exp(-0.5 * z * z)) / (s2 * Math.sqrt(2 * Math.PI) * r)];
        });
        const fmax = 1.12 * Math.max(...f.map((d) => d[1]));
        const g = f.map((d) => [d[0], d[1] / fmax]);
        const half = g.filter((d) => d[0] <= mor);
        R.path(half.concat([[Math.min(mor, rmax), 0], [1, 0]]), { fill: COLOR.blue, "fill-opacity": 0.12, stroke: "none" });
        R.path(g, { stroke: COLOR.blue, "stroke-width": 2.2 });
      }
      if (mor <= rmax) {
        svgEl("line", { x1: R.sx(mor), x2: R.sx(mor), y1: R.sy(0), y2: R.sy(1), stroke: COLOR.ink, "stroke-width": 1.5, "stroke-dasharray": "4 3" }, R.over);
        svgEl("text", { x: R.sx(mor) + 5, y: R.sy(0.92), class: "sx-note", text: "MOR " + mor.toFixed(2) + "（半分がこれより小さい）" }, R.over);
      }
      let pairOr = null;
      if (pair) {
        pairOr = Math.exp(sigma * Math.abs(pair[0] - pair[1]));
        const x = Math.min(pairOr, rmax);
        svgEl("circle", { cx: R.sx(x), cy: R.sy(0.04), r: 5, fill: COLOR.red, stroke: "#fff", "stroke-width": 1 }, R.over);
      }

      const out = [
        ["施設の効果の標準偏差 σ", sigma.toFixed(2) + (Math.abs(sigma - sigmaHat) < 1e-9 ? "（データから推定した値）" : "")],
        ["ICC = σ² / (σ² + π²/3)", icc.toFixed(3)],
        ["MOR = exp(√2 × σ × 0.674)", mor.toFixed(2)],
        ["施設の 95% が入る出血割合", sigma > 0 ? range(lo, hi) : "全施設が " + pct(center)],
      ];
      if (pair) {
        const pa = expit(mu + sigma * pair[0]);
        const pb = expit(mu + sigma * pair[1]);
        out.push(["選んだ二つの施設", `A ${pct(pa)}、B ${pct(pb)}（オッズ比 ${pairOr.toFixed(2)}）`]);
      }
      table(out);
    }
    draw();
  }

  window.SX.register("icc-mor", iccMorWidget);
})();
