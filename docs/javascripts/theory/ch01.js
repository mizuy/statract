// Interactive figures, see docs/javascripts/theory-widgets.js.
(function () {
  "use strict";
  const { COLOR, svgEl, pct, signed, expit, controls, slider, buttons, readout, panel, newSvg, legend, register } = window.SX;

  // ---------- 1.2 logistic curve ----------

  // bleed ~ size_mm fitted to the 3000 patients (logistic_curve.png).
  const FIT = { b0: -3.6548, b1: 0.033846 };
  // Observed bleeding by 10 mm bins of lesion size, as in logistic_curve.png.
  const BINS = [
    [5, 22, 733],
    [15, 61, 1479],
    [25, 29, 521],
    [35, 13, 165],
    [45, 6, 63],
    [55, 5, 39],
  ];

  const plus = (x, d) => (x > 0 && Number(x.toFixed(d)) !== 0 ? "+" : "") + signed(x, d);

  function logisticWidget(root) {
    const state = { b0: FIT.b0, b1: FIT.b1, x0: 15, step: 10 };
    const ctl = controls(root);
    const b0Slider = slider(ctl, {
      label: "切片 β0（0 mm での対数オッズ）",
      min: -8,
      max: 2,
      step: 0.01,
      value: state.b0,
      format: (v) => signed(v, 2),
      onInput: (v) => {
        state.b0 = v;
        draw();
      },
    });
    const b1Slider = slider(ctl, {
      label: "傾き β1（1 mm あたり）",
      min: -0.1,
      max: 0.2,
      step: 0.001,
      value: state.b1,
      format: (v) => signed(v, 3),
      onInput: (v) => {
        state.b1 = v;
        draw();
      },
    });
    slider(ctl, {
      label: "出発点の病変径",
      min: 0,
      max: 140,
      step: 1,
      value: state.x0,
      format: (v) => v + " mm",
      onInput: (v) => {
        state.x0 = v;
        draw();
      },
    });
    buttons(ctl, "大きくする幅", [
      { label: "1 mm", onClick: () => setStep(1) },
      { label: "10 mm", onClick: () => setStep(10) },
    ]).select(1);
    buttons(ctl, "係数", [
      {
        label: "このデータの推定値に戻す",
        onClick: () => {
          state.b0 = FIT.b0;
          state.b1 = FIT.b1;
          b0Slider.set(FIT.b0);
          b1Slider.set(FIT.b1);
          draw();
        },
      },
    ]);
    function setStep(s) {
      state.step = s;
      draw();
    }

    const svg = newSvg(root, 680, 330, "病変径と出血の対数オッズと確率");
    const common = { top: 24, width: 250, height: 230, x: [0, 150], xticks: [0, 50, 100, 150], xlab: "病変径（mm）" };
    const L = panel(
      svg,
      Object.assign({}, common, {
        left: 62,
        y: [-8, 6],
        yticks: [-8, -6, -4, -2, 0, 2, 4, 6],
        yfmt: (t) => signed(t, 0),
        ylab: "出血の対数オッズ",
        ylabOffset: 36,
        title: "対数オッズ：直線",
      })
    );
    const R = panel(
      svg,
      Object.assign({}, common, {
        left: 400,
        y: [0, 1],
        yticks: [0, 0.25, 0.5, 0.75, 1],
        yfmt: (t) => Math.round(100 * t) + "%",
        ylab: "出血の確率",
        ylabOffset: 44,
        title: "確率：S 字",
      })
    );
    // Range of the data (5–80 mm).
    for (const P of [L, R]) {
      const y0 = P === L ? -8 : 0;
      const y1 = P === L ? 6 : 1;
      svgEl(
        "rect",
        { x: P.sx(5), y: P.sy(y1), width: P.sx(80) - P.sx(5), height: P.sy(y0) - P.sy(y1), fill: COLOR.gray, "fill-opacity": 0.12 },
        svg.insertBefore(svgEl("g", {}), P.layer)
      );
    }
    legend(svg, 400, 312, [{ label: "10 mm ごとの実際の出血割合（灰色の点）", attrs: { stroke: "none" } }]);
    svgEl("circle", { cx: 411, cy: 312, r: 4, fill: COLOR.gray }, svg);
    svgEl("text", { x: 62, y: 316, class: "sx-note", text: "灰色の帯：データのある範囲（5–80 mm）" }, svg);
    const table = readout(root);

    function draw() {
      const { b0, b1, x0, step } = state;
      const x1 = x0 + step;
      const eta = (x) => b0 + b1 * x;
      const xs = [];
      for (let i = 0; i <= 300; i++) xs.push((150 * i) / 300);
      L.clear();
      R.clear();
      L.path(xs.map((x) => [x, eta(x)]), { stroke: COLOR.blue, "stroke-width": 2.5 });
      R.path(xs.map((x) => [x, expit(eta(x))]), { stroke: COLOR.blue, "stroke-width": 2.5 });
      for (const [mid, y, n] of BINS) {
        svgEl("circle", { cx: R.sx(mid), cy: R.sy(y / n), r: Math.sqrt(n) * 0.13 + 1.5, fill: COLOR.gray }, R.layer);
      }
      // The step from x0 to x1 on both scales.
      const mark = (P, f, lo) => {
        for (const [x, c] of [[x0, COLOR.ink], [x1, COLOR.red]]) {
          const y = f(x);
          svgEl("line", { x1: P.sx(x), x2: P.sx(x), y1: P.sy(lo), y2: P.sy(y), class: "sx-guide" }, P.layer);
          svgEl("line", { x1: P.sx(0), x2: P.sx(x), y1: P.sy(y), y2: P.sy(y), class: "sx-guide" }, P.layer);
          svgEl("circle", { cx: P.sx(x), cy: P.sy(y), r: 4.5, fill: c, stroke: "#fff", "stroke-width": 1 }, P.layer);
        }
      };
      mark(L, eta, -8);
      mark(R, (x) => expit(eta(x)), 0);

      const p0 = expit(eta(x0));
      const p1 = expit(eta(x1));
      const odds = (p) => p / (1 - p);
      const fmtOdds = (o) => (o < 0.0001 ? "0.0001 未満" : o < 0.01 ? o.toFixed(4) : o < 10 ? o.toFixed(3) : o < 1000 ? o.toFixed(1) : "1000 超");
      const fmtP = (p) => (p < 0.0005 ? "0.1% 未満" : p > 0.9995 ? "99.9% 超" : pct(p));
      const or1 = Math.exp(b1);
      const orStep = Math.exp(b1 * step);
      table([
        ["モデル", `対数オッズ ＝ ${signed(b0, 2)} ${b1 < 0 ? "−" : "＋"} ${Math.abs(b1).toFixed(3)} × 病変径`],
        ["オッズ比 e^β1（1 mm あたり）", `${or1.toFixed(3)}（10 mm なら ${Math.exp(10 * b1).toFixed(2)} 倍）`],
        [`${x0} mm`, `確率 ${fmtP(p0)}、オッズ ${fmtOdds(odds(p0))}`],
        [`${x1} mm`, `確率 ${fmtP(p1)}、オッズ ${fmtOdds(odds(p1))}`],
        [`${step} mm 大きくすると`, `オッズは ${orStep.toFixed(3)} 倍（どこから始めても同じ）、確率は ${plus(100 * (p1 - p0), 1)} ポイント（始める場所で変わる）`],
      ]);
    }
    draw();
  }

  register("logistic-curve", logisticWidget);
})();
