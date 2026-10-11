// Interactive figures, see docs/javascripts/theory-widgets.js.
(function () {
  "use strict";
  const { COLOR, svgEl, pct, expit, logit, loadCsv, controls, slider, buttons, readout, newSvg, register } = window.SX;

  // ---------- 16.2 Fagan nomogram: pre-test probability x likelihood ratio ----------

  const PROB_TICKS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999];
  const LR_TICKS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200];
  const LOGIT_MAX = logit(0.999);

  function probLabel(p) {
    const v = 100 * p;
    return (v < 1 || v > 99 ? v.toFixed(1) : v.toFixed(0)) + "%";
  }

  function lrLabel(v) {
    return v < 1 ? String(Number(v.toPrecision(2))) : v.toFixed(v < 10 && v % 1 ? 1 : 0);
  }

  function oddsLabel(o) {
    return o < 0.01 ? o.toPrecision(2) : o < 0.1 ? o.toFixed(3) : o < 10 ? o.toFixed(2) : o.toFixed(1);
  }

  async function faganWidget(root) {
    const rows = await loadCsv("ch16_table.csv");
    const bands = await loadCsv("ch16_bands.csv");
    const here = rows.find((r) => String(r.setting).indexOf("50,000") >= 0);
    const other = rows.find((r) => String(r.setting).indexOf("higher risk") >= 0);
    const state = { pre: here.prevalence, lr: here.lr_pos };

    // Slider positions: 0-4000 on the logit scale (pre-test) and on the log scale (LR).
    const preToPos = (p) => Math.round((4000 * (logit(p) + LOGIT_MAX)) / (2 * LOGIT_MAX));
    const posToPre = (i) => expit(-LOGIT_MAX + (2 * LOGIT_MAX * i) / 4000);
    const lrToPos = (v) => Math.round(1000 * (Math.log10(v) + 2));
    const posToLr = (i) => Math.pow(10, i / 1000 - 2);

    const ctl = controls(root);
    const preSlider = slider(ctl, {
      label: "検査前確率",
      min: 0,
      max: 4000,
      step: 1,
      value: preToPos(state.pre),
      toValue: posToPre,
      format: (v) => pct(v, v < 0.01 ? 2 : 1),
      onInput: (v) => {
        state.pre = v;
        prePick.select(-1);
        draw();
      },
    });
    const lrSlider = slider(ctl, {
      label: "尤度比",
      min: 0,
      max: 4000,
      step: 1,
      value: lrToPos(state.lr),
      toValue: posToLr,
      format: (v) => v.toFixed(v < 1 ? 3 : 2),
      onInput: (v) => {
        state.lr = v;
        lrPick.select(-1);
        draw();
      },
    });
    const setPre = (p) => {
      state.pre = p;
      preSlider.set(preToPos(p));
      draw();
    };
    const setLr = (v) => {
      state.lr = v;
      lrSlider.set(lrToPos(v));
      draw();
    };
    const prePick = buttons(ctl, "検査前確率を", [
      { label: `この集団（${pct(here.prevalence)}）`, onClick: () => setPre(here.prevalence) },
      { label: `リスクの高い病院（${pct(other.prevalence)}）`, onClick: () => setPre(other.prevalence) },
      { label: "30%", onClick: () => setPre(0.3) },
    ]);
    const lowBand = bands[0];
    const highBand = bands[bands.length - 1];
    const lrPick = buttons(ctl, "尤度比を", [
      { label: `陽性 LR+ ${here.lr_pos.toFixed(2)}`, onClick: () => setLr(here.lr_pos) },
      { label: `陰性 LR− ${here.lr_neg.toFixed(2)}`, onClick: () => setLr(here.lr_neg) },
      { label: `予測 ${pct(lowBand.upper, 0)} 未満 ${lowBand.lr.toFixed(2)}`, onClick: () => setLr(lowBand.lr) },
      { label: `予測 ${pct(highBand.lower, 0)} 以上 ${highBand.lr.toFixed(2)}`, onClick: () => setLr(highBand.lr) },
      { label: "10", onClick: () => setLr(10) },
      { label: "0.1", onClick: () => setLr(0.1) },
    ]);
    prePick.select(0);
    lrPick.select(0);

    // Three parallel axes. A straight line through the left and middle values
    // meets the right axis at the post-test probability, because
    // logit(post) = logit(pre) + log(LR).
    const W = 680;
    const H = 460;
    const TOP = 48;
    const K = 28; // pixels per unit of log odds
    const Y0 = TOP + LOGIT_MAX * K;
    const XL = 170;
    const XM = 340;
    const XR = 510;
    const yLeft = (p) => Y0 + logit(p) * K; // pre-test: 0.1% at the top
    const yRight = (p) => Y0 - logit(p) * K; // post-test: 99.9% at the top
    const yMid = (v) => Y0 - (Math.log(v) * K) / 2;

    const svg = newSvg(root, W, H, "Fagan のノモグラム：検査前確率と尤度比から検査後確率を読む");
    const axes = svgEl("g", { class: "sx-axes" }, svg);
    const axis = (x, ticks, y, fmt, side) => {
      svgEl("line", { x1: x, x2: x, y1: TOP, y2: TOP + 2 * LOGIT_MAX * K, class: "sx-frame" }, axes);
      ticks.forEach((t, i) => {
        const yy = y(t);
        if (yy < TOP - 0.5 || yy > TOP + 2 * LOGIT_MAX * K + 0.5) return;
        // The middle axis is half as steep, so its labels alternate sides.
        const left = side === "left" || (side === "both" && i % 2 === 1);
        const dx = left ? -1 : 1;
        svgEl("line", { x1: x, x2: x + 5 * dx, y1: yy, y2: yy, class: "sx-tick" }, axes);
        svgEl("text", { x: x + 8 * dx, y: yy + 4, "text-anchor": left ? "end" : "start", text: fmt(t) }, axes);
      });
    };
    axis(XL, PROB_TICKS, yLeft, probLabel, "left");
    axis(XM, LR_TICKS, yMid, lrLabel, "both");
    axis(XR, PROB_TICKS, yRight, probLabel, "right");
    for (const [x, t] of [
      [XL, "検査前確率"],
      [XM, "尤度比"],
      [XR, "検査後確率"],
    ])
      svgEl("text", { x, y: TOP - 18, "text-anchor": "middle", class: "sx-title", text: t }, axes);

    const layer = svgEl("g", {}, svg);
    const note = svgEl("text", { x: XR + 60, y: TOP + 20, class: "sx-title" }, svg);
    const note2 = svgEl("text", { x: XR + 60, y: TOP + 40, class: "sx-note" }, svg);
    const table = readout(root);

    function draw() {
      const { pre, lr } = state;
      const preOdds = pre / (1 - pre);
      const postOdds = preOdds * lr;
      const post = postOdds / (1 + postOdds);
      layer.textContent = "";
      const y1 = yLeft(Math.min(Math.max(pre, 0.001), 0.999));
      const y3 = yRight(Math.min(Math.max(post, 0.001), 0.999));
      svgEl("line", { x1: XL, x2: XR, y1, y2: y3, stroke: COLOR.red, "stroke-width": 2 }, layer);
      svgEl("circle", { cx: XL, cy: y1, r: 4.5, fill: COLOR.blue }, layer);
      svgEl("circle", { cx: XM, cy: yMid(lr), r: 4.5, fill: COLOR.ink }, layer);
      svgEl("circle", { cx: XR, cy: y3, r: 4.5, fill: COLOR.red }, layer);
      note.textContent = `検査後 ${pct(post, post < 0.01 ? 2 : 1)}`;
      note2.textContent = `検査前 ${pct(pre, pre < 0.01 ? 2 : 1)} から`;
      table([
        ["検査前オッズ", `${pct(pre, pre < 0.01 ? 2 : 1)} ÷ ${pct(1 - pre, 1)} = ${oddsLabel(preOdds)}`],
        ["検査後オッズ", `${oddsLabel(preOdds)} × 尤度比 ${lr.toFixed(lr < 1 ? 3 : 2)} = ${oddsLabel(postOdds)}`],
        ["検査後確率", `${oddsLabel(postOdds)} ÷ (1 + ${oddsLabel(postOdds)}) = ${pct(post, post < 0.01 ? 2 : 1)}`],
      ]);
    }
    draw();
  }

  register("fagan", faganWidget);
})();
