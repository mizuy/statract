// Interactive figures, see docs/javascripts/theory-widgets.js.
(function () {
  "use strict";
  const { COLOR, svgEl, pct, signed, controls, slider, buttons, readout, panel, newSvg, loadCsv, register } = window.SX;

  // ---------- 3.4 stratification and standardization ----------

  async function standardizationWidget(root) {
    const strata = await loadCsv("ch3_strata.csv");
    const groups = await loadCsv("ch2_by_group.csv");
    const g1 = groups.find((r) => r.clip === 1);
    const g0 = groups.find((r) => r.clip === 0);
    const trueAtt = g1.y1 - g1.y0;
    const trueAtc = g0.y1 - g0.y0;
    const N1 = strata.reduce((s, r) => s + r.n_clip, 0);
    const N0 = strata.reduce((s, r) => s + r.n_no_clip, 0);
    const N = N1 + N0;
    const tAll = N1 / N;
    const state = { t: tAll };

    // Target population: a share t of it looks like the clip group, the rest like the no-clip group.
    const weights = (t) => strata.map((r) => (t * r.n_clip) / N1 + ((1 - t) * r.n_no_clip) / N0);
    const name = (t) =>
      Math.abs(t - tAll) < 0.0005 ? "全員（ATE）" : t > 0.9995 ? "クリップ群（ATT）" : t < 0.0005 ? "非クリップ群" : null;

    const ctl = controls(root);
    const pre = buttons(ctl, "対象集団", [
      { label: "全員（ATE）", onClick: () => set(tAll) },
      { label: "クリップ群（ATT）", onClick: () => set(1) },
      { label: "非クリップ群", onClick: () => set(0) },
    ]);
    pre.select(0);
    const tSlider = slider(ctl, {
      label: "クリップ群に似せる割合",
      min: 0,
      max: 1,
      step: 0.001,
      value: state.t,
      format: (v) => pct(v, 0),
      onInput: (v) => {
        state.t = v;
        pre.select(Math.abs(v - tAll) < 0.0005 ? 0 : v > 0.9995 ? 1 : v < 0.0005 ? 2 : -1);
        draw();
      },
    });
    function set(t) {
      state.t = t;
      tSlider.set(t);
      draw();
    }

    const H = 8;
    const rowH = 30;
    const top = 44;
    const svg = newSvg(root, 680, top + H * rowH + 50, "層ごとのリスク差と標準化の重み");
    const head = (x, t, anchor) => svgEl("text", { x, y: top - 12, class: "sx-label", "text-anchor": anchor || "start", text: t }, svg);
    head(4, "病変径");
    head(84, "抗血栓薬");
    head(150, "部位");
    head(200, "重み（対象集団での割合）");
    head(676, "クリップあり/なし", "end");
    const yRow = (i) => top + rowH * i + rowH / 2;
    strata.forEach((r, i) => {
      const y = yRow(i) + 4;
      if (i % 2 === 0)
        svgEl("rect", { x: 0, y: top + rowH * i, width: 680, height: rowH, fill: COLOR.gray, "fill-opacity": 0.07 }, svg);
      svgEl("text", { x: 4, y, text: r.large ? "20 mm 以上" : "20 mm 未満" }, svg);
      svgEl("text", { x: 84, y, text: r.antithrombotic ? "あり" : "なし" }, svg);
      svgEl("text", { x: 150, y, text: r.proximal ? "近位" : "遠位" }, svg);
    });
    const barX = 200;
    const barW = 110;
    const barMax = 0.4;
    const bars = svgEl("g", {}, svg);
    const P = panel(svg, {
      left: 390,
      top,
      width: 270,
      height: H * rowH,
      x: [-0.2, 0.07],
      y: [H, 0],
      xticks: [-0.2, -0.15, -0.1, -0.05, 0, 0.05],
      xfmt: (t) => signed(100 * t, 0),
      xlab: "層の中のリスク差（ポイント）",
    });
    svgEl("line", { x1: P.sx(0), x2: P.sx(0), y1: top, y2: top + H * rowH, class: "sx-guide" }, svg);
    const legendY = top + H * rowH + 46;
    svgEl("line", { x1: 4, x2: 26, y1: legendY - 4, y2: legendY - 4, stroke: COLOR.blue, "stroke-width": 2.5 }, svg);
    svgEl("text", { x: 32, y: legendY, text: "標準化したリスク差" }, svg);
    svgEl("line", { x1: 170, x2: 192, y1: legendY - 4, y2: legendY - 4, stroke: COLOR.red, "stroke-width": 1.8, "stroke-dasharray": "5 4" }, svg);
    svgEl("text", { x: 198, y: legendY, text: "本当の値（この 3000 人）" }, svg);
    const table = readout(root);

    function draw() {
      const t = state.t;
      const w = weights(t);
      const est = strata.reduce((s, r, i) => s + w[i] * r.risk_difference, 0);
      const truth = t * trueAtt + (1 - t) * trueAtc;

      bars.textContent = "";
      P.clear();
      strata.forEach((r, i) => {
        const y = yRow(i);
        const len = (Math.min(w[i], barMax) / barMax) * barW;
        svgEl("rect", { x: barX, y: y - 7, width: len, height: 14, fill: COLOR.blue, "fill-opacity": 0.75 }, bars);
        svgEl("text", { x: barX + len + 4, y: y + 4, class: "sx-note", text: pct(w[i], 1) }, bars);
        svgEl(
          "circle",
          { cx: P.sx(r.risk_difference), cy: y, r: 2 + 16 * Math.sqrt(w[i]), fill: COLOR.ink, "fill-opacity": 0.55, stroke: "#fff", "stroke-width": 0.8 },
          P.layer
        );
        svgEl(
          "text",
          { x: P.sx(0.07) - 4, y: y + 4, "text-anchor": "end", class: "sx-note", text: `${r.n_clip}/${r.n_no_clip} 人` },
          P.over
        );
      });
      P.path([[truth, 0], [truth, H]], { stroke: COLOR.red, "stroke-width": 1.8, "stroke-dasharray": "5 4" });
      P.path([[est, 0], [est, H]], { stroke: COLOR.blue, "stroke-width": 2.5 });

      const big = w.indexOf(Math.max(...w));
      const label = (r) => `${r.large ? "20 mm 以上" : "20 mm 未満"}・抗血栓薬${r.antithrombotic ? "あり" : "なし"}・${r.proximal ? "近位" : "遠位"}`;
      const last = strata.findIndex((r) => r.large && r.antithrombotic && r.proximal);
      const target = name(t) || `クリップ群 ${pct(t, 0)}、非クリップ群 ${pct(1 - t, 0)} の割合で混ぜた集団`;
      table([
        ["対象集団", target],
        ["標準化したリスク差", `${signed(100 * est, 1)} ポイント`],
        ["本当の値（この 3000 人）", `${signed(100 * truth, 1)} ポイント`],
        ["重みのいちばん大きい層", `${label(strata[big])}（${pct(w[big], 1)}）`],
        ["差のいちばん大きい層", `${label(strata[last])}：重み ${pct(w[last], 1)}、差 ${signed(100 * strata[last].risk_difference, 1)} ポイント（クリップなしは ${strata[last].n_no_clip} 人）`],
      ]);
    }
    draw();
  }

  register("standardization", standardizationWidget);
})();
