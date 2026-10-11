// Interactive figures, see docs/javascripts/theory-widgets.js.
(function () {
  "use strict";
  const { COLOR, svgEl, signed, controls, slider, buttons, readout, panel, newSvg, legend, register } = window.SX;

  // ---------- 10.1 / 10.2 ridge and LASSO with two coefficients ----------

  function penaltyGeometryWidget(root) {
    // A toy loss: 0.5 (b − bhat)' H (b − bhat), with two correlated predictors.
    const bhat = [1.2, 0.45];
    const H = [
      [1, 0.6],
      [0.6, 1],
    ];
    const c = [H[0][0] * bhat[0] + H[0][1] * bhat[1], H[1][0] * bhat[0] + H[1][1] * bhat[1]];
    const loss = (b) => {
      const d0 = b[0] - bhat[0];
      const d1 = b[1] - bhat[1];
      return 0.5 * (H[0][0] * d0 * d0 + 2 * H[0][1] * d0 * d1 + H[1][1] * d1 * d1);
    };

    function ridge(lam) {
      const a = H[0][0] + lam;
      const d = H[1][1] + lam;
      const off = H[0][1];
      const det = a * d - off * off;
      return [(d * c[0] - off * c[1]) / det, (a * c[1] - off * c[0]) / det];
    }

    const soft = (z, t) => (z > t ? z - t : z < -t ? z + t : 0);
    function lasso(lam) {
      const b = [0, 0];
      for (let it = 0; it < 200; it++) {
        b[0] = soft(c[0] - H[0][1] * b[1], lam) / H[0][0];
        b[1] = soft(c[1] - H[1][0] * b[0], lam) / H[1][1];
      }
      return b;
    }

    // Eigen decomposition of H, for drawing the contours of the loss.
    const tr = H[0][0] + H[1][1];
    const det = H[0][0] * H[1][1] - H[0][1] * H[0][1];
    const ev1 = tr / 2 + Math.sqrt((tr * tr) / 4 - det);
    const ev2 = tr / 2 - Math.sqrt((tr * tr) / 4 - det);
    const v1 = Math.abs(H[0][1]) > 1e-12 ? norm([ev1 - H[1][1], H[0][1]]) : [1, 0];
    const v2 = [-v1[1], v1[0]];
    function norm(v) {
      const s = Math.hypot(v[0], v[1]);
      return [v[0] / s, v[1] / s];
    }
    function ellipse(level) {
      const r = Math.sqrt(2 * level);
      const pts = [];
      for (let i = 0; i <= 120; i++) {
        const t = (2 * Math.PI * i) / 120;
        const a = (r * Math.cos(t)) / Math.sqrt(ev1);
        const b = (r * Math.sin(t)) / Math.sqrt(ev2);
        pts.push([bhat[0] + a * v1[0] + b * v2[0], bhat[1] + a * v1[1] + b * v2[1]]);
      }
      return pts;
    }

    const LMAX = 2;
    const state = { lam: 1 };
    const ctl = controls(root);
    const lamSlider = slider(ctl, {
      label: "罰の強さ λ",
      min: 0,
      max: LMAX,
      step: 0.01,
      value: state.lam,
      format: (v) => v.toFixed(2),
      onInput: (v) => {
        state.lam = v;
        pre.select(-1);
        draw();
      },
    });
    const pre = buttons(ctl, "λ を", [
      { label: "0（最尤法）", onClick: () => set(0) },
      { label: "0.5", onClick: () => set(0.5) },
      { label: "1.0", onClick: () => set(1) },
      { label: "1.6（LASSO はすべて 0）", onClick: () => set(1.6) },
    ]);
    pre.select(2);
    function set(v) {
      state.lam = v;
      lamSlider.set(v);
      draw();
    }

    const svg = newSvg(root, 660, 345, "ridge と LASSO の罰と、当てはまりの等高線");
    const P = panel(svg, {
      left: 48,
      top: 24,
      width: 320,
      height: 240,
      x: [-0.4, 1.6],
      y: [-0.5, 1.0],
      xticks: [-0.4, 0, 0.4, 0.8, 1.2, 1.6],
      yticks: [-0.5, 0, 0.5, 1],
      xfmt: (t) => signed(t, 1),
      yfmt: (t) => signed(t, 1),
      xlab: "β₁",
      ylab: "β₂",
      ylabOffset: 36,
    });
    const Q = panel(svg, {
      left: 440,
      top: 24,
      width: 200,
      height: 240,
      x: [0, LMAX],
      y: [-0.1, 1.3],
      xticks: [0, 0.5, 1, 1.5, 2],
      yticks: [0, 0.4, 0.8, 1.2],
      xfmt: (t) => String(t),
      yfmt: (t) => t.toFixed(1),
      xlab: "罰の強さ λ",
      ylab: "係数",
      ylabOffset: 34,
    });

    // Fixed layers: background contours, axes through 0, and the two paths.
    const bg = svgEl("g", {}, P.layer);
    for (const level of [0.02, 0.08, 0.18, 0.32, 0.5, 0.72, 0.98]) {
      P.path(ellipse(level), { stroke: COLOR.gray, "stroke-width": 0.8, opacity: 0.55 }, bg);
    }
    P.path([[-0.4, 0], [1.6, 0]], { stroke: COLOR.ink, "stroke-width": 0.8 }, bg);
    P.path([[0, -0.5], [0, 1]], { stroke: COLOR.ink, "stroke-width": 0.8 }, bg);
    const grid = [];
    for (let i = 0; i <= 200; i++) grid.push((LMAX * i) / 200);
    const ridgePathFar = [];
    for (let i = 0; i <= 200; i++) ridgePathFar.push(ridge(Math.pow(10, -2 + (4 * i) / 200)));
    P.path(ridgePathFar, { stroke: COLOR.blue, "stroke-width": 1, "stroke-dasharray": "3 3" }, bg);
    P.path(grid.map(lasso), { stroke: COLOR.orange, "stroke-width": 1, "stroke-dasharray": "3 3" }, bg);
    svgEl("circle", { cx: P.sx(bhat[0]), cy: P.sy(bhat[1]), r: 4, fill: COLOR.ink }, bg);
    svgEl("text", { x: P.sx(bhat[0]) + 7, y: P.sy(bhat[1]) - 6, class: "sx-note", text: "最尤推定" }, bg);

    const pathAttrs = [
      { stroke: COLOR.blue, "stroke-width": 2 },
      { stroke: COLOR.blue, "stroke-width": 2, "stroke-dasharray": "5 3" },
      { stroke: COLOR.orange, "stroke-width": 2 },
      { stroke: COLOR.orange, "stroke-width": 2, "stroke-dasharray": "5 3" },
    ];
    Q.path([[0, 0], [LMAX, 0]], { stroke: COLOR.gray, "stroke-width": 0.8 });
    Q.path(grid.map((l) => [l, ridge(l)[0]]), pathAttrs[0]);
    Q.path(grid.map((l) => [l, ridge(l)[1]]), pathAttrs[1]);
    Q.path(grid.map((l) => [l, lasso(l)[0]]), pathAttrs[2]);
    Q.path(grid.map((l) => [l, lasso(l)[1]]), pathAttrs[3]);
    const keepQ = Array.from(Q.layer.childNodes);

    legend(svg, 50, 326, [{ label: "ridge（円の罰）", attrs: { stroke: COLOR.blue, "stroke-width": 2 } }]);
    legend(svg, 200, 326, [{ label: "LASSO（菱形の罰）", attrs: { stroke: COLOR.orange, "stroke-width": 2 } }]);
    legend(svg, 440, 326, [{ label: "β₁（実線）、β₂（破線）", attrs: { stroke: COLOR.ink, "stroke-width": 1.5, "stroke-dasharray": "5 3" } }]);

    const table = readout(root);
    const dyn = svgEl("g", {}, P.layer);

    function draw() {
      const lam = state.lam;
      const r = ridge(lam);
      const l = lasso(lam);
      dyn.textContent = "";
      P.over.textContent = "";
      // Contours of the loss through each solution.
      if (lam > 0) {
        P.path(ellipse(loss(r)), { stroke: COLOR.blue, "stroke-width": 1.2, opacity: 0.8 }, dyn);
        P.path(ellipse(loss(l)), { stroke: COLOR.orange, "stroke-width": 1.2, opacity: 0.8 }, dyn);
      }
      // Penalty regions of the same size as each solution.
      const rad = Math.hypot(r[0], r[1]);
      svgEl(
        "ellipse",
        {
          cx: P.sx(0),
          cy: P.sy(0),
          rx: P.sx(rad) - P.sx(0),
          ry: P.sy(0) - P.sy(rad),
          fill: COLOR.blue,
          "fill-opacity": 0.1,
          stroke: COLOR.blue,
          "stroke-width": 1.8,
        },
        dyn
      );
      const s = Math.abs(l[0]) + Math.abs(l[1]);
      if (s > 1e-9) {
        P.path([[s, 0], [0, s], [-s, 0], [0, -s], [s, 0]], { fill: COLOR.orange, "fill-opacity": 0.12, stroke: COLOR.orange, "stroke-width": 1.8 }, dyn);
      }
      svgEl("circle", { cx: P.sx(r[0]), cy: P.sy(r[1]), r: 5.5, fill: COLOR.blue, stroke: "#fff", "stroke-width": 1.5 }, P.over);
      svgEl("circle", { cx: P.sx(l[0]), cy: P.sy(l[1]), r: 5.5, fill: COLOR.orange, stroke: "#fff", "stroke-width": 1.5 }, P.over);

      Q.layer.textContent = "";
      keepQ.forEach((n) => Q.layer.appendChild(n));
      Q.path([[lam, -0.1], [lam, 1.3]], { stroke: COLOR.red, "stroke-width": 1.5 });
      for (const [v, color] of [[r[0], COLOR.blue], [r[1], COLOR.blue], [l[0], COLOR.orange], [l[1], COLOR.orange]]) {
        svgEl("circle", { cx: Q.sx(lam), cy: Q.sy(v), r: 3.5, fill: color, stroke: "#fff", "stroke-width": 1 }, Q.over);
      }

      const f = (v) => (v === 0 ? "0（ちょうど 0）" : signed(v, 2));
      table([
        ["罰の強さ λ", lam.toFixed(2) + (lam === 0 ? "（最尤法と同じ）" : "")],
        ["ridge", `β₁ = ${signed(r[0], 2)}、β₂ = ${signed(r[1], 2)}`],
        ["LASSO", `β₁ = ${f(l[0])}、β₂ = ${f(l[1])}`],
        ["最尤推定（λ = 0）", `β₁ = ${bhat[0].toFixed(2)}、β₂ = ${bhat[1].toFixed(2)}`],
      ]);
    }
    draw();
  }

  register("penalty-geometry", penaltyGeometryWidget);
})();
