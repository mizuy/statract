// Interactive figures, see docs/javascripts/theory-widgets.js.
// 7.2 a risk calculator from the seven-predictor model, with the log-odds terms.
(function () {
  "use strict";
  const { COLOR, svgEl, pct, signed, expit, logit, loadCsv, controls, slider, buttons, readout, panel, newSvg, register } = window.SX;

  const TERMS = [
    { key: "age", term: "Age (per year)", label: "年齢", show: (v) => v + " 歳" },
    { key: "male", term: "Male", label: "性別", show: (v) => (v ? "男性" : "女性") },
    { key: "antithrombotic", term: "Antithrombotic", label: "抗血栓薬", show: (v) => (v ? "あり" : "なし") },
    { key: "hypertension", term: "Hypertension", label: "高血圧", show: (v) => (v ? "あり" : "なし") },
    { key: "size_mm", term: "Lesion size (per mm)", label: "病変径", show: (v) => v + " mm" },
    { key: "proximal", term: "Proximal colon", label: "部位", show: (v) => (v ? "近位" : "遠位") },
    { key: "clip", term: "Prophylactic clip", label: "予防的クリップ", show: (v) => (v ? "あり" : "なし") },
  ];

  async function riskWidget(root) {
    const [model, patients] = await Promise.all([loadCsv("ch7_model.csv"), loadCsv("ch7_patients.csv")]);
    for (const t of TERMS) t.beta = Math.log(model.find((r) => r.term === t.term).exp_estimate);
    // ch7_model.csv has the odds ratios only. Each example patient's predicted
    // risk gives the intercept back: logit(risk) minus the other terms. The
    // three agree to rounding error; their mean is used.
    const sum = (x) => TERMS.reduce((s, t) => s + t.beta * x[t.key], 0);
    const intercepts = patients.map((p) => logit(p.risk_pct / 100) - sum(p));
    const b0 = intercepts.reduce((s, v) => s + v, 0) / intercepts.length;

    const state = Object.fromEntries(TERMS.map((t) => [t.key, patients[0][t.key]]));
    const ctl = controls(root);
    const pre = buttons(
      ctl,
      "例の患者",
      patients.map((p, i) => ({ label: `${p.patient} さん`, onClick: () => setPatient(i) }))
    );
    pre.select(0);
    const touched = () => {
      pre.select(-1);
      draw();
    };
    const ageS = slider(ctl, {
      label: "年齢",
      min: 30,
      max: 95,
      step: 1,
      value: state.age,
      format: (v) => v + " 歳",
      onInput: (v) => {
        state.age = v;
        touched();
      },
    });
    const sizeS = slider(ctl, {
      label: "病変径",
      min: 5,
      max: 80,
      step: 1,
      value: state.size_mm,
      format: (v) => v + " mm",
      onInput: (v) => {
        state.size_mm = v;
        touched();
      },
    });
    const toggles = {};
    for (const [key, label, on, off] of [
      ["male", "性別", "男性", "女性"],
      ["antithrombotic", "抗血栓薬", "あり", "なし"],
      ["hypertension", "高血圧", "あり", "なし"],
      ["proximal", "部位", "近位", "遠位"],
      ["clip", "予防的クリップ", "あり", "なし"],
    ]) {
      const set = (v) => {
        state[key] = v;
        touched();
      };
      toggles[key] = buttons(ctl, label, [
        { label: on, onClick: () => set(1) },
        { label: off, onClick: () => set(0) },
      ]);
    }
    const syncToggles = () => {
      for (const k of Object.keys(toggles)) toggles[k].select(state[k] ? 0 : 1);
    };
    syncToggles();
    function setPatient(i) {
      for (const t of TERMS) state[t.key] = patients[i][t.key];
      ageS.set(state.age);
      sizeS.set(state.size_mm);
      syncToggles();
      draw();
    }

    // Waterfall of the linear predictor: the intercept, then each term.
    const X0 = -6;
    const X1 = 2;
    const ROW = 25;
    const n = TERMS.length + 2;
    const top = 46;
    const svg = newSvg(root, 660, top + n * ROW + 48, "線形予測子の項ごとの寄与と予測確率");
    const P = panel(svg, {
      left: 150,
      top,
      width: 440,
      height: n * ROW,
      x: [X0, X1],
      y: [n, 0],
      xticks: [-6, -5, -4, -3, -2, -1, 0, 1, 2],
      xfmt: (t) => signed(t, 0),
      xlab: "対数オッズ（線形予測子 η への寄与）",
    });
    // Probability scale along the top edge.
    for (const p of [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8]) {
      const x = P.sx(logit(p));
      svgEl("line", { x1: x, x2: x, y1: top - 4, y2: top, class: "sx-tick" }, svg);
      svgEl("text", { x, y: top - 8, "text-anchor": "middle", text: 100 * p + "%" }, svg);
    }
    svgEl("text", { x: 150, y: 16, class: "sx-title", text: "予測確率の目盛" }, svg);
    const labels = svgEl("g", {}, svg);
    const table = readout(root);

    function draw() {
      P.clear();
      labels.textContent = "";
      const rows = [{ label: "切片", value: b0, text: "", kind: "base" }].concat(
        TERMS.map((t) => ({ label: t.label, text: t.show(state[t.key]), value: t.beta * state[t.key] }))
      );
      let at = 0;
      rows.forEach((r, i) => {
        const yc = P.sy(i + 0.5);
        const from = at;
        at += r.value;
        const x0 = P.sx(Math.min(from, at));
        const x1 = P.sx(Math.max(from, at));
        const color = r.kind === "base" ? COLOR.gray : r.value > 0 ? COLOR.red : COLOR.blue;
        if (Math.abs(r.value) > 1e-12)
          svgEl("rect", { x: x0, y: yc - 8, width: Math.max(x1 - x0, 1), height: 16, fill: color }, P.layer);
        else svgEl("line", { x1: x0, x2: x0, y1: yc - 8, y2: yc + 8, stroke: COLOR.gray, "stroke-width": 1 }, P.layer);
        if (i < rows.length - 1)
          svgEl("line", { x1: P.sx(at), x2: P.sx(at), y1: yc + 8, y2: yc + ROW - 8, class: "sx-guide" }, P.layer);
        svgEl("text", { x: 144, y: yc + 4, "text-anchor": "end", class: "sx-label", text: r.text ? `${r.label}（${r.text}）` : r.label }, labels);
        svgEl("text", { x: 646, y: yc + 4, "text-anchor": "end", text: (r.value > 0 ? "+" : "") + signed(r.value, 2) }, labels);
      });
      const eta = at;
      const yc = P.sy(n - 0.5);
      svgEl("line", { x1: P.sx(0), x2: P.sx(0), y1: P.sy(0), y2: P.sy(n), class: "sx-guide" }, P.layer);
      svgEl("line", { x1: P.sx(Math.max(X0, Math.min(X1, eta))), x2: P.sx(Math.max(X0, Math.min(X1, eta))), y1: P.sy(0), y2: P.sy(n), stroke: COLOR.ink, "stroke-width": 1.2 }, P.over);
      svgEl("circle", { cx: P.sx(eta), cy: yc, r: 6, fill: COLOR.ink }, P.layer);
      svgEl("text", { x: 144, y: yc + 4, "text-anchor": "end", class: "sx-label", text: "合計 η" }, labels);
      svgEl("text", { x: 646, y: yc + 4, "text-anchor": "end", class: "sx-label", text: signed(eta, 2) }, labels);

      const p = expit(eta);
      const match = patients.find((q) => TERMS.every((t) => q[t.key] === state[t.key]));
      table([
        ["線形予測子 η", `切片 ${signed(b0, 2)} ＋ 項の和 ${signed(eta - b0, 2)} ＝ ${signed(eta, 2)}`],
        ["予測確率", "1 / (1 + e^−η) = " + pct(p) + (match ? `（表の ${match.patient} さんと同じ）` : "")],
        ["オッズ e^η", `${p < 0.5 ? "1 対 " + ((1 - p) / p).toFixed(1) : ((p / (1 - p)).toFixed(1) + " 対 1")}`],
      ]);
    }
    draw();
  }

  register("risk-calculator", riskWidget);
})();
