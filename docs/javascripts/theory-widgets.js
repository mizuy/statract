// Interactive figures for the theory text (docs/theory).
//
// Markup: <div class="sx-widget" data-widget="likelihood"></div>
// Data come from the CSV files that tools/sync_example_assets.py copies into
// docs/examples/assets/theory_bleeding/, so the widgets show the same numbers
// as the static figures and tables.
(function () {
  "use strict";

  const SVGNS = "http://www.w3.org/2000/svg";
  const COLOR = { blue: "#0073C2", red: "#CD534C", gray: "#868686", orange: "#E8871E", ink: "#333333" };
  const ASSETS = "examples/assets/theory_bleeding/";

  // ---------- small helpers ----------

  function el(tag, attrs, parent) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "text") node.textContent = v;
      else node.setAttribute(k, v);
    }
    if (parent) parent.appendChild(node);
    return node;
  }

  function svgEl(tag, attrs, parent) {
    const node = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "text") node.textContent = v;
      else node.setAttribute(k, v);
    }
    if (parent) parent.appendChild(node);
    return node;
  }

  function pct(x, digits) {
    if (!isFinite(x)) return "—";
    return (100 * x).toFixed(digits === undefined ? 1 : digits) + "%";
  }

  function range(lo, hi, digits) {
    const d = digits === undefined ? 1 : digits;
    return (100 * lo).toFixed(d) + "–" + (100 * hi).toFixed(d) + "%";
  }

  // Minus sign as in the text (U+2212).
  function signed(x, digits) {
    const s = Math.abs(x).toFixed(digits);
    return x < 0 && Number(s) !== 0 ? "−" + s : s;
  }

  function siteBase() {
    const config = document.getElementById("__config");
    let base = ".";
    if (config) {
      try {
        base = JSON.parse(config.textContent).base || ".";
      } catch (e) {
        base = ".";
      }
    }
    return new URL(base.replace(/\/?$/, "/"), window.location.href);
  }

  async function loadCsv(name) {
    const url = new URL(ASSETS + name, siteBase());
    const response = await fetch(url);
    if (!response.ok) throw new Error("cannot load " + url);
    const lines = (await response.text()).trim().split(/\r?\n/);
    const header = lines[0].split(",");
    return lines.slice(1).map((line) => {
      const cells = line.split(",");
      const row = {};
      header.forEach((h, i) => {
        const v = cells[i];
        row[h] = v !== "" && !isNaN(Number(v)) ? Number(v) : v;
      });
      return row;
    });
  }

  function logGamma(x) {
    // Lanczos approximation (g = 7, n = 9).
    const c = [
      0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
      -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
      1.5056327351493116e-7,
    ];
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
    x -= 1;
    let a = c[0];
    const t = x + 7.5;
    for (let i = 1; i < 9; i++) a += c[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }

  function betaLogPdf(x, a, b) {
    if (x <= 0 || x >= 1) return -Infinity;
    return (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - (logGamma(a) + logGamma(b) - logGamma(a + b));
  }

  // Quantiles of Beta(a, b) by integrating the density on a fine grid.
  function betaQuantiles(a, b, probs) {
    const m = 40000;
    const xs = new Float64Array(m + 1);
    const cdf = new Float64Array(m + 1);
    let prev = 0;
    for (let i = 0; i <= m; i++) {
      xs[i] = i / m;
      const f = Math.exp(betaLogPdf(xs[i], a, b));
      const fi = isFinite(f) ? f : 0;
      if (i > 0) cdf[i] = cdf[i - 1] + (prev + fi) / (2 * m);
      prev = fi;
    }
    const total = cdf[m];
    return probs.map((p) => {
      const target = p * total;
      let i = 1;
      while (i < m && cdf[i] < target) i++;
      const f = (target - cdf[i - 1]) / Math.max(cdf[i] - cdf[i - 1], 1e-300);
      return xs[i - 1] + f / m;
    });
  }

  function expit(x) {
    return 1 / (1 + Math.exp(-x));
  }

  function logit(p) {
    return Math.log(p / (1 - p));
  }

  function bisect(f, lo, hi) {
    let flo = f(lo);
    for (let i = 0; i < 80; i++) {
      const mid = (lo + hi) / 2;
      const fm = f(mid);
      if ((fm < 0) === (flo < 0)) {
        lo = mid;
        flo = fm;
      } else {
        hi = mid;
      }
    }
    return (lo + hi) / 2;
  }

  function golden(f, lo, hi) {
    const r = (Math.sqrt(5) - 1) / 2;
    let a = lo;
    let b = hi;
    let c = b - r * (b - a);
    let d = a + r * (b - a);
    let fc = f(c);
    let fd = f(d);
    for (let i = 0; i < 60; i++) {
      if (fc > fd) {
        b = d;
        d = c;
        fd = fc;
        c = b - r * (b - a);
        fc = f(c);
      } else {
        a = c;
        c = d;
        fc = fd;
        d = a + r * (b - a);
        fd = f(d);
      }
    }
    return (a + b) / 2;
  }

  // ---------- controls ----------

  function controls(root) {
    return el("div", { class: "sx-controls" }, root);
  }

  function slider(parent, o) {
    const wrap = el("label", { class: "sx-slider" }, parent);
    el("span", { class: "sx-slider__label", text: o.label }, wrap);
    const input = el(
      "input",
      { type: "range", min: o.min, max: o.max, step: o.step, value: o.value, "aria-label": o.label },
      wrap
    );
    const out = el("output", { class: "sx-slider__value" }, wrap);
    const read = () => (o.toValue ? o.toValue(Number(input.value)) : Number(input.value));
    const show = () => {
      out.textContent = o.format(read());
    };
    input.addEventListener("input", () => {
      show();
      o.onInput(read());
    });
    show();
    return {
      input,
      get value() {
        return read();
      },
      set(position) {
        input.value = position;
        show();
      },
      setMax(max) {
        input.max = max;
        show();
      },
    };
  }

  function buttons(parent, label, items) {
    const wrap = el("div", { class: "sx-buttons", role: "group", "aria-label": label }, parent);
    el("span", { class: "sx-buttons__label", text: label }, wrap);
    const nodes = items.map((item) => {
      const b = el("button", { type: "button", class: "md-button sx-button", text: item.label }, wrap);
      b.addEventListener("click", () => {
        nodes.forEach((n) => n.classList.remove("sx-button--on"));
        b.classList.add("sx-button--on");
        item.onClick();
      });
      return b;
    });
    return {
      select(i) {
        nodes.forEach((n, j) => n.classList.toggle("sx-button--on", i === j));
      },
    };
  }

  function readout(root) {
    const table = el("table", { class: "sx-readout" }, root);
    const body = el("tbody", {}, table);
    return function render(rows) {
      body.textContent = "";
      for (const [k, v] of rows) {
        const tr = el("tr", {}, body);
        el("th", { scope: "row", text: k }, tr);
        el("td", { text: v }, tr);
      }
    };
  }

  // ---------- plotting ----------

  // A panel inside an SVG, with axes, ticks and a clipped data layer.
  function panel(svg, o) {
    const left = o.left;
    const top = o.top;
    const w = o.width;
    const h = o.height;
    const sx = (x) => left + ((x - o.x[0]) / (o.x[1] - o.x[0])) * w;
    const sy = (y) => top + h - ((y - o.y[0]) / (o.y[1] - o.y[0])) * h;
    const id = "sx-clip-" + Math.random().toString(36).slice(2);
    const defs = svgEl("defs", {}, svg);
    const clip = svgEl("clipPath", { id }, defs);
    svgEl("rect", { x: left, y: top, width: w, height: h }, clip);

    const axes = svgEl("g", { class: "sx-axes" }, svg);
    svgEl("rect", { x: left, y: top, width: w, height: h, class: "sx-frame" }, axes);
    for (const t of o.xticks || []) {
      const x = sx(t);
      svgEl("line", { x1: x, x2: x, y1: top + h, y2: top + h + 4, class: "sx-tick" }, axes);
      svgEl("text", { x, y: top + h + 16, "text-anchor": "middle", text: o.xfmt ? o.xfmt(t) : t }, axes);
    }
    for (const t of o.yticks || []) {
      const y = sy(t);
      svgEl("line", { x1: left - 4, x2: left, y1: y, y2: y, class: "sx-tick" }, axes);
      svgEl("text", { x: left - 7, y: y + 4, "text-anchor": "end", text: o.yfmt ? o.yfmt(t) : t }, axes);
    }
    if (o.xlab) svgEl("text", { x: left + w / 2, y: top + h + 34, "text-anchor": "middle", class: "sx-label", text: o.xlab }, axes);
    if (o.ylab)
      svgEl(
        "text",
        {
          x: 0,
          y: 0,
          transform: `translate(${left - (o.ylabOffset || 44)},${top + h / 2}) rotate(-90)`,
          "text-anchor": "middle",
          class: "sx-label",
          text: o.ylab,
        },
        axes
      );
    if (o.title) svgEl("text", { x: left, y: top - 8, class: "sx-title", text: o.title }, axes);

    const layer = svgEl("g", { "clip-path": `url(#${id})` }, svg);
    const over = svgEl("g", {}, svg);
    return {
      sx,
      sy,
      layer,
      over,
      clear() {
        layer.textContent = "";
        over.textContent = "";
      },
      path(points, attrs, parent) {
        const d = points
          .filter((p) => isFinite(p[1]))
          .map((p, i) => (i ? "L" : "M") + sx(p[0]).toFixed(2) + "," + sy(p[1]).toFixed(2))
          .join("");
        return svgEl("path", Object.assign({ d, fill: "none" }, attrs), parent || layer);
      },
    };
  }

  function newSvg(root, width, height, label) {
    const box = el("div", { class: "sx-figure" }, root);
    const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": label }, box);
    return svg;
  }

  function legend(svg, x, y, items) {
    const g = svgEl("g", { class: "sx-legend" }, svg);
    items.forEach((item, i) => {
      const yy = y + i * 17;
      svgEl("line", Object.assign({ x1: x, x2: x + 22, y1: yy, y2: yy }, item.attrs), g);
      svgEl("text", { x: x + 28, y: yy + 4, text: item.label }, g);
    });
    return g;
  }

  // ---------- 13.2 likelihood ----------

  function likelihoodWidget(root) {
    const state = { n: 300, y: 11 };
    const ctl = controls(root);
    const presets = buttons(ctl, "データ", [
      { label: "300 人（出血 11 人）", onClick: () => setData(300, 11) },
      { label: "3000 人（出血 136 人）", onClick: () => setData(3000, 136) },
    ]);
    const nSlider = slider(ctl, {
      label: "人数",
      min: 20,
      max: 3000,
      step: 10,
      value: state.n,
      format: (v) => v + " 人",
      onInput: (v) => {
        const share = state.y / state.n;
        state.n = v;
        state.y = Math.min(Math.round(share * v), maxEvents(v));
        ySlider.setMax(maxEvents(v));
        ySlider.set(state.y);
        presets.select(-1);
        draw();
      },
    });
    const maxEvents = (n) => Math.floor(0.12 * n);
    const ySlider = slider(ctl, {
      label: "出血した人",
      min: 0,
      max: maxEvents(state.n),
      step: 1,
      value: state.y,
      format: (v) => v + " 人",
      onInput: (v) => {
        state.y = v;
        presets.select(-1);
        draw();
      },
    });
    presets.select(0);

    function setData(n, y) {
      state.n = n;
      state.y = y;
      nSlider.set(n);
      ySlider.setMax(maxEvents(n));
      ySlider.set(y);
      draw();
    }

    const svg = newSvg(root, 640, 330, "出血割合の対数尤度");
    const P = panel(svg, {
      left: 62,
      top: 24,
      width: 420,
      height: 250,
      x: [0, 0.15],
      y: [-6, 0.4],
      xticks: [0, 0.03, 0.06, 0.09, 0.12, 0.15],
      yticks: [-6, -4, -1.92, 0],
      xfmt: (t) => Math.round(100 * t) + "%",
      yfmt: (t) => signed(t, t === -1.92 ? 2 : 0),
      xlab: "出血割合 p",
      ylab: "対数尤度（頂上を 0 にそろえた値）",
    });
    legend(svg, 496, 40, [
      { label: "対数尤度", attrs: { stroke: COLOR.blue, "stroke-width": 2.5 } },
      { label: "放物線（Wald）", attrs: { stroke: COLOR.gray, "stroke-width": 1.5, "stroke-dasharray": "5 4" } },
      { label: "頂上から 1.92", attrs: { stroke: COLOR.red, "stroke-width": 1.5, "stroke-dasharray": "2 3" } },
    ]);
    const table = readout(root);

    function loglik(p) {
      const { n, y } = state;
      if (p <= 0 || p >= 1) return y === 0 && p === 0 ? 0 : -Infinity;
      return (y ? y * Math.log(p) : 0) + (n - y) * Math.log(1 - p);
    }

    function draw() {
      const { n, y } = state;
      const mle = y / n;
      const top = loglik(mle);
      const se = Math.sqrt((mle * (1 - mle)) / n);
      const rel = (p) => loglik(p) - top;
      const drop = (p) => rel(p) + 1.92;
      const lrLow = y === 0 ? 0 : bisect(drop, 1e-9, mle);
      const lrHigh = bisect(drop, mle, 0.999);

      P.clear();
      const ps = [];
      for (let i = 0; i <= 600; i++) ps.push((0.15 * i) / 600);
      P.path([[0, -1.92], [0.15, -1.92]], { stroke: COLOR.red, "stroke-width": 1.5, "stroke-dasharray": "2 3" });
      if (se > 0) {
        P.path(
          ps.map((p) => [p, -((p - mle) ** 2) / (2 * se * se)]),
          { stroke: COLOR.gray, "stroke-width": 1.5, "stroke-dasharray": "5 4" }
        );
      }
      P.path(ps.map((p) => [p, rel(p)]), { stroke: COLOR.blue, "stroke-width": 2.5 });
      // Intervals as bars just above the bottom of the panel.
      const bar = (lo, hi, yv, color) => {
        P.path([[lo, yv], [hi, yv]], { stroke: color, "stroke-width": 4, "stroke-linecap": "butt" });
      };
      bar(lrLow, lrHigh, -5.2, COLOR.red);
      if (se > 0) bar(Math.max(mle - 1.96 * se, -1), mle + 1.96 * se, -5.6, COLOR.gray);
      svgEl("text", { x: P.sx(0.151), y: P.sy(-5.2) + 4, class: "sx-note", text: "尤度比の区間" }, P.over);
      svgEl("text", { x: P.sx(0.151), y: P.sy(-5.6) + 4, class: "sx-note", text: "Wald の区間" }, P.over);
      svgEl("line", { x1: P.sx(mle), x2: P.sx(mle), y1: P.sy(0.4), y2: P.sy(-6), class: "sx-guide" }, P.layer);

      // Two decimals once the interval is narrow, as in the table of 13.2.
      const digits = se > 0 && se < 0.005 ? 2 : 1;
      table([
        ["データ", `${n} 人中 ${y} 人が出血`],
        ["最尤推定", pct(mle)],
        ["標準誤差", se > 0 ? pct(se, 2) : "計算できない（出血 0 人）"],
        ["Wald の 95% 信頼区間", se > 0 ? range(mle - 1.96 * se, mle + 1.96 * se, digits) : "計算できない"],
        ["尤度比の 95% 信頼区間", range(lrLow, lrHigh, digits)],
      ]);
    }
    draw();
  }

  // ---------- 13.5 Bayes for a proportion ----------

  function bayesWidget(root) {
    const state = { n: 300, y: 11, mean: 0.05, size: 100 };
    const ctl = controls(root);
    buttons(ctl, "データ", [
      { label: "300 人（出血 11 人）", onClick: () => setData(300, 11) },
      { label: "3000 人（出血 136 人）", onClick: () => setData(3000, 136) },
    ]).select(0);
    const priors = buttons(ctl, "事前分布", [
      { label: "平ら", onClick: () => setPrior(0.5, 2) },
      { label: "過去の研究（5%）", onClick: () => setPrior(0.05, 100) },
      { label: "外れた事前分布（10%）", onClick: () => setPrior(0.1, 200) },
    ]);
    priors.select(1);
    const meanSlider = slider(ctl, {
      label: "事前分布の中心",
      min: 0.01,
      max: 0.5,
      step: 0.005,
      value: state.mean,
      format: (v) => pct(v),
      onInput: (v) => {
        state.mean = v;
        priors.select(-1);
        draw();
      },
    });
    // Prior size in patients' worth of information, on a rough log scale.
    const SIZES = [2, 3, 5, 10, 20, 30, 50, 100, 150, 200, 300, 500, 1000, 2000];
    const sizeSlider = slider(ctl, {
      label: "事前の情報（何人分）",
      min: 0,
      max: SIZES.length - 1,
      step: 1,
      value: SIZES.indexOf(state.size),
      toValue: (i) => SIZES[i],
      format: (v) => v + " 人分",
      onInput: (v) => {
        state.size = v;
        priors.select(-1);
        draw();
      },
    });

    function setData(n, y) {
      state.n = n;
      state.y = y;
      draw();
    }
    function setPrior(mean, size) {
      state.mean = mean;
      state.size = size;
      meanSlider.set(mean);
      sizeSlider.set(SIZES.indexOf(size));
      draw();
    }

    const svg = newSvg(root, 640, 320, "事前分布、尤度、事後分布");
    let P = null;
    legend(svg, 500, 40, [
      { label: "事前分布", attrs: { stroke: COLOR.gray, "stroke-width": 2, "stroke-dasharray": "2 3" } },
      { label: "尤度（面積 1）", attrs: { stroke: COLOR.red, "stroke-width": 1.5, "stroke-dasharray": "6 4" } },
      { label: "事後分布", attrs: { stroke: COLOR.blue, "stroke-width": 2.5 } },
    ]);
    const plotLayer = svgEl("g", {}, svg);
    const table = readout(root);

    function draw() {
      const { n, y, mean, size } = state;
      const a = mean * size;
      const b = (1 - mean) * size;
      const A = a + y;
      const B = b + n - y;
      const xs = [];
      for (let i = 1; i < 600; i++) xs.push((0.2 * i) / 600);
      const dens = (aa, bb) => xs.map((x) => [x, Math.exp(betaLogPdf(x, aa, bb))]);
      const prior = dens(a, b);
      const lik = dens(y + 1, n - y + 1);
      const post = dens(A, B);
      const ymax = 1.1 * Math.max(...post.map((p) => p[1]), ...lik.map((p) => p[1]));
      const step = ymax > 60 ? 20 : ymax > 30 ? 10 : 5;
      const yticks = [];
      for (let t = 0; t <= ymax; t += step) yticks.push(t);

      plotLayer.textContent = "";
      P = panel(plotLayer, {
        left: 52,
        top: 24,
        width: 430,
        height: 240,
        x: [0, 0.2],
        y: [0, ymax],
        xticks: [0, 0.05, 0.1, 0.15, 0.2],
        yticks,
        xfmt: (t) => Math.round(100 * t) + "%",
        xlab: "出血割合",
        ylab: "確率密度",
        ylabOffset: 36,
      });
      P.path(prior, { stroke: COLOR.gray, "stroke-width": 2, "stroke-dasharray": "2 3" });
      P.path(lik, { stroke: COLOR.red, "stroke-width": 1.5, "stroke-dasharray": "6 4" });
      P.path(post, { stroke: COLOR.blue, "stroke-width": 2.5 });

      const [lo, hi] = betaQuantiles(A, B, [0.025, 0.975]);
      table([
        ["事前分布", `Beta(${fmtAB(a)}, ${fmtAB(b)})：中心 ${pct(mean)}、${size} 人分`],
        ["データ", `${n} 人中 ${y} 人が出血（最尤推定 ${pct(y / n)}）`],
        ["事後分布の平均", pct(A / (A + B))],
        ["95% 信用区間", range(lo, hi)],
      ]);
    }
    const fmtAB = (v) => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(1));
    draw();
  }

  // ---------- 14.2 partial pooling ----------

  async function shrinkageWidget(root) {
    const rows = await loadCsv("ch14_rates.csv");
    const n = rows.map((r) => r.n);
    const y = rows.map((r) => r.events);
    const truth = rows.map((r) => r.true_rate);
    const own = rows.map((r) => r.events / r.n);
    const H = rows.length;
    const total = y.reduce((s, v) => s + v, 0) / n.reduce((s, v) => s + v, 0);

    // Marginal log-likelihood of the random-intercept model, integrating the
    // hospital effect on a grid (trapezoid rule with normal weights).
    const K = 121;
    const z = [];
    const wz = [];
    for (let k = 0; k < K; k++) {
      const t = -6 + (12 * k) / (K - 1);
      z.push(t);
      wz.push(Math.exp(-0.5 * t * t));
    }
    const wsum = wz.reduce((s, v) => s + v, 0);
    for (let k = 0; k < K; k++) wz[k] /= wsum;

    function marginal(mu, sigma) {
      let ll = 0;
      for (let h = 0; h < H; h++) {
        let m = -Infinity;
        const lp = new Float64Array(K);
        for (let k = 0; k < K; k++) {
          const eta = mu + sigma * z[k];
          lp[k] = -y[h] * Math.log1p(Math.exp(-eta)) - (n[h] - y[h]) * Math.log1p(Math.exp(eta));
          if (lp[k] > m) m = lp[k];
        }
        let s = 0;
        for (let k = 0; k < K; k++) s += wz[k] * Math.exp(lp[k] - m);
        ll += m + Math.log(s);
      }
      return ll;
    }

    const sigmas = [];
    const mus = [];
    const lls = [];
    for (let i = 0; i <= 150; i++) {
      const s = i / 100;
      const mu = golden((m) => marginal(m, s), -5, -1);
      sigmas.push(s);
      mus.push(mu);
      lls.push(marginal(mu, s));
    }
    const best = lls.indexOf(Math.max(...lls));
    // Refine the maximum between grid points.
    const sigmaHat = golden((s) => marginal(golden((m) => marginal(m, s), -5, -1), s), Math.max(0, sigmas[best] - 0.01), sigmas[best] + 0.01);
    const muAt = (s) => golden((m) => marginal(m, s), -5, -1);

    // Conditional modes of the hospital effects (what fit_mixed reports).
    function pooled(mu, sigma) {
      if (sigma < 1e-6) return n.map(() => expit(mu));
      return n.map((nh, h) => {
        let u = 0;
        for (let it = 0; it < 50; it++) {
          const p = expit(mu + u);
          const g = y[h] - nh * p - u / (sigma * sigma);
          const hess = -nh * p * (1 - p) - 1 / (sigma * sigma);
          u -= g / hess;
        }
        return expit(mu + u);
      });
    }
    const rmse = (est) => Math.sqrt(est.reduce((s, e, h) => s + (e - truth[h]) ** 2, 0) / H);
    const rmseCurve = sigmas.map((s, i) => [s, rmse(pooled(mus[i], s))]);
    const rmseOwn = rmse(own);
    const rmseComplete = rmse(n.map(() => total));

    const ctl = controls(root);
    const state = { sigma: Math.round(100 * sigmaHat) / 100 };
    const sSlider = slider(ctl, {
      label: "施設差 σ（対数オッズの標準偏差）",
      min: 0,
      max: 1.5,
      step: 0.01,
      value: state.sigma,
      format: (v) => v.toFixed(2),
      onInput: (v) => {
        state.sigma = v;
        draw();
      },
    });
    buttons(ctl, "σ を", [
      { label: "0（完全プーリング）", onClick: () => setSigma(0) },
      { label: `データから推定（${sigmaHat.toFixed(2)}）`, onClick: () => setSigma(Math.round(100 * sigmaHat) / 100) },
      { label: "1.5（プーリングなしに近い）", onClick: () => setSigma(1.5) },
    ]).select(1);
    function setSigma(v) {
      state.sigma = v;
      sSlider.set(v);
      draw();
    }

    const svg = newSvg(root, 680, 360, "施設ごとの出血割合の縮小");
    const L = panel(svg, {
      left: 58,
      top: 24,
      width: 250,
      height: 280,
      x: [-0.25, 1.25],
      y: [0, 0.12],
      yticks: [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12],
      yfmt: (t) => Math.round(100 * t) + "%",
      ylab: "出血割合",
      ylabOffset: 42,
    });
    svgEl("text", { x: L.sx(0), y: 322, "text-anchor": "middle", class: "sx-label", text: "自施設のデータだけ" }, svg);
    svgEl("text", { x: L.sx(1), y: 322, "text-anchor": "middle", class: "sx-label", text: "部分プーリング" }, svg);
    const R1 = panel(svg, {
      left: 420,
      top: 24,
      width: 240,
      height: 110,
      x: [0, 1.5],
      y: [Math.min(...lls) - Math.max(...lls) - 0.5, 0.5],
      yticks: [Math.ceil(Math.min(...lls) - Math.max(...lls)), 0],
      yfmt: (t) => signed(t, 0),
      xticks: [0, 0.5, 1, 1.5],
      title: "対数尤度（頂上を 0 にそろえた値）",
    });
    const R2 = panel(svg, {
      left: 420,
      top: 194,
      width: 240,
      height: 110,
      x: [0, 1.5],
      y: [0, 0.03],
      yticks: [0, 0.01, 0.02, 0.03],
      yfmt: (t) => Math.round(100 * t) + "%",
      xticks: [0, 0.5, 1, 1.5],
      xlab: "施設差 σ",
      title: "本当の割合からの誤差",
    });
    const llTop = Math.max(...lls);
    R1.path(sigmas.map((s, i) => [s, lls[i] - llTop]), { stroke: COLOR.ink, "stroke-width": 1.8 });
    R2.path(rmseCurve, { stroke: COLOR.ink, "stroke-width": 1.8 });
    R2.path([[0, rmseOwn], [1.5, rmseOwn]], { stroke: COLOR.red, "stroke-width": 1.2, "stroke-dasharray": "4 3" });
    svgEl("text", { x: R2.sx(1.48), y: R2.sy(rmseOwn) - 4, "text-anchor": "end", class: "sx-note", fill: COLOR.red, text: "自施設のデータだけ" }, R2.layer);
    const markers = svgEl("g", {}, svg);
    const table = readout(root);
    const H05 = rows.findIndex((r) => r.hospital === "H05");
    const H19 = rows.findIndex((r) => r.hospital === "H19");

    function draw() {
      const s = state.sigma;
      const mu = s === 0 ? logit(total) : muAt(s);
      const est = pooled(mu, s);
      L.clear();
      L.path([[-0.25, expit(mu)], [1.25, expit(mu)]], { stroke: COLOR.gray, "stroke-width": 1, "stroke-dasharray": "4 3" });
      for (let h = 0; h < H; h++) {
        L.path([[0, own[h]], [1, est[h]]], { stroke: COLOR.gray, "stroke-width": 0.8, opacity: 0.7 });
      }
      for (let h = 0; h < H; h++) {
        const r = Math.sqrt(n[h]) * 0.45 + 1.5;
        svgEl("circle", { cx: L.sx(0), cy: L.sy(own[h]), r, fill: COLOR.red, "fill-opacity": 0.8 }, L.layer);
        svgEl("circle", { cx: L.sx(1), cy: L.sy(est[h]), r, fill: COLOR.blue, "fill-opacity": 0.8 }, L.layer);
      }
      markers.textContent = "";
      for (const P of [R1, R2]) {
        svgEl("line", { x1: P.sx(s), x2: P.sx(s), y1: P.sy(P === R1 ? 0.5 : 0.03), y2: P.sy(P === R1 ? R1Low : 0), stroke: COLOR.blue, "stroke-width": 2 }, markers);
      }
      table([
        ["施設差 σ", s.toFixed(2) + (Math.abs(s - sigmaHat) < 0.006 ? "（データから推定した値）" : "")],
        ["全体の中心（点線）", pct(expit(mu))],
        ["H05（80 人中 1 人、本当は " + pct(truth[H05]) + "）", pct(est[H05])],
        ["H19（16 人中 0 人、本当は " + pct(truth[H19]) + "）", pct(est[H19])],
        ["本当の割合からの誤差", `${pct(rmse(est))}（完全プーリング ${pct(rmseComplete)}、自施設のデータだけ ${pct(rmseOwn)}）`],
      ]);
    }
    const R1Low = Math.min(...lls) - llTop - 0.5;
    draw();
  }

  // ---------- 6.3 bias factor and E-value ----------

  async function evalueWidget(root) {
    const [row] = await loadCsv("ch6_evalue.csv");
    const rr = row.risk_ratio;
    const upper = row.conf_high;
    const needPoint = 1 / rr;
    const needCi = 1 / upper;
    const state = { eu: 2.27, uy: 2.4 };

    const ctl = controls(root);
    const eu = slider(ctl, {
      label: "交絡因子とクリップのリスク比 RR_EU",
      min: 1,
      max: 6,
      step: 0.01,
      value: state.eu,
      format: (v) => v.toFixed(2),
      onInput: (v) => {
        state.eu = v;
        pre.select(-1);
        draw();
      },
    });
    const uy = slider(ctl, {
      label: "交絡因子と出血のリスク比 RR_UY",
      min: 1,
      max: 6,
      step: 0.01,
      value: state.uy,
      format: (v) => v.toFixed(2),
      onInput: (v) => {
        state.uy = v;
        pre.select(-1);
        draw();
      },
    });
    const ev = row.e_value;
    const pre = buttons(ctl, "例", [
      { label: "抗血栓薬くらい（2.27、2.4）", onClick: () => set(2.27, 2.4) },
      { label: `E-value（${ev.toFixed(2)}、${ev.toFixed(2)}）`, onClick: () => set(ev, ev) },
    ]);
    pre.select(0);
    function set(a, b) {
      state.eu = a;
      state.uy = b;
      eu.set(a);
      uy.set(b);
      draw();
    }

    const svg = newSvg(root, 600, 360, "未測定交絡の強さとバイアス因子");
    const P = panel(svg, {
      left: 60,
      top: 20,
      width: 300,
      height: 290,
      x: [1, 6],
      y: [1, 6],
      xticks: [1, 2, 3, 4, 5, 6],
      yticks: [1, 2, 3, 4, 5, 6],
      xlab: "RR_EU（交絡因子とクリップ）",
      ylab: "RR_UY（交絡因子と出血）",
      ylabOffset: 40,
    });
    // Curves where the bias factor equals what is needed: RR_UY = B(RR_EU − 1)/(RR_EU − B).
    const curve = (B) => {
      const pts = [];
      for (let i = 0; i <= 400; i++) {
        const x = B + 0.0005 + ((6 - B) * i) / 400;
        pts.push([x, (B * (x - 1)) / (x - B)]);
      }
      return pts;
    };
    const area = curve(needPoint).filter((p) => p[1] <= 7);
    const fillPts = area.concat([[6, 6], [area[0][0], 6]]);
    P.path(fillPts, { fill: COLOR.red, "fill-opacity": 0.08, stroke: "none" });
    P.path(curve(needPoint), { stroke: COLOR.red, "stroke-width": 2 });
    P.path(curve(needCi), { stroke: COLOR.red, "stroke-width": 1.5, "stroke-dasharray": "5 4" });
    legend(svg, 380, 40, [
      { label: "推定値 " + rr.toFixed(2) + " を 1 にできる", attrs: { stroke: COLOR.red, "stroke-width": 2 } },
      { label: "区間の上限 " + upper.toFixed(2) + " を 1 にできる", attrs: { stroke: COLOR.red, "stroke-width": 1.5, "stroke-dasharray": "5 4" } },
    ]);
    const table = readout(root);

    function draw() {
      const { eu: a, uy: b } = state;
      const B = (a * b) / (a + b - 1);
      P.over.textContent = "";
      svgEl("circle", { cx: P.sx(a), cy: P.sy(b), r: 6, fill: COLOR.blue, stroke: "#fff", "stroke-width": 1.5 }, P.over);
      const moved = rr * B;
      const movedHigh = upper * B;
      const verdict =
        B >= needPoint
          ? "推定値を 1 まで動かせます。この強さの交絡因子なら、効果を説明しきれます。"
          : B >= needCi
            ? "推定値は 1 に届きませんが、区間の上限は 1 を超えます。「効果がない」とは言い切れなくなります。"
            : "推定値も区間の上限も 1 に届きません。この強さでは、効果を説明しきれません。";
      table([
        ["バイアス因子 B", `${B.toFixed(2)}（推定値を 1 にするには ${needPoint.toFixed(2)}、区間の上限なら ${needCi.toFixed(2)}）`],
        ["1 に最も近づけたリスク比", `${moved.toFixed(2)}（区間の上限 ${movedHigh.toFixed(2)}）`],
        ["読み方", verdict],
      ]);
    }
    draw();
  }

  // ---------- 9.4 decision curve ----------

  async function dcaWidget(root) {
    const rows = await loadCsv("ch9_dca.csv");
    const models = [...new Set(rows.map((r) => r.model))];
    const names = { "Seven predictors": "七つの変数のモデル", "Age + antithrombotic": "簡単なモデル" };
    const colors = [COLOR.blue, COLOR.orange];
    const thresholds = [...new Set(rows.map((r) => Math.round(r.threshold * 1000) / 1000))].sort((a, b) => a - b);
    const at = (model, t) => rows.find((r) => r.model === model && Math.abs(r.threshold - t) < 1e-6);
    const state = { i: thresholds.indexOf(0.05) };

    const ctl = controls(root);
    slider(ctl, {
      label: "閾値 p_t",
      min: 0,
      max: thresholds.length - 1,
      step: 1,
      value: state.i,
      toValue: (i) => i,
      format: (i) => pct(thresholds[i], 0),
      onInput: (i) => {
        state.i = i;
        draw();
      },
    });

    const svg = newSvg(root, 640, 330, "決定曲線");
    const P = panel(svg, {
      left: 66,
      top: 20,
      width: 400,
      height: 250,
      x: [0, 0.2],
      y: [-0.01, 0.036],
      xticks: [0, 0.05, 0.1, 0.15, 0.2],
      yticks: [-0.01, 0, 0.01, 0.02, 0.03],
      xfmt: (t) => Math.round(100 * t) + "%",
      yfmt: (t) => signed(t, 2),
      xlab: "閾値 p_t",
      ylab: "正味の利益",
      ylabOffset: 50,
    });
    const first = models[0];
    P.path(thresholds.map((t) => [t, at(first, t).treat_all]), { stroke: COLOR.gray, "stroke-width": 1.5, "stroke-dasharray": "5 4" });
    P.path([[0, 0], [0.2, 0]], { stroke: COLOR.ink, "stroke-width": 1.5, "stroke-dasharray": "2 3" });
    models.forEach((m, j) => {
      P.path(thresholds.map((t) => [t, at(m, t).net_benefit]), { stroke: colors[j], "stroke-width": 2.2 });
    });
    legend(
      svg,
      480,
      40,
      models
        .map((m, j) => ({ label: names[m] || m, attrs: { stroke: colors[j], "stroke-width": 2.2 } }))
        .concat([
          { label: "全員を入院", attrs: { stroke: COLOR.gray, "stroke-width": 1.5, "stroke-dasharray": "5 4" } },
          { label: "誰も入院させない", attrs: { stroke: COLOR.ink, "stroke-width": 1.5, "stroke-dasharray": "2 3" } },
        ])
    );
    const table = readout(root);

    function draw() {
      const t = thresholds[state.i];
      P.over.textContent = "";
      svgEl("line", { x1: P.sx(t), x2: P.sx(t), y1: P.sy(0.036), y2: P.sy(-0.01), stroke: COLOR.red, "stroke-width": 1.5 }, P.over);
      models.forEach((m, j) => {
        const v = at(m, t).net_benefit;
        svgEl("circle", { cx: P.sx(t), cy: P.sy(v), r: 4.5, fill: colors[j], stroke: "#fff", "stroke-width": 1 }, P.over);
      });
      const all = at(first, t).treat_all;
      const ratio = (1 - t) / t;
      const per100 = (v) => `${signed(v, 4)}（100 人あたり ${signed(100 * v, 1)} 人分）`;
      const out = [["閾値の意味", `出血する 1 人を入院で見るためなら、出血しない ${Math.abs(ratio - Math.round(ratio)) < 0.05 ? Math.round(ratio) : ratio.toFixed(1)} 人の入院を許す`]];
      models.forEach((m) => out.push([names[m] || m, per100(at(m, t).net_benefit)]));
      out.push(["全員を入院", all < -0.01 ? `${signed(all, 4)}（図の範囲より下）` : per100(all)]);
      out.push(["誰も入院させない", "0"]);
      table(out);
    }
    draw();
  }

  // ---------- boot ----------

  const WIDGETS = {
    likelihood: likelihoodWidget,
    "bayes-rate": bayesWidget,
    shrinkage: shrinkageWidget,
    evalue: evalueWidget,
    dca: dcaWidget,
  };

  function boot(scope) {
    for (const root of (scope || document).querySelectorAll(".sx-widget[data-widget]")) {
      if (root.dataset.ready) continue;
      root.dataset.ready = "1";
      const make = WIDGETS[root.dataset.widget];
      if (!make) continue;
      root.textContent = "";
      Promise.resolve()
        .then(() => make(root))
        .catch((err) => {
          root.textContent = "図を読み込めませんでした（" + err.message + "）。";
        });
    }
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(({ body }) => boot(body));
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => boot(document));
  } else {
    boot(document);
  }
})();
