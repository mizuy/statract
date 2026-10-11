// Interactive figures, see docs/javascripts/theory-widgets.js.
(function () {
  "use strict";
  const { COLOR, svgEl, pct, signed, rng, controls, slider, buttons, readout, panel, newSvg, legend } = window.SX;

  // A round step giving about four to six ticks between 0 and max.
  function niceStep(max) {
    const raw = max / 5;
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const m = raw / p;
    return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
  }
  function ticksTo(max) {
    const s = niceStep(max);
    const out = [];
    for (let t = 0; t <= max + 1e-12; t += s) out.push(Math.round(t / s) * s);
    return { ticks: out, step: s };
  }
  const trim = (x, d) => String(Number(x.toFixed(d)));

  // ---------- 15.5 hazard, survival and proportional hazards ----------

  // Baseline: the average patient without a clip. As in the data of this chapter,
  // the cumulative hazard reaches 0.05 at day 30: H0(t) = 0.05 (t / 30)^shape.
  const H30 = 0.05;
  const DAYS = 30;

  function hazardWidget(root) {
    const state = { shape: 0.5, hr: 0.69, ph: true };
    const ctl = controls(root);
    const base = buttons(ctl, "ベースラインハザード", [
      { label: "一定（形 1）", onClick: () => setShape(1) },
      { label: "Weibull 形 0.5（この章のデータ）", onClick: () => setShape(0.5) },
    ]);
    base.select(1);
    const shapeSlider = slider(ctl, {
      label: "Weibull の形",
      min: 0.3,
      max: 3,
      step: 0.05,
      value: state.shape,
      format: (v) => v.toFixed(2),
      onInput: (v) => {
        state.shape = v;
        base.select(v === 1 ? 0 : v === 0.5 ? 1 : -1);
        draw();
      },
    });
    slider(ctl, {
      label: "クリップのハザード比",
      min: 0.2,
      max: 3,
      step: 0.01,
      value: state.hr,
      format: (v) => v.toFixed(2),
      onInput: (v) => {
        state.hr = v;
        draw();
      },
    });
    buttons(ctl, "効果", [
      { label: "どの時点でも同じ（比例ハザード）", onClick: () => setPh(true) },
      { label: "30 日に向けて弱まる", onClick: () => setPh(false) },
    ]).select(0);
    function setShape(v) {
      state.shape = v;
      shapeSlider.set(v);
      draw();
    }
    function setPh(v) {
      state.ph = v;
      draw();
    }

    const svg = newSvg(root, 680, 560, "ハザード、累積発生割合、対数累積ハザード");
    const plotLayer = svgEl("g", {}, svg);
    legend(svg, 440, 330, [
      { label: "クリップなし（ベースライン）", attrs: { stroke: COLOR.gray, "stroke-width": 2.2 } },
      { label: "クリップあり", attrs: { stroke: COLOR.blue, "stroke-width": 2.2 } },
    ]);
    const note = svgEl("text", { x: 440, y: 380, class: "sx-note" }, svg);
    const note2 = svgEl("text", { x: 440, y: 396, class: "sx-note" }, svg);
    const table = readout(root);

    function draw() {
      const { shape: k, hr, ph } = state;
      const H0 = (t) => H30 * Math.pow(t / DAYS, k);
      const h0 = (t) => (H30 * k * Math.pow(t / DAYS, k - 1)) / DAYS;
      const hrAt = (t) => (ph ? hr : Math.exp(Math.log(hr) * (1 - t / DAYS)));
      // Grid on a square-root scale, so the steep start is resolved.
      const M = 600;
      const ts = [];
      for (let i = 0; i <= M; i++) ts.push(DAYS * (i / M) ** 2);
      const H1 = [0];
      for (let i = 1; i <= M; i++) H1.push(H1[i - 1] + (H0(ts[i]) - H0(ts[i - 1])) * hrAt((ts[i] + ts[i - 1]) / 2));
      const g0 = ts.map((t) => H0(t));
      const H1at = (t) => {
        let i = 1;
        while (i < M && ts[i] < t) i++;
        const f = (t - ts[i - 1]) / (ts[i] - ts[i - 1]);
        return H1[i - 1] + f * (H1[i] - H1[i - 1]);
      };

      plotLayer.textContent = "";
      // Hazard, per 1000 person-days, from day 0.5 on.
      const hs = ts.filter((t) => t >= 0.5);
      const hmaxRaw = 1000 * Math.max(...hs.map((t) => h0(t) * Math.max(1, hrAt(t))));
      const ht = ticksTo(1.1 * hmaxRaw);
      const hmax = ht.ticks[ht.ticks.length - 1] < 1.1 * hmaxRaw ? ht.ticks[ht.ticks.length - 1] + ht.step : ht.ticks[ht.ticks.length - 1];
      const A = panel(plotLayer, {
        left: 58,
        top: 26,
        width: 240,
        height: 200,
        x: [0, DAYS],
        y: [0, hmax],
        xticks: [0, 7, 14, 21, 30],
        yticks: ht.ticks.filter((t) => t <= hmax),
        yfmt: (t) => trim(t, 2),
        xlab: "切除からの日数",
        ylab: "1000 人日あたり",
        ylabOffset: 42,
        title: "ハザード",
      });
      A.path(hs.map((t) => [t, 1000 * h0(t)]), { stroke: COLOR.gray, "stroke-width": 2.2 });
      A.path(hs.map((t) => [t, 1000 * h0(t) * hrAt(t)]), { stroke: COLOR.blue, "stroke-width": 2.2 });

      // Cumulative incidence 1 − S(t) = 1 − exp(−H(t)).
      const cmaxRaw = Math.max(1 - Math.exp(-g0[M]), 1 - Math.exp(-H1[M]));
      const ct = ticksTo(1.15 * cmaxRaw);
      const cmax = ct.ticks[ct.ticks.length - 1] < 1.15 * cmaxRaw ? ct.ticks[ct.ticks.length - 1] + ct.step : ct.ticks[ct.ticks.length - 1];
      const B = panel(plotLayer, {
        left: 410,
        top: 26,
        width: 250,
        height: 200,
        x: [0, DAYS],
        y: [0, cmax],
        xticks: [0, 7, 14, 21, 30],
        yticks: ct.ticks.filter((t) => t <= cmax),
        yfmt: (t) => trim(100 * t, 1) + "%",
        xlab: "切除からの日数",
        ylabOffset: 46,
        title: "累積発生割合（1 − 生存曲線）",
      });
      B.path(ts.map((t, i) => [t, 1 - Math.exp(-g0[i])]), { stroke: COLOR.gray, "stroke-width": 2.2 });
      B.path(ts.map((t, i) => [t, 1 - Math.exp(-H1[i])]), { stroke: COLOR.blue, "stroke-width": 2.2 });

      // log H(t) against log t: parallel lines under proportional hazards.
      const lt = ts.filter((t) => t >= 0.5);
      const lo = Math.log(0.5);
      const hi = Math.log(DAYS);
      const ys = lt.map((t) => Math.log(H0(t))).concat(lt.map((t) => Math.log(H1at(t))));
      const ylo = Math.floor(Math.min(...ys));
      const yhi = Math.ceil(Math.max(...ys));
      const yt = [];
      for (let v = ylo; v <= yhi; v++) yt.push(v);
      const C = panel(plotLayer, {
        left: 58,
        top: 300,
        width: 340,
        height: 200,
        x: [lo, hi],
        y: [ylo, yhi],
        xticks: [1, 3, 7, 14, 30].map(Math.log),
        xfmt: (t) => String(Math.round(Math.exp(t))),
        yticks: yt,
        yfmt: (t) => signed(t, 0),
        xlab: "切除からの日数（対数目盛り）",
        ylab: "log 累積ハザード",
        ylabOffset: 34,
        title: "対数累積ハザード",
      });
      C.path(lt.map((t) => [Math.log(t), Math.log(H0(t))]), { stroke: COLOR.gray, "stroke-width": 2.2 });
      C.path(lt.map((t) => [Math.log(t), Math.log(H1at(t))]), { stroke: COLOR.blue, "stroke-width": 2.2 });
      // Gap at day 3 and day 30.
      for (const t of [3, 30]) {
        const x = C.sx(Math.log(t)) - (t === 30 ? 3 : 0);
        svgEl("line", { x1: x, x2: x, y1: C.sy(Math.log(H0(t))), y2: C.sy(Math.log(H1at(t))), stroke: COLOR.red, "stroke-width": 2 }, C.over);
      }
      const gap3 = Math.log(H1at(3) / H0(3));
      const gap30 = Math.log(H1[M] / g0[M]);
      note.textContent = "赤い線：二本の間の幅";
      note2.textContent = ph ? `どこでも log HR = ${signed(Math.log(hr), 2)}（平行）` : `3 日目 ${signed(gap3, 2)}、30 日目 ${signed(gap30, 2)}（平行でない）`;

      const r30 = (1 - Math.exp(-H1[M])) / (1 - Math.exp(-g0[M]));
      const per1000 = (t) => `なし ${(1000 * h0(t)).toFixed(2)}、あり ${(1000 * h0(t) * hrAt(t)).toFixed(2)}（比 ${hrAt(t).toFixed(2)}）`;
      table([
        ["1 日目のハザード（1000 人日あたり）", per1000(1)],
        ["14 日目のハザード（1000 人日あたり）", per1000(14)],
        ["30 日までの累積発生割合", `なし ${pct(1 - Math.exp(-g0[M]), 2)}、あり ${pct(1 - Math.exp(-H1[M]), 2)}（リスク比 ${r30.toFixed(2)}）`],
        ["対数累積ハザードの差", `3 日目 ${signed(gap3, 2)}、30 日目 ${signed(gap30, 2)}`],
      ]);
    }
    draw();
  }

  // ---------- 15.2 Kaplan–Meier versus naive proportions ----------

  function kmWidget(root) {
    // Toy data: 200 patients, Weibull (shape 0.5) bleeding times with a true
    // 30-day cumulative incidence of 30%. Each patient also has a fixed chance
    // draw for being lost and a day on which follow-up would stop.
    const N = 200;
    const TRUE30 = 0.3;
    const lam = -Math.log(1 - TRUE30);
    const r = rng(1502);
    const pts = [];
    for (let i = 0; i < N; i++) {
      let u = 0;
      while (u === 0) u = r.uniform();
      const t = DAYS * (-Math.log(u) / lam) ** 2;
      pts.push({ t, lostDraw: r.uniform(), lossDay: DAYS * r.uniform() });
    }
    const complete = pts.filter((p) => p.t <= DAYS).length / N;
    const state = { loss: 0.4 };

    const ctl = controls(root);
    slider(ctl, {
      label: "30 日より前に追跡が途切れる人の割合",
      min: 0,
      max: 0.8,
      step: 0.05,
      value: state.loss,
      format: (v) => Math.round(100 * v) + "%",
      onInput: (v) => {
        state.loss = v;
        draw();
      },
    });

    const svg = newSvg(root, 640, 330, "Kaplan–Meier 法と、打ち切りを無視した割合");
    const P = panel(svg, {
      left: 58,
      top: 24,
      width: 380,
      height: 250,
      x: [0, 32],
      y: [0, 0.6],
      xticks: [0, 7, 14, 21, 30],
      yticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
      yfmt: (t) => Math.round(100 * t) + "%",
      xlab: "切除からの日数",
      ylab: "累積出血割合",
      ylabOffset: 42,
    });
    legend(svg, 455, 40, [
      { label: "Kaplan–Meier 法", attrs: { stroke: COLOR.blue, "stroke-width": 2.5 } },
      { label: "打ち切りがなかった場合", attrs: { stroke: COLOR.gray, "stroke-width": 1.5, "stroke-dasharray": "5 4" } },
      { label: "出血 ÷ 全員", attrs: { stroke: COLOR.orange, "stroke-width": 3 } },
      { label: "出血 ÷ 打ち切りを除いた人", attrs: { stroke: COLOR.red, "stroke-width": 3 } },
    ]);
    svgEl("text", { x: 455, y: 125, class: "sx-note", text: "縦の短い線：打ち切り" }, svg);
    const table = readout(root);

    function stepCurve(times, events) {
      // Cumulative incidence 1 − KM from (time, event) pairs sorted by time.
      const order = times.map((t, i) => i).sort((a, b) => times[a] - times[b]);
      let atRisk = times.length;
      let s = 1;
      const out = [[0, 0]];
      let i = 0;
      while (i < order.length) {
        const t = times[order[i]];
        let d = 0;
        let c = 0;
        while (i < order.length && times[order[i]] === t) {
          if (events[order[i]]) d++;
          else c++;
          i++;
        }
        if (d > 0) {
          out.push([t, 1 - s]);
          s *= 1 - d / atRisk;
          out.push([t, 1 - s]);
        }
        atRisk -= d + c;
      }
      out.push([DAYS, 1 - s]);
      return { points: out, at30: 1 - s };
    }

    function draw() {
      const loss = state.loss;
      const time = [];
      const event = [];
      for (const p of pts) {
        const stop = p.lostDraw < loss ? p.lossDay : DAYS;
        if (p.t <= stop) {
          time.push(p.t);
          event.push(true);
        } else {
          time.push(stop);
          event.push(false);
        }
      }
      const events = event.filter(Boolean).length;
      const lost = time.filter((t, i) => !event[i] && t < DAYS).length;
      const km = stepCurve(time, event);
      const full = stepCurve(pts.map((p) => Math.min(p.t, DAYS)), pts.map((p) => p.t <= DAYS));
      const naiveAll = events / N;
      const naiveDrop = events / (N - lost);

      P.clear();
      P.path(full.points, { stroke: COLOR.gray, "stroke-width": 1.5, "stroke-dasharray": "5 4" });
      P.path(km.points, { stroke: COLOR.blue, "stroke-width": 2.5 });
      // Censoring marks on the KM curve.
      const kmAt = (t) => {
        let v = 0;
        for (const q of km.points) if (q[0] <= t) v = q[1];
        return v;
      };
      time.forEach((t, i) => {
        if (!event[i] && t < DAYS) {
          const y = P.sy(kmAt(t));
          svgEl("line", { x1: P.sx(t), x2: P.sx(t), y1: y - 4, y2: y + 4, stroke: COLOR.blue, "stroke-width": 1, opacity: 0.6 }, P.layer);
        }
      });
      const mark = (v, color) => P.path([[30, v], [32, v]], { stroke: color, "stroke-width": 3 });
      mark(naiveAll, COLOR.orange);
      mark(naiveDrop, COLOR.red);
      svgEl("line", { x1: P.sx(30), x2: P.sx(30), y1: P.sy(0), y2: P.sy(0.6), class: "sx-guide" }, P.layer);

      table([
        ["データ（例として作った 200 人）", `出血 ${events} 人、30 日より前に打ち切り ${lost} 人`],
        ["打ち切りがなかった場合の 30 日の割合", pct(complete)],
        ["Kaplan–Meier 法", pct(km.at30)],
        ["出血 ÷ 全員", `${events} / ${N} = ${pct(naiveAll)}`],
        ["出血 ÷ 打ち切りを除いた人", `${events} / ${N - lost} = ${pct(naiveDrop)}`],
      ]);
    }
    draw();
  }

  window.SX.register("hazard-survival", hazardWidget);
  window.SX.register("km-censoring", kmWidget);
})();
