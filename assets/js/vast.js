(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const C = {
    red: '#E8445F', redSoft: '#F5B9B9', blue: '#4F86B5', blueSoft: '#CFE3F3',
    edge: '#B9D3EA', purple: '#7B5EA7', gray: '#9a9595', band: '#F1F1F1', text: '#333'
  };

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function text(parent, x, y, str, attrs) {
    const t = el('text', Object.assign({ x, y, 'text-anchor': 'middle', 'font-size': 18, fill: C.text, class: 'svg-text' }, attrs || {}), parent);
    t.textContent = str;
    return t;
  }

  function renderTex(node, tex) {
    if (window.katex) katex.render(tex, node, { throwOnError: false });
    else node.textContent = tex;
  }

  function math(parent, x, y, tex, o) {
    o = o || {};
    const w = o.w || 260, h = o.h || 40;
    const fo = el('foreignObject', { x: x - w / 2, y: y - h / 2, width: w, height: h }, parent);
    if (o.fade) fo.setAttribute('style', 'transition: opacity .35s');
    const d = document.createElement('div');
    d.className = 'svg-math';
    d.style.color = o.color || C.text;
    d.style.fontSize = (o.size || 20) + 'px';
    if (o.bold) d.style.fontWeight = 700;
    fo.appendChild(d);
    fo.tex = t => renderTex(d, t);
    fo.moveX = nx => fo.setAttribute('x', nx - w / 2);
    fo.tex(tex);
    return fo;
  }

  function caption(id, tex) {
    const d = document.getElementById(id);
    if (d) renderTex(d, tex);
  }

  const tweenIds = new Map();
  function tween(key, from, to, dur, fn) {
    const id = (tweenIds.get(key) || 0) + 1;
    tweenIds.set(key, id);
    const t0 = performance.now();
    function step() {
      if (tweenIds.get(key) !== id) return;
      const p = Math.min(1, Math.max(0, (performance.now() - t0) / dur));
      const e = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
      fn(from + (to - from) * e);
      if (p < 1) requestAnimationFrame(step);
    }
    step();
  }

  function arrowMarker(svg, id, color) {
    const defs = el('defs', {}, svg);
    const m = el('marker', { id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto' }, defs);
    el('path', { d: 'M0,0 L10,5 L0,10 z', fill: color }, m);
  }

  function chain(svg, n, y) {
    const xs = [...Array(n).keys()].map(i => 70 + i * 122);
    xs.forEach((x, i) => { if (i % 2 === 0) el('rect', { x: x - 52, y: y - 42, width: 104, height: 84, fill: C.band, rx: 4 }, svg); });
    const arrows = [];
    for (let i = 0; i < n - 1; i++) {
      arrows.push(el('line', { x1: xs[i] + 24, y1: y, x2: xs[i + 1] - 26, y2: y, stroke: '#222', 'stroke-width': 2,
        'marker-end': 'url(#' + svg.id + '-arr)', style: 'transition: opacity .4s' }, svg));
    }
    const nodes = xs.map((x, i) => el('circle', { cx: x, cy: y, r: 15, 'stroke-width': 3,
      fill: i === 0 ? C.blue : '#fff', stroke: i === 0 ? C.blue : C.red, style: 'transition: all .4s' }, svg));
    return { xs, arrows, nodes };
  }

  function animEndpoint() {
    const svg = document.getElementById('anim-endpoint');
    if (!svg) return;
    arrowMarker(svg, svg.id + '-arr', '#222');
    const y = 95, N = 8;
    const { xs, arrows, nodes } = chain(svg, N, y);
    const gammas = [];
    for (let t = 0; t < N - 1; t++) {
      gammas.push(math(svg, (xs[t] + xs[t + 1]) / 2, 170, t === 1 ? '\\gamma' : `\\gamma^{${t}}`,
        { w: 80, h: 32, color: C.red, size: 22, fade: true }));
    }
    const bracket = el('path', { fill: 'none', stroke: '#888', 'stroke-width': 1.6, 'stroke-dasharray': '5 4' }, svg);
    const pointer = el('g', {}, svg);
    el('path', { d: 'M0,-8 L0,12 M-7,5 L0,13 L7,5', stroke: C.red, 'stroke-width': 3, fill: 'none' }, pointer);
    const ptrLbl = math(pointer, 0, -22, 'k', { w: 90, h: 26, color: C.red, size: 18 });

    let curX = xs[1];
    function setK(k) {
      nodes.forEach((c, j) => {
        if (j === 0) return;
        c.setAttribute('fill', j === k ? C.red : '#fff');
        c.setAttribute('stroke', j <= k ? C.red : C.gray);
        c.setAttribute('opacity', j <= k ? 1 : 0.3);
        c.setAttribute('r', j === k ? 18 : 15);
      });
      arrows.forEach((a, t) => a.setAttribute('opacity', t < k ? 1 : 0.2));
      gammas.forEach(g => g.setAttribute('opacity', 0));
      gammas.slice(0, k).forEach((g, t) => setTimeout(() => g.setAttribute('opacity', 1), 120 * t));
      tween('endpoint', curX, xs[k], 600, x => {
        pointer.setAttribute('transform', `translate(${x},36)`);
        bracket.setAttribute('d', `M${xs[0]},142 L${xs[0]},150 L${x},150 L${x},142`);
      });
      curX = xs[k];
      ptrLbl.tex(`k = ${k}`);
      const terms = [];
      for (let t = 0; t < k; t++) terms.push(`\\gamma^{${t}} r(s_{${t}}, a_{${t}})`);
      caption('anim-endpoint-cap', `V_{\\text{target}} = ${terms.join(' + ')} + \\gamma^{${k}}\\, V(s_{${k}})`);
    }
    const seq = [3, 6, 2, 7, 4, 1, 5];
    let i = 0;
    setK(seq[0]);
    setInterval(() => { i = (i + 1) % seq.length; setK(seq[i]); }, 2600);
  }

  function animStitch() {
    const svg = document.getElementById('anim-stitch');
    if (!svg) return;
    const root = { x: 70, y: 160 };
    const scores = [[0.31, 0.42], [0.55, 0.38, 0.47], [0.49, 0.74], [0.58, 0.86, 0.40], [0.66, 0.52], [0.71, 0.45, 0.30]];
    const colX = k => 215 + k * 140;
    const ys = n => n === 2 ? [120, 200] : [90, 160, 230];
    const cursor = el('rect', { x: colX(0) - 48, y: 56, width: 96, height: 205, rx: 8, fill: '#FDEBEE', opacity: 0, style: 'transition: opacity .3s' }, svg);
    arrowMarker(svg, svg.id + '-karr', '#888');
    text(svg, 500, 12, 'horizon k', { 'font-size': 15, fill: '#888' });
    el('line', {
      x1: 36, y1: 20, x2: 978, y2: 20, stroke: '#888', 'stroke-width': 1.8,
      'marker-end': 'url(#' + svg.id + '-karr)'
    }, svg);
    const cands = [];
    scores.forEach((col, k) => {
      math(svg, colX(k), 40, `k = ${k + 1}`, { size: 16, color: '#666', w: 90, h: 26 });
      ys(col.length).forEach((y, j) => {
        el('path', { d: `M${root.x + 16},${root.y} C ${root.x + 90},${root.y} ${colX(k) - 70},${y} ${colX(k) - 14},${y}`,
          fill: 'none', stroke: C.edge, 'stroke-width': 1.3, opacity: 1 - k * 0.1 }, svg);
        const c = el('circle', { cx: colX(k), cy: y, r: 13, fill: '#fff', stroke: C.blue, 'stroke-width': 2.5,
          opacity: 1 - k * 0.1, style: 'transition: all .3s' }, svg);
        const s = text(svg, colX(k) + 36, y + 5, col[j].toFixed(2), { 'font-size': 14, fill: '#555', opacity: 0, style: 'transition: opacity .3s' });
        cands.push({ k, j, v: col[j], c, s });
      });
    });
    el('circle', { cx: root.x, cy: root.y, r: 18, fill: C.blue }, svg);
    math(svg, root.x, root.y + 38, 's', { size: 22, w: 40, h: 30 });
    const best = text(svg, 0, 0, 'Optimal target', { fill: C.red, 'font-weight': 700, 'font-size': 15, opacity: 0 });

    function reset() {
      cands.forEach(o => { o.c.setAttribute('fill', '#fff'); o.c.setAttribute('stroke', C.blue); o.c.setAttribute('r', 13); o.s.setAttribute('opacity', 0); });
      best.setAttribute('opacity', 0);
      cursor.setAttribute('opacity', 0);
    }
    function run() {
      reset();
      let top = null;
      scores.forEach((_, k) => setTimeout(() => {
        cursor.setAttribute('x', colX(k) - 48);
        cursor.setAttribute('opacity', 1);
        cands.filter(o => o.k === k).forEach(o => {
          o.s.setAttribute('opacity', 1);
          o.c.setAttribute('fill', C.blueSoft);
          if (!top || o.v > top.v) {
            if (top) { top.c.setAttribute('fill', C.blueSoft); top.c.setAttribute('stroke', C.blue); top.c.setAttribute('r', 13); }
            top = o;
            o.c.setAttribute('fill', C.red); o.c.setAttribute('stroke', C.red); o.c.setAttribute('r', 16);
          }
        });
        caption('anim-stitch-cap', `\\text{scanning } k = ${k + 1}\\qquad \\text{best so far: } k = ${top.k + 1},\\ ` +
          `G(s, s_k, k) + \\gamma^{k} V(s_k) = ${top.v.toFixed(2)}`);
      }, 900 * k));
      setTimeout(() => {
        const kb = top.k + 1;
        cursor.setAttribute('opacity', 0);
        best.setAttribute('x', top.c.getAttribute('cx'));
        best.setAttribute('y', +top.c.getAttribute('cy') + 36);
        best.setAttribute('opacity', 1);
        caption('anim-stitch-cap', `(k, s_k) \\sim \\pi_{\\text{stitch}}(\\cdot \\mid s),\\ k = ${kb}:\\quad ` +
          `V^{\\pi}_{\\text{target}} = G(s, s_{${kb}}, ${kb}) + \\gamma^{${kb}}\\, V(s_{${kb}})`);
      }, 900 * scores.length);
    }
    run();
    setInterval(run, 900 * scores.length + 3000);
  }

  const METHODS = ['IQL', 'FQL', 'IFQL', 'FQL-n', 'IFQL-n', 'QC', 'VALUE-FLOWS', 'FLOQ', 'VAST'];
  const CATS = [['One-step TD', 3], ['n-step TD', 2], ['Chunking', 1], ['Generative value', 2], ['Stitching', 1]];
  const CAT_COLOR = ['#D5D5D5', '#B9D3EA', '#8DB4D9', '#4F86B5', '#E8445F'];
  const methodColor = METHODS.map((_, i) => CAT_COLOR[i < 3 ? 0 : i < 5 ? 1 : i < 6 ? 2 : i < 8 ? 3 : 4]);
  const ROWS = [
    ['scene', 'Basic', [28, 57, 46, 42, 32, 60, 60, 57, 61], [null, 7, 12, 8, 2, 1, 5, 5, 3]],
    ['cube-single', 'Basic', [83, 97, 87, 97, 82, 93, 95, 96, 98], [null, 4, 5, 3, 6, 5, 4, 5, 2]],
    ['cube-double', 'Basic', [7, 30, 12, 4, 4, 67, 61, 49, 75], [null, 8, 5, 3, 3, 7, 11, 16, 6]],
    ['puzzle-4x4', 'Basic', [7, 15, 27, 27, 12, 29, 21, 31, 46], [null, 5, 8, 6, 4, 10, 8, 9, 10]],
    ['antmaze-large', 'Basic', [53, 82, 32, 80, 27, 6, 48, 93, 81], [null, 7, 17, 9, 7, 5, 15, 6, 7]],
    ['antmaze-giant', 'Basic', [4, 11, 0, 37, 0, 0, 3, 29, 20], [null, 16, 2, 16, 0, 0, 3, 24, 6]],
    ['Overall (Basic)', 'Basic', [30, 49, 34, 48, 26, 43, 48, 59, 64], [null, 9, 10, 9, 4, 6, 9, 13, 6], true],
    ['cube-triple', 'Hard', [null, 1, 1, 0, 0, 4, 13, 4, 17], [null, 4, 1, 0, 0, 5, 10, 5, 8]],
    ['cube-quadruple', 'Hard', [null, 0, 0, 0, 0, 0, 0, 0, 1], [null, 0, 0, 0, 0, 0, 0, 0, 1]],
    ['puzzle-4x5', 'Hard', [null, 15, 16, 18, 14, 20, 12, 13, 20], [null, 3, 7, 1, 4, 0, 4, 3, 0]],
    ['puzzle-4x6', 'Hard', [null, 4, 0, 4, 3, 4, 3, 3, 8], [null, 5, 0, 9, 5, 4, 6, 3, 17]],
    ['Overall (Hard)', 'Hard', [null, 5, 4, 6, 4, 7, 7, 5, 12], [null, 4, 4, 5, 3, 3, 6, 3, 9], true]
  ];

  function results() {
    const canvas = document.getElementById('result-chart');
    const btnBox = document.getElementById('result-buttons');
    if (!canvas || !btnBox || typeof Chart === 'undefined') return;
    const chart = new Chart(canvas, {
      type: 'bar',
      data: { labels: METHODS, datasets: [{ data: [], backgroundColor: methodColor, borderRadius: 4 }] },
      options: {
        indexAxis: 'y', maintainAspectRatio: false, animation: { duration: 700 },
        plugins: {
          legend: { display: false },
          title: { display: true, font: { size: 16, family: 'Google Sans' } },
          tooltip: { callbacks: { label: c => {
            const r = ROWS[chart.$row], s = r[3][c.dataIndex], v = r[2][c.dataIndex];
            return v === null ? 'not reported' : `success rate ${v}` + (s === null ? '' : ` ± ${s}`);
          } } }
        },
        scales: { x: { min: 0, title: { display: true, text: 'Success rate (%)' } },
          y: { ticks: { font: c => ({ weight: c.index === 8 ? 'bold' : 'normal', size: 13 }) } } }
      }
    });
    function show(idx) {
      chart.$row = idx;
      const r = ROWS[idx];
      chart.data.datasets[0].data = r[2];
      chart.options.plugins.title.text = r[0] + (r[4] ? '' : `  (${r[1]})`);
      chart.options.scales.x.max = Math.max(20, Math.ceil(Math.max(...r[2].filter(v => v !== null)) / 10) * 10 + 5);
      chart.update();
      btnBox.querySelectorAll('button').forEach(b => b.classList.toggle('is-active', +b.dataset.idx === idx));
    }
    ['Basic', 'Hard'].forEach(g => {
      const lab = document.createElement('span');
      lab.className = 'group-label'; lab.textContent = g + ':';
      btnBox.appendChild(lab);
      ROWS.forEach((r, idx) => {
        if (r[1] !== g) return;
        const b = document.createElement('button');
        b.textContent = r[4] ? 'Overall' : r[0];
        b.dataset.idx = idx;
        b.onclick = () => show(idx);
        btnBox.appendChild(b);
      });
    });
    show(ROWS.length - 1);
  }

  function table() {
    const tb = document.getElementById('result-table');
    if (!tb) return;
    const grpStart = new Set();
    CATS.reduce((at, [, s], i) => { if (i) grpStart.add(at); return at + s; }, 0);
    const grp = i => grpStart.has(i) ? ' class="grp"' : '';
    let h = '<thead><tr class="cat"><th></th>' + CATS.map(([n, s], i) => {
      const at = CATS.slice(0, i).reduce((a, c) => a + c[1], 0);
      return `<th colspan="${s}"${grp(at)}><span class="cat-label">${n}</span></th>`;
    }).join('') + '</tr><tr class="head"><th>Domain (5 tasks each)</th>' +
      METHODS.map((m, i) => `<th${grp(i)}>${m}</th>`).join('') + '</tr></thead><tbody>';
    ROWS.forEach(r => {
      const best = Math.max(...r[2].filter(v => v !== null));
      const rowCls = [r[4] ? 'overall' : '', r[0] === 'cube-triple' ? 'section-split' : ''].filter(Boolean).join(' ');
      h += `<tr${rowCls ? ` class="${rowCls}"` : ''}><td>${r[0]}</td>` + r[2].map((v, i) => {
        const cls = [v === best ? 'best' : '', r[4] && i === METHODS.length - 1 ? 'ours' : '', grpStart.has(i) ? 'grp' : ''].filter(Boolean).join(' ');
        const std = r[3][i] === null ? '' : ` <span class="std">±${r[3][i]}</span>`;
        return `<td${cls ? ` class="${cls}"` : ''}>${v === null ? '–' : v + std}</td>`;
      }).join('') + '</tr>';
    });
    tb.innerHTML = h + '</tbody>';
  }

  function fadeIn() {
    const items = document.querySelectorAll('.fade-in');
    if (!('IntersectionObserver' in window)) { items.forEach(e => e.classList.add('is-visible')); return; }
    const io = new IntersectionObserver(es => es.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add('is-visible'); io.unobserve(e.target); }
    }), { threshold: 0.12 });
    items.forEach(e => io.observe(e));
  }

  function bibCopy() {
    const b = document.getElementById('bib-copy');
    if (!b) return;
    b.onclick = () => navigator.clipboard.writeText(document.getElementById('bib-text').innerText.trim())
      .then(() => { b.textContent = 'Copied!'; setTimeout(() => (b.textContent = 'Copy'), 1500); });
  }

  document.addEventListener('DOMContentLoaded', () => {
    animEndpoint(); animStitch(); results(); table(); fadeIn(); bibCopy();
    if (window.renderMathInElement) {
      renderMathInElement(document.body, { delimiters: [
        { left: '$$', right: '$$', display: true }, { left: '\\(', right: '\\)', display: false }] });
    }
  });
})();
