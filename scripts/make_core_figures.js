const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(ROOT, 'results', 'figures');
fs.mkdirSync(OUT, { recursive: true });

const C = {
  ink: '#152238',
  muted: '#5E6B7A',
  light: '#F4F7FA',
  grid: '#D7DEE7',
  static: '#7B8794',
  dynamic: '#177E89',
  response: '#4C78A8',
  cka: '#F28E2B',
  rsa: '#8F63B8',
  trajectory: '#177E89',
  red: '#C84C4C',
  green: '#2E8B57',
  white: '#FFFFFF',
};

const esc = (value) => String(value)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');

function wrapText(text, maxChars) {
  const words = text.split(/\s+/);
  const lines = [];
  let line = '';
  for (const word of words) {
    if (!line || line.length + 1 + word.length <= maxChars) line += `${line ? ' ' : ''}${word}`;
    else { lines.push(line); line = word; }
  }
  if (line) lines.push(line);
  return lines;
}

function textLines(x, y, lines, opts = {}) {
  const { size = 28, weight = 400, fill = C.ink, anchor = 'start', gap = 1.22, cls = '' } = opts;
  return `<text x="${x}" y="${y}" text-anchor="${anchor}" font-size="${size}" font-weight="${weight}" fill="${fill}" class="${cls}">${lines.map((line, i) => `<tspan x="${x}" dy="${i === 0 ? 0 : size * gap}">${esc(line)}</tspan>`).join('')}</text>`;
}

function svgDoc(width, height, title, body) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">
  <title id="title">${esc(title)}</title>
  <desc id="desc">${esc(title)}</desc>
  <defs>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="4" stdDeviation="5" flood-color="#152238" flood-opacity="0.10"/></filter>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="${C.muted}"/></marker>
    <pattern id="hatch" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="10" stroke="${C.dynamic}" stroke-width="3" opacity="0.45"/></pattern>
  </defs>
  <rect width="100%" height="100%" fill="${C.white}"/>
  <style>
    text { font-family: Arial, Helvetica, sans-serif; }
    .axis { font-size: 24px; fill: ${C.muted}; }
    .value { font-size: 24px; font-weight: 700; fill: ${C.ink}; }
  </style>
  ${body}
</svg>`;
}

async function saveFigure(name, svg, width) {
  const svgPath = path.join(OUT, `${name}.svg`);
  const pngPath = path.join(OUT, `${name}.png`);
  fs.writeFileSync(svgPath, svg, 'utf8');
  await sharp(Buffer.from(svg)).resize({ width }).png({ compressionLevel: 9, quality: 100 }).toFile(pngPath);
}

function workflowFigure() {
  const W = 2400, H = 1450;
  const card = (x, y, w, h, title, lines, accent, label = '') => {
    let out = `<g filter="url(#shadow)"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="22" fill="${C.white}" stroke="${C.grid}" stroke-width="2"/><rect x="${x}" y="${y}" width="10" height="${h}" rx="5" fill="${accent}"/></g>`;
    if (label) out += `<rect x="${x + 30}" y="${y + 25}" width="${Math.max(90, label.length * 15)}" height="38" rx="19" fill="${accent}" opacity="0.12"/><text x="${x + 45}" y="${y + 52}" font-size="21" font-weight="700" fill="${accent}">${esc(label)}</text>`;
    out += textLines(x + 32, y + (label ? 105 : 62), [title], { size: 31, weight: 700 });
    out += textLines(x + 32, y + (label ? 150 : 107), lines, { size: 23, fill: C.muted, gap: 1.3 });
    return out;
  };
  const arrow = (x1, y1, x2, y2) => `<path d="M ${x1} ${y1} L ${x2} ${y2}" stroke="${C.muted}" stroke-width="4" fill="none" marker-end="url(#arrow)"/>`;
  let b = textLines(120, 105, ['Experimental design: testing temporal computation beyond response accuracy'], { size: 48, weight: 700 });
  b += textLines(120, 157, ['All comparison models share data, neurons, splits, behavioral inputs, readout family, and evaluation interval.'], { size: 25, fill: C.muted });

  b += card(110, 260, 420, 270, 'Dynamic Sensorium 2023', ['Natural movies', 'Mouse V1 population responses', 'Synchronized behavior'], C.response, 'DATA');
  b += card(680, 225, 455, 245, 'Static model', ['Frame-wise 2D core', '2,814,015 parameters', 'No learned temporal context'], C.static, 'MODEL A');
  b += card(680, 555, 455, 245, 'Reduced Dynamic model', ['Factorized 3D core', '2,862,063 parameters', 'Learned temporal context'], C.dynamic, 'MODEL B');
  b += arrow(530, 375, 680, 347);
  b += arrow(530, 430, 680, 675);
  b += `<path d="M 1135 347 C 1260 347, 1240 475, 1350 475" stroke="${C.muted}" stroke-width="4" fill="none" marker-end="url(#arrow)"/>`;
  b += `<path d="M 1135 675 C 1260 675, 1240 525, 1350 525" stroke="${C.muted}" stroke-width="4" fill="none" marker-end="url(#arrow)"/>`;
  b += card(1350, 365, 430, 270, 'Frozen neural predictions', ['Same oracle movies', 'Matched neurons and timestamps', 'No evaluation-driven fitting'], C.cka, 'OUTPUT');

  b += card(1885, 210, 390, 185, 'Response', ['Per-neuron correlation', 'Predictive accuracy'], C.response, 'LEVEL 1');
  b += card(1885, 455, 390, 185, 'RSA / CKA', ['Output-space geometry', 'Predicted neural responses'], C.rsa, 'LEVEL 2');
  b += card(1885, 700, 390, 205, 'Neural trajectory', ['Position and RMSE', 'Velocity, speed, acceleration'], C.trajectory, 'LEVEL 3');
  b += arrow(1780, 470, 1885, 303);
  b += arrow(1780, 500, 1885, 548);
  b += arrow(1780, 530, 1885, 800);

  b += card(885, 995, 520, 255, 'Brain-defined GPFA', ['Fit only on neural training data', 'Frozen before model evaluation', 'No model-specific latent alignment'], C.green, 'LATENT SPACE');
  b += arrow(530, 505, 885, 1080);
  b += arrow(1405, 1120, 1885, 850);

  b += `<rect x="120" y="1300" width="2150" height="92" rx="18" fill="${C.light}"/>`;
  b += textLines(150, 1338, ['Key controls'], { size: 23, weight: 700, fill: C.ink });
  const controls = ['Parameter matching', 'Reliability + null tests', 'Response matching', 'Temporal-history ablation'];
  controls.forEach((t, i) => {
    const x = 420 + i * 440;
    b += `<circle cx="${x}" cy="1345" r="9" fill="${[C.dynamic, C.green, C.response, C.red][i]}"/>`;
    b += textLines(x + 20, 1353, [t], { size: 22, fill: C.ink });
  });
  return svgDoc(W, H, 'Experimental design for Static-Dynamic neural trajectory evaluation', b);
}

function comparisonFigure() {
  const W = 2200, H = 1350;
  const metrics = [
    { family: 'Response', label: 'Response correlation', value: 0.04469, lo: 0.02335, hi: 0.06484, color: C.response },
    { family: 'Representation', label: 'Temporal CKA', value: 0.08543, lo: 0.03039, hi: 0.16555, color: C.cka },
    { family: 'Representation', label: 'Temporal RSA', value: 0.08923, lo: 0.03081, hi: 0.19298, color: C.rsa },
    { family: 'Trajectory', label: 'GPFA position', value: 0.22380, lo: 0.08833, hi: 0.45126, color: C.trajectory },
    { family: 'Trajectory', label: 'GPFA velocity direction', value: 0.19473, lo: 0.13357, hi: 0.24861, color: C.trajectory },
    { family: 'Trajectory', label: 'GPFA speed profile', value: 0.01084, lo: -0.05126, hi: 0.08586, color: C.trajectory },
    { family: 'Trajectory', label: 'GPFA acceleration direction', value: 0.26793, lo: 0.18217, hi: 0.33953, color: C.trajectory },
  ];
  let b = textLines(110, 100, ['Where does the Dynamic model outperform the Static model?'], { size: 48, weight: 700 });
  b += textLines(110, 154, ['Paired condition comparison for the parameter-matched models; points show Dynamic - Static and bars show 95% bootstrap intervals.'], { size: 25, fill: C.muted });

  const chartLeft = 800, chartRight = 2070, chartTop = 270, chartBottom = 1095;
  const xMin = -0.10, xMax = 0.50;
  const sx = v => chartLeft + (chartRight - chartLeft) * (v - xMin) / (xMax - xMin);
  const rowY = [335, 485, 585, 760, 860, 960, 1060];
  const familyBands = [
    { label: 'RESPONSE', y: 280, h: 110, color: C.response },
    { label: 'OUTPUT-SPACE GEOMETRY', y: 425, h: 220, color: C.rsa },
    { label: 'TRAJECTORY', y: 690, h: 430, color: C.trajectory },
  ];
  familyBands.forEach(g => {
    b += `<rect x="100" y="${g.y}" width="1970" height="${g.h}" rx="18" fill="${g.color}" opacity="0.055"/>`;
    b += `<text x="135" y="${g.y + 38}" font-size="20" font-weight="700" fill="${g.color}">${g.label}</text>`;
  });
  for (let t = -0.1; t <= 0.5001; t += 0.1) {
    const x = sx(Number(t.toFixed(1)));
    b += `<line x1="${x}" y1="${chartTop}" x2="${x}" y2="${chartBottom}" stroke="${Math.abs(t) < 0.001 ? C.ink : C.grid}" stroke-width="${Math.abs(t) < 0.001 ? 3 : 1.5}"/>`;
    b += `<text x="${x}" y="${chartBottom + 52}" text-anchor="middle" font-size="22" fill="${C.muted}">${t > 0 ? '+' : ''}${t.toFixed(1)}</text>`;
  }
  metrics.forEach((m, i) => {
    const y = rowY[i];
    b += `<text x="170" y="${y + 9}" font-size="26" font-weight="${m.family === 'Trajectory' ? 500 : 700}" fill="${C.ink}">${esc(m.label)}</text>`;
    b += `<line x1="${sx(m.lo)}" y1="${y}" x2="${sx(m.hi)}" y2="${y}" stroke="${m.color}" stroke-width="7" stroke-linecap="round"/>`;
    b += `<line x1="${sx(m.lo)}" y1="${y - 15}" x2="${sx(m.lo)}" y2="${y + 15}" stroke="${m.color}" stroke-width="4"/>`;
    b += `<line x1="${sx(m.hi)}" y1="${y - 15}" x2="${sx(m.hi)}" y2="${y + 15}" stroke="${m.color}" stroke-width="4"/>`;
    b += `<circle cx="${sx(m.value)}" cy="${y}" r="13" fill="${C.white}" stroke="${m.color}" stroke-width="7"/>`;
    const labelX = Math.min(sx(m.hi) + 24, chartRight - 65);
    b += `<text x="${labelX}" y="${y + 8}" font-size="22" font-weight="700" fill="${m.color}">${m.value >= 0 ? '+' : ''}${m.value.toFixed(3)}</text>`;
  });
  b += `<text x="${(chartLeft + chartRight)/2}" y="${chartBottom + 105}" text-anchor="middle" font-size="27" font-weight="700" fill="${C.ink}">Dynamic - Static agreement with neural data</text>`;
  b += `<text x="${sx(0) - 18}" y="230" text-anchor="end" font-size="21" fill="${C.muted}">Static better</text>`;
  b += `<text x="${sx(0) + 18}" y="230" font-size="21" fill="${C.muted}">Dynamic better</text>`;
  b += textLines(110, 1285, ['Response and time-aware RSA/CKA detect gains. Direction-sensitive trajectory metrics show additional separation; speed-profile evidence remains weak.'], { size: 25, fill: C.muted });
  return svgDoc(W, H, 'Static-Dynamic comparison across response, RSA, CKA, and trajectory metrics', b);
}

function ablationFigure() {
  const W = 2200, H = 1350;
  const xVals = [0, 25, 50, 75, 100];
  const raw = {
    'Position': [0.726219, 0.552833, 0.186907, -0.129463, -0.194901],
    'Velocity': [0.498462, 0.452724, 0.304845, 0.065920, -0.067456],
    'Speed': [0.537538, 0.512594, 0.339678, 0.032340, -0.034254],
    'Acceleration': [0.452488, 0.410768, 0.286180, 0.091460, -0.029123],
  };
  const colors = { Position: C.trajectory, Velocity: '#2E8B57', Speed: '#C84C4C', Acceleration: '#8F63B8' };
  const dashes = { Position: '', Velocity: '12 8', Speed: '3 8', Acceleration: '18 6 3 6' };
  const normalized = {};
  for (const [name, vals] of Object.entries(raw)) normalized[name] = vals.map(v => v / vals[0]);
  const x0 = 185, x1 = 2070, y0 = 270, y1 = 1090;
  const sx = v => x0 + (x1 - x0) * v / 100;
  const yMin = -0.55, yMax = 1.08;
  const sy = v => y1 - (y1 - y0) * (v - yMin) / (yMax - yMin);
  let b = textLines(110, 100, ['Temporal-history ablation reveals graded degradation'], { size: 48, weight: 700 });
  b += textLines(110, 154, ['Off-center temporal-kernel weights are scaled from fully retained to fully removed; trajectory similarity is normalized to the intact Dynamic value.'], { size: 25, fill: C.muted });
  b += `<rect x="${x0}" y="${y0}" width="${x1-x0}" height="${y1-y0}" fill="${C.white}" stroke="${C.grid}" stroke-width="2"/>`;
  [-0.5, 0, 0.5, 1.0].forEach(v => {
    const y = sy(v);
    b += `<line x1="${x0}" y1="${y}" x2="${x1}" y2="${y}" stroke="${v === 0 ? C.muted : C.grid}" stroke-width="${v === 0 ? 2 : 1.5}"/>`;
    b += `<text x="${x0 - 24}" y="${y + 8}" text-anchor="end" font-size="22" fill="${C.muted}">${v.toFixed(1)}</text>`;
  });
  xVals.forEach(v => {
    const x = sx(v);
    b += `<line x1="${x}" y1="${y0}" x2="${x}" y2="${y1}" stroke="${C.grid}" stroke-width="1.2"/>`;
    b += `<text x="${x}" y="${y1 + 46}" text-anchor="middle" font-size="22" fill="${C.muted}">${v}%</text>`;
  });
  for (const [name, vals] of Object.entries(normalized)) {
    const points = vals.map((v, i) => `${sx(xVals[i])},${sy(v)}`).join(' ');
    b += `<polyline points="${points}" fill="none" stroke="${colors[name]}" stroke-width="${name === 'Position' ? 7 : 5}" stroke-linecap="round" stroke-linejoin="round" ${dashes[name] ? `stroke-dasharray="${dashes[name]}"` : ''}/>`;
    vals.forEach((v, i) => b += `<circle cx="${sx(xVals[i])}" cy="${sy(v)}" r="${name === 'Position' ? 9 : 7}" fill="${C.white}" stroke="${colors[name]}" stroke-width="4"/>`);
  }
  b += `<text x="${(x0+x1)/2}" y="${y1 + 105}" text-anchor="middle" font-size="26" font-weight="700" fill="${C.ink}">Temporal-history ablation severity</text>`;
  b += `<text x="62" y="${(y0+y1)/2}" transform="rotate(-90 62 ${(y0+y1)/2})" text-anchor="middle" font-size="26" font-weight="700" fill="${C.ink}">Normalized trajectory similarity (intact Dynamic = 1)</text>`;

  const names = Object.keys(raw);
  names.forEach((name, i) => {
    const x = 395 + i * 430, y = 1235;
    b += `<line x1="${x}" y1="${y}" x2="${x + 64}" y2="${y}" stroke="${colors[name]}" stroke-width="5" ${dashes[name] ? `stroke-dasharray="${dashes[name]}"` : ''}/><circle cx="${x+32}" cy="${y}" r="6" fill="${C.white}" stroke="${colors[name]}" stroke-width="3"/>`;
    b += `<text x="${x + 82}" y="${y + 8}" font-size="22" fill="${C.ink}">${name}</text>`;
  });
  b += `<rect x="1440" y="300" width="570" height="95" rx="16" fill="${C.light}"/>`;
  b += textLines(1470, 340, ['Trajectory similarity metrics show strict', 'monotonic degradation (Spearman rho = -1).'], { size: 22, weight: 700, fill: C.ink, gap: 1.25 });
  return svgDoc(W, H, 'Temporal ablation degradation curves', b);
}

(async () => {
  await saveFigure('figure-1-experimental-workflow', workflowFigure(), 2400);
  await saveFigure('figure-2-static-dynamic-comparison', comparisonFigure(), 2200);
  await saveFigure('figure-3-temporal-ablation', ablationFigure(), 2200);
  console.log(`Wrote figures to ${OUT}`);
})().catch((error) => { console.error(error); process.exit(1); });
