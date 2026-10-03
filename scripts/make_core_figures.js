const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const ROOT = path.resolve(__dirname, '..');
const outputArg = process.argv.find(value => value.startsWith('--output-dir='));
const OUT = outputArg ? path.resolve(outputArg.slice('--output-dir='.length)) : path.join(ROOT, 'results', 'figures');
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
  let b = textLines(120, 105, ['Experimental design: comparing Static and Dynamic beyond response accuracy'], { size: 48, weight: 700 });
  b += textLines(120, 157, ['Models share data and evaluation contracts and have close total counts, while their core architectures and temporal access remain different.'], { size: 25, fill: C.muted });

  b += card(110, 260, 420, 270, 'Dynamic Sensorium 2023', ['Natural movies', 'Mouse V1 population responses', 'Synchronized behavior'], C.response, 'DATA');
  b += card(680, 225, 455, 245, 'Static model', ['Frame-wise 2D core', '2,814,015 parameters', 'No learned temporal context'], C.static, 'MODEL A');
  b += card(680, 555, 455, 245, 'Reduced Dynamic model', ['Factorized 3D core', '2,862,063 parameters', 'Learned temporal context'], C.dynamic, 'MODEL B');
  b += arrow(530, 375, 680, 347);
  b += arrow(530, 430, 680, 675);
  b += `<path d="M 1135 347 C 1260 347, 1240 475, 1350 475" stroke="${C.muted}" stroke-width="4" fill="none" marker-end="url(#arrow)"/>`;
  b += `<path d="M 1135 675 C 1260 675, 1240 525, 1350 525" stroke="${C.muted}" stroke-width="4" fill="none" marker-end="url(#arrow)"/>`;
  b += card(1350, 365, 430, 270, 'Frozen neural predictions', ['Same oracle movies', 'Matched neurons and timestamps', 'Frozen encoding checkpoints'], C.cka, 'OUTPUT');

  b += card(1885, 210, 390, 185, 'Response', ['Per-neuron correlation', 'Predictive accuracy'], C.response, 'LEVEL 1');
  b += card(1885, 455, 390, 185, 'RSA / CKA', ['Output-space geometry', 'Predicted neural responses'], C.rsa, 'LEVEL 2');
  b += card(1885, 700, 390, 205, 'Neural trajectory', ['Position and RMSE', 'Velocity, speed, acceleration'], C.trajectory, 'LEVEL 3');
  b += arrow(1780, 470, 1885, 303);
  b += arrow(1780, 500, 1885, 548);
  b += arrow(1780, 530, 1885, 800);

  b += card(885, 995, 520, 255, 'Neural-data-defined GPFA', ['Fit only on neural training data', 'Frozen before model evaluation', 'No model-specific latent alignment', 'Offline posterior uses later observations'], C.green, 'LATENT SPACE');
  b += `<path d="M 530 505 L 620 900 L 885 1080" stroke="${C.muted}" stroke-width="4" fill="none" marker-end="url(#arrow)"/>`;
  b += arrow(1405, 1120, 1885, 850);

  b += `<rect x="120" y="1300" width="2150" height="92" rx="18" fill="${C.light}"/>`;
  b += textLines(150, 1338, ['Key controls'], { size: 23, weight: 700, fill: C.ink });
  const controls = ['Total-parameter matching', 'Reliability + null tests', 'Response-score-matched output perturbation', 'Temporal-history ablation'];
  controls.forEach((t, i) => {
    const x = 420 + i * 440;
    b += `<circle cx="${x}" cy="1345" r="9" fill="${[C.dynamic, C.green, C.response, C.red][i]}"/>`;
    b += textLines(x + 20, 1353, [t], { size: i === 2 ? 18 : 22, fill: C.ink });
  });
  return svgDoc(W, H, 'Experimental design for Static-Dynamic neural trajectory evaluation', b);
}

function comparisonFigure() {
  const W = 2200, H = 1350;
  // Display the saved statistics, without recomputing scientific comparisons.
  const readRows = (name) => {
    const lines = fs.readFileSync(path.join(ROOT, 'results', 'tables', '04_model_comparison', name), 'utf8').trim().split(/\r?\n/);
    const fields = lines.shift().split(',');
    return Object.fromEntries(lines.map(line => {
      const values = line.split(',');
      const row = Object.fromEntries(fields.map((field, i) => [field, values[i]]));
      return [row.metric, row];
    }));
  };
  const conventional = readRows('q2_condition_conventional_bootstrap.csv');
  const trajectory = readRows('gpfa_model_paired_bootstrap.csv');
  const saved = (rows, metric, family, label, color, valueField) => {
    const row = rows[metric];
    if (!row) throw new Error(`Missing saved figure metric: ${metric}`);
    // Retain the original five-decimal plotting convention.
    const value = key => Number(Number(row[key]).toFixed(5));
    return { family, label, value: value(valueField), lo: value('ci_low'), hi: value('ci_high'), color };
  };
  const metrics = [
    saved(conventional, 'response_r', 'Response', 'Condition-mean response r', C.response, 'dynamic_minus_static'),
    saved(conventional, 'temporal_cka', 'Representation', 'Temporal CKA', C.cka, 'dynamic_minus_static'),
    saved(conventional, 'temporal_rsa', 'Representation', 'Temporal RSA', C.rsa, 'dynamic_minus_static'),
    saved(trajectory, 'position_correlation', 'Trajectory', 'GPFA position', C.trajectory, 'mean_oriented_advantage'),
    saved(trajectory, 'velocity_direction_cosine', 'Trajectory', 'GPFA velocity direction', C.trajectory, 'mean_oriented_advantage'),
    saved(trajectory, 'speed_profile_correlation', 'Trajectory', 'GPFA speed profile', C.trajectory, 'mean_oriented_advantage'),
    saved(trajectory, 'acceleration_direction_cosine', 'Trajectory', 'GPFA acceleration direction', C.trajectory, 'mean_oriented_advantage'),
  ];
  let b = textLines(110, 100, ['Where does the Dynamic model outperform the Static model?'], { size: 48, weight: 700 });
  b += textLines(110, 154, ['One-session, six-condition pilot; 512 neurons. Metric families are shown separately.'], { size: 25, fill: C.muted });
  b += textLines(110, 196, ['Do not compare absolute effect sizes across metric families.'], { size: 27, weight: 700, fill: C.ink });

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
  b += textLines(110, 1240, ['Bars: 95% bootstrap intervals. Trajectory points: bootstrap means; other points: observed mean differences.'], { size: 24, fill: C.muted });
  b += textLines(110, 1285, ['Response and time-aware RSA/CKA detect gains. Trajectory direction also differs; speed-profile evidence remains inconclusive.'], { size: 25, fill: C.muted });
  return svgDoc(W, H, 'Static-Dynamic comparison across response, RSA, CKA, and trajectory metrics', b);
}

function ablationFigure() {
  const W = 2200, H = 1350;
  // Read saved point estimates; retain the original six-decimal display convention.
  const lines = fs.readFileSync(path.join(ROOT, 'results', 'tables', '04_model_comparison', 'q5_temporal_ablation_curve.csv'), 'utf8').trim().split(/\r?\n/);
  const fields = lines.shift().split(',');
  const rows = lines.map(line => Object.fromEntries(line.split(',').map((value, i) => [fields[i], Number(value)])));
  const xVals = rows.map(row => row.severity * 100);
  const columns = { Position: 'position_correlation', 'Velocity direction': 'velocity_direction_cosine', 'Speed profile': 'speed_profile_correlation', 'Acceleration direction': 'acceleration_direction_cosine' };
  const raw = Object.fromEntries(Object.entries(columns).map(([label, metric]) => [label, rows.map(row => Number(row[metric].toFixed(6)))]));
  const colors = { Position: C.trajectory, 'Velocity direction': '#2E8B57', 'Speed profile': '#C84C4C', 'Acceleration direction': '#8F63B8' };
  const dashes = { Position: '', 'Velocity direction': '12 8', 'Speed profile': '3 8', 'Acceleration direction': '18 6 3 6' };
  const normalized = {};
  for (const [name, vals] of Object.entries(raw)) normalized[name] = vals.map(v => v / vals[0]);
  const x0 = 185, x1 = 2070, y0 = 270, y1 = 1090;
  const sx = v => x0 + (x1 - x0) * v / 100;
  const yMin = -0.55, yMax = 1.08;
  const sy = v => y1 - (y1 - y0) * (v - yMin) / (yMax - yMin);
  let b = textLines(110, 100, ['Graded trajectory degradation under temporal-weight ablation'], { size: 48, weight: 700 });
  b += textLines(110, 154, ['Observed dose-response after scaling off-center temporal-kernel weights; no magnitude-matched non-temporal damage control is included.'], { size: 25, fill: C.muted });
  b += textLines(110, 196, ['One session, six movies, 512 neurons; offline full-window GPFA inference.'], { size: 25, fill: C.muted });
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
  b += `<text x="${(x0+x1)/2}" y="${y1 + 105}" text-anchor="middle" font-size="26" font-weight="700" fill="${C.ink}">Off-center temporal-weight ablation severity</text>`;
  b += `<text x="62" y="${(y0+y1)/2}" transform="rotate(-90 62 ${(y0+y1)/2})" text-anchor="middle" font-size="26" font-weight="700" fill="${C.ink}">Normalized trajectory similarity (intact Dynamic = 1)</text>`;

  const names = Object.keys(raw);
  names.forEach((name, i) => {
    const x = 395 + i * 430, y = 1235;
    b += `<line x1="${x}" y1="${y}" x2="${x + 64}" y2="${y}" stroke="${colors[name]}" stroke-width="5" ${dashes[name] ? `stroke-dasharray="${dashes[name]}"` : ''}/><circle cx="${x+32}" cy="${y}" r="6" fill="${C.white}" stroke="${colors[name]}" stroke-width="3"/>`;
    b += `<text x="${x + 82}" y="${y + 8}" font-size="22" fill="${C.ink}">${name}</text>`;
  });
  b += `<rect x="1440" y="300" width="570" height="95" rx="16" fill="${C.light}"/>`;
  b += textLines(1470, 340, ['The displayed curves are monotonic.', 'Temporal specificity remains untested.'], { size: 22, weight: 700, fill: C.ink, gap: 1.25 });
  return svgDoc(W, H, 'Graded trajectory degradation under temporal-weight ablation', b);
}

(async () => {
  const selected = process.argv.find(value => value.startsWith('--figure='))?.split('=')[1];
  if (!selected || selected === '1') await saveFigure('figure-1-experimental-workflow', workflowFigure(), 2400);
  if (!selected || selected === '2') await saveFigure('figure-2-static-dynamic-comparison', comparisonFigure(), 2200);
  if (!selected || selected === '3') await saveFigure('figure-3-temporal-ablation', ablationFigure(), 2200);
  console.log(`Wrote figures to ${OUT}`);
})().catch((error) => { console.error(error); process.exit(1); });
