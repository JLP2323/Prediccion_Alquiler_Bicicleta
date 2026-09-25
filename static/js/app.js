/**
 * VeloPredict Analytics – Frontend Logic
 * Handles: API fetching, chart rendering, simulator, UI state management
 */

// ─────────────────────────────────────────────
// State
// ─────────────────────────────────────────────
const charts = {};
let currentDayType = 'work';   // 'work' | 'weekend' | 'holiday'
let currentWeather = 1;         // 1-4
let appData = null;

// ─────────────────────────────────────────────
// Utility helpers
// ─────────────────────────────────────────────
const fmt = (n, dec = 0) => Number(n).toLocaleString('es-ES', { maximumFractionDigits: dec });

function setLoading(show) {
  document.getElementById('loadingOverlay').style.display = show ? 'flex' : 'none';
}

function updateHourLabel(val) {
  const h = parseInt(val, 10);
  const period = h < 12 ? 'AM' : 'PM';
  const h12 = h === 0 ? 12 : h > 12 ? h - 12 : h;
  document.getElementById('hrLabel').textContent = `${h12}:00 ${period}`;
}

function selectDayType(type) {
  currentDayType = type;
  ['work', 'weekend', 'holiday'].forEach(t => {
    document.getElementById(`dt-${t}`).classList.remove('active');
  });
  document.getElementById(`dt-${type}`).classList.add('active');
}

function selectWeather(wt) {
  currentWeather = wt;
  [1, 2, 3, 4].forEach(i => {
    document.getElementById(`wt-${i}`).classList.remove('active');
  });
  document.getElementById(`wt-${wt}`).classList.add('active');
}

function switchTab(tab) {
  ['hourly', 'weather', 'seasonal'].forEach(t => {
    document.getElementById(`tab-${t}`).classList.remove('active');
    document.getElementById(`panel-${t}`).classList.add('hidden');
  });
  document.getElementById(`tab-${tab}`).classList.add('active');
  document.getElementById(`panel-${tab}`).classList.remove('hidden');
}

function animateNumber(el, target, duration = 800, suffix = '') {
  const start = 0;
  const step = timestamp => {
    if (!step.startTime) step.startTime = timestamp;
    const progress = Math.min((timestamp - step.startTime) / duration, 1);
    const ease = progress < 0.5 ? 2 * progress * progress : -1 + (4 - 2 * progress) * progress;
    el.textContent = fmt(Math.floor(ease * target)) + suffix;
    if (progress < 1) requestAnimationFrame(step);
    else el.textContent = fmt(target) + suffix;
  };
  requestAnimationFrame(step);
}

// ─────────────────────────────────────────────
// Chart helpers
// ─────────────────────────────────────────────
const CHART_DEFAULTS = {
  font: { family: 'Inter, sans-serif' },
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: 'rgba(15,23,42,0.9)',
      padding: 10,
      cornerRadius: 8,
      titleFont: { size: 12, weight: '600' },
      bodyFont: { size: 12 },
    }
  },
  responsive: true,
  maintainAspectRatio: false,
};

function destroyChart(id) {
  if (charts[id]) { charts[id].destroy(); delete charts[id]; }
}

// Hourly line chart
function renderHourlyChart(data) {
  destroyChart('hourly');
  const labels = Array.from({ length: 24 }, (_, i) => {
    const p = i < 12 ? 'AM' : 'PM';
    const h = i === 0 ? 12 : i > 12 ? i - 12 : i;
    return `${h}${p}`;
  });
  const working = labels.map((_, i) => data.hourly_working[i] ?? 0);
  const weekend = labels.map((_, i) => data.hourly_weekend[i] ?? 0);

  charts['hourly'] = new Chart(document.getElementById('hourlyChart'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Laborable',
          data: working,
          borderColor: '#059669',
          backgroundColor: 'rgba(5,150,105,0.08)',
          borderWidth: 2.5,
          pointRadius: 3,
          pointHoverRadius: 6,
          pointBackgroundColor: '#059669',
          tension: 0.4,
          fill: true,
        },
        {
          label: 'Fin de semana',
          data: weekend,
          borderColor: '#38bdf8',
          backgroundColor: 'rgba(56,189,248,0.08)',
          borderWidth: 2.5,
          pointRadius: 3,
          pointHoverRadius: 6,
          pointBackgroundColor: '#38bdf8',
          tension: 0.4,
          fill: true,
        }
      ]
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: {
        ...CHART_DEFAULTS.plugins,
        legend: { display: true, position: 'top', labels: { usePointStyle: true, font: { family: 'Inter' }, boxWidth: 8 } },
        tooltip: {
          ...CHART_DEFAULTS.plugins.tooltip,
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${fmt(ctx.parsed.y, 1)} bicicletas`
          }
        }
      },
      scales: {
        x: { grid: { color: 'rgba(0,0,0,0.04)' }, ticks: { font: { size: 11, family: 'Inter' }, color: '#94a3b8', maxRotation: 0 } },
        y: { grid: { color: 'rgba(0,0,0,0.04)' }, ticks: { font: { size: 11, family: 'Inter' }, color: '#94a3b8' }, beginAtZero: true }
      },
      interaction: { mode: 'index', intersect: false },
    }
  });
}

// Weather bar chart
function renderWeatherChart(data) {
  destroyChart('weather');
  const labels = Object.keys(data.weather_impact);
  const values = Object.values(data.weather_impact);
  const colors = ['#34d399', '#60a5fa', '#818cf8', '#f87171'];

  charts['weather'] = new Chart(document.getElementById('weatherChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Demanda promedio',
        data: values,
        backgroundColor: colors.slice(0, labels.length),
        borderRadius: 10,
        borderSkipped: false,
      }]
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: {
        ...CHART_DEFAULTS.plugins,
        tooltip: {
          ...CHART_DEFAULTS.plugins.tooltip,
          callbacks: { label: ctx => ` Promedio: ${fmt(ctx.parsed.y, 1)} bicicletas` }
        }
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 12, family: 'Inter' }, color: '#64748b' } },
        y: { grid: { color: 'rgba(0,0,0,0.04)' }, ticks: { font: { size: 11, family: 'Inter' }, color: '#94a3b8' }, beginAtZero: true }
      },
    }
  });
}

// Monthly area chart
function renderMonthlyChart(data) {
  destroyChart('monthly');
  const months = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic'];
  const values = months.map((_, i) => data.monthly_avg[i + 1] ?? 0);

  charts['monthly'] = new Chart(document.getElementById('monthlyChart'), {
    type: 'line',
    data: {
      labels: months,
      datasets: [{
        label: 'Promedio mensual',
        data: values,
        borderColor: '#8b5cf6',
        backgroundColor: 'rgba(139,92,246,0.12)',
        borderWidth: 2.5,
        pointRadius: 4,
        pointHoverRadius: 7,
        pointBackgroundColor: '#8b5cf6',
        tension: 0.4,
        fill: true,
      }]
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: {
        ...CHART_DEFAULTS.plugins,
        legend: { display: false },
        tooltip: {
          ...CHART_DEFAULTS.plugins.tooltip,
          callbacks: { label: ctx => ` ${fmt(ctx.parsed.y, 1)} bicicletas` }
        }
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 11, family: 'Inter' }, color: '#94a3b8' } },
        y: { grid: { color: 'rgba(0,0,0,0.04)' }, ticks: { font: { size: 11, family: 'Inter' }, color: '#94a3b8' }, beginAtZero: true }
      },
    }
  });
}

// Seasonal doughnut
function renderSeasonChart(data) {
  destroyChart('season');
  const labels = Object.keys(data.seasonal_avg);
  const values = Object.values(data.seasonal_avg);
  const colors = ['#60a5fa', '#34d399', '#fbbf24', '#f97316'];

  charts['season'] = new Chart(document.getElementById('seasonChart'), {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors.slice(0, labels.length),
        borderWidth: 2,
        borderColor: '#fff',
        hoverOffset: 6,
      }]
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: {
        ...CHART_DEFAULTS.plugins,
        legend: { display: true, position: 'bottom', labels: { font: { family: 'Inter', size: 12 }, usePointStyle: true, padding: 12 } },
        tooltip: {
          ...CHART_DEFAULTS.plugins.tooltip,
          callbacks: { label: ctx => ` ${ctx.label}: ${fmt(ctx.parsed, 1)} bicicletas` }
        }
      },
      cutout: '62%',
    }
  });
}

// ─────────────────────────────────────────────
// Data loading
// ─────────────────────────────────────────────
async function loadData() {
  setLoading(true);
  try {
    const res = await fetch('/api/stats');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    appData = json;

    renderKPIs(json.stats);
    renderCharts(json.charts);
    updateStatusBadge(true, json.stats.model_metrics);

  } catch (err) {
    console.error('Error loading data:', err);
    updateStatusBadge(false);
  } finally {
    setLoading(false);
  }
}

function renderKPIs(stats) {
  // Total rentals
  const totalEl = document.getElementById('kpiTotal');
  totalEl.innerHTML = '';
  animateNumber(totalEl, stats.total_rentals);

  // Avg hourly
  const avgEl = document.getElementById('kpiAvg');
  avgEl.innerHTML = `<span>${fmt(stats.avg_hourly, 1)}</span>`;

  // User distribution
  const casualPct = stats.casual_pct;
  const regPct = stats.registered_pct;
  const casualPctEl = document.getElementById('kpiCasualPct');
  casualPctEl.innerHTML = `${casualPct}%`;
  document.getElementById('casualBar').style.width = `${casualPct}%`;
  document.getElementById('casualLbl').textContent = `Casual ${casualPct}%`;
  document.getElementById('regLbl').textContent = `Registrado ${regPct}%`;

  // Temperature
  document.getElementById('kpiTemp').innerHTML = `${stats.avg_temp_c}°C`;
  document.getElementById('kpiAtemp').innerHTML = `Sens. térmica: <span class="font-medium">${stats.avg_atemp_c}°C</span>`;

  // Model metrics
  if (stats.model_metrics) {
    document.getElementById('modelMetrics').classList.remove('hidden');
    document.getElementById('metR2').textContent = stats.model_metrics.R2;
    document.getElementById('metMAE').textContent = fmt(stats.model_metrics.MAE, 1);
    document.getElementById('metRMSE').textContent = fmt(stats.model_metrics.RMSE, 1);
  }
}

function renderCharts(charts_data) {
  renderHourlyChart(charts_data);
  renderWeatherChart(charts_data);
  renderMonthlyChart(charts_data);
  renderSeasonChart(charts_data);
}

function updateStatusBadge(ok, metrics = null) {
  const badge = document.getElementById('statusBadge');
  if (ok) {
    badge.innerHTML = `
      <span class="relative flex w-2 h-2">
        <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
        <span class="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
      </span>
      <span>Modelo Activo</span>`;
    badge.className = 'flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-50 text-emerald-700 text-xs font-medium border border-emerald-200';

    if (metrics) {
      const r2El = document.getElementById('r2Badge');
      r2El.classList.remove('hidden');
      r2El.classList.add('flex');
      document.getElementById('r2Value').textContent = metrics.R2;
    }
  } else {
    badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-red-400 inline-block"></span><span>Error al cargar</span>`;
    badge.className = 'flex items-center gap-2 px-3 py-1.5 rounded-full bg-red-50 text-red-600 text-xs font-medium border border-red-200';
  }
}

// ─────────────────────────────────────────────
// Prediction
// ─────────────────────────────────────────────
async function runPrediction() {
  const btn = document.getElementById('predictBtn');
  btn.disabled = true;
  btn.innerHTML = `
    <svg class="spinner w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
    </svg>
    Calculando…`;

  // Gather form values
  const hr = parseInt(document.getElementById('sliderHr').value, 10);
  const temp_c = parseFloat(document.getElementById('sliderTemp').value);
  const hum = parseFloat(document.getElementById('sliderHum').value);
  const windspeed_kmh = parseFloat(document.getElementById('sliderWind').value);
  const season = parseInt(document.getElementById('selSeason').value, 10);
  const mnth = parseInt(document.getElementById('selMonth').value, 10);

  // Day type mapping
  let workingday = 0, holiday = 0, weekday = 3;
  if (currentDayType === 'work') {
    workingday = 1; holiday = 0; weekday = 3; // Wednesday
  } else if (currentDayType === 'weekend') {
    workingday = 0; holiday = 0; weekday = 6; // Saturday
  } else {
    workingday = 0; holiday = 1; weekday = 0; // Sunday
  }

  const atemp_c = temp_c * 0.9; // Approximation

  const payload = {
    hr, temp_c, atemp_c, hum, windspeed_kmh,
    season, mnth, holiday, weekday, workingday,
    weathersit: currentWeather, yr: 1
  };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Error en la predicción');
    }

    const data = await res.json();
    renderResult(data);

  } catch (err) {
    console.error('Prediction error:', err);
    document.getElementById('resultCnt').textContent = '!';
    document.getElementById('resultPlaceholder').textContent = `Error: ${err.message}`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="zap" class="w-5 h-5"></i>Calcular Predicción`;
    lucide.createIcons();
  }
}

function renderResult(data) {
  // Animate number
  const cntEl = document.getElementById('resultCnt');
  animateNumber(cntEl, data.cnt);

  // Demand badge
  const badge = document.getElementById('demandBadge');
  badge.textContent = data.demand_label;
  badge.className = 'text-xs font-semibold px-2.5 py-1 rounded-full';
  const colorMap = { green: 'demand-low', amber: 'demand-mid', red: 'demand-high' };
  badge.classList.add(colorMap[data.demand_color] || 'demand-low');
  badge.classList.remove('hidden');

  // Breakdown
  document.getElementById('resultBreakdown').classList.remove('hidden');
  document.getElementById('casualEst').textContent = fmt(data.casual_est);
  document.getElementById('registeredEst').textContent = fmt(data.registered_est);

  const casualPct = data.cnt > 0 ? (data.casual_est / data.cnt * 100) : 18.8;
  document.getElementById('casualMiniBar').style.width = `${casualPct}%`;

  // Hide placeholder
  document.getElementById('resultPlaceholder').textContent = '';

  // Pop animation
  const card = document.getElementById('resultCard');
  card.classList.remove('result-pop');
  void card.offsetWidth;
  card.classList.add('result-pop');
}

// ─────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Init Lucide icons
  lucide.createIcons();

  // Init hour label
  updateHourLabel(document.getElementById('sliderHr').value);

  // Bootstrap data
  loadData();
});
