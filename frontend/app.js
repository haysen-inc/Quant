/* ============================================================
   SPY Quant Terminal — Dashboard v10
   Pure display dashboard, no training UI.
   ============================================================ */

const API = window.location.origin + '/api';
const $ = (id) => document.getElementById(id);

let ohlcData = null;
let storeData = null;
let simCapital = 10000;
let simLeverage = 1;
let currentRange = 90;

const STRATEGY = {
    base:   { label: '模型', color: '#3b82f6', colorLoss: '#f87171', buy: '开多', win: '平盈', loss: '平亏' },
    expert: { label: '专家', color: '#f59e0b', colorLoss: '#ef5350', buy: '开多', win: '平盈', loss: '平亏' },
};

let candleChart, candleSeries;
let equityChart, eqBase, eqExpert;
let sigBK2Chart, sigBK2Series;
let sigSK2Chart, sigSK2Series;
let sigSP1Chart, sigSP1Series;
let sigSP2Chart, sigSP2Series;

// ============================================================
// Charts
// ============================================================

function getChartTheme() {
    return {
        layout: { background: { type: 'solid', color: '#141820' }, textColor: '#6b7280', fontSize: 11, fontFamily: "'Inter', sans-serif" },
        grid: { vertLines: { color: 'rgba(255,255,255,0.03)' }, horzLines: { color: 'rgba(255,255,255,0.03)' } },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: { color: 'rgba(255,255,255,0.08)', labelBackgroundColor: '#2a2e39' },
            horzLine: { color: 'rgba(255,255,255,0.08)', labelBackgroundColor: '#2a2e39' } },
        rightPriceScale: { borderColor: 'rgba(255,255,255,0.06)' },
        timeScale: { borderColor: 'rgba(255,255,255,0.06)', timeVisible: true, secondsVisible: false },
        autoSize: true,
    };
}

function initCharts() {
    const t = getChartTheme();
    candleChart = LightweightCharts.createChart($('chart-candle'), t);
    candleSeries = candleChart.addCandlestickSeries({ upColor: '#26a69a', downColor: '#ef5350', borderVisible: false, wickUpColor: '#26a69a', wickDownColor: '#ef5350' });

    equityChart = LightweightCharts.createChart($('chart-equity'), t);
    const fmt = { type: 'custom', formatter: (v) => '$' + v.toFixed(0) };
    eqBase = equityChart.addLineSeries({ color: '#3b82f6', lineWidth: 2, priceFormat: fmt });
    eqExpert = equityChart.addLineSeries({ color: '#ef4444', lineWidth: 1, lineStyle: 2, priceFormat: fmt });

    const sf = { type: 'custom', formatter: (v) => v.toFixed(2) };
    sigBK2Chart = LightweightCharts.createChart($('chart-sig-bk2'), t);
    sigBK2Series = sigBK2Chart.addLineSeries({ color: '#10b981', lineWidth: 1, priceFormat: sf });
    sigSK2Chart = LightweightCharts.createChart($('chart-sig-sk2'), t);
    sigSK2Series = sigSK2Chart.addLineSeries({ color: '#ef4444', lineWidth: 1, priceFormat: sf });
    sigSP1Chart = LightweightCharts.createChart($('chart-sig-sp1'), t);
    sigSP1Series = sigSP1Chart.addLineSeries({ color: '#f59e0b', lineWidth: 1, priceFormat: sf });
    sigSP2Chart = LightweightCharts.createChart($('chart-sig-sp2'), t);
    sigSP2Series = sigSP2Chart.addLineSeries({ color: '#8b5cf6', lineWidth: 1, priceFormat: sf });

    [sigBK2Chart, sigSK2Chart, sigSP1Chart, sigSP2Chart].forEach((ch) => {
        ch._refLine = ch.addLineSeries({ color: 'rgba(255,255,255,0.15)', lineWidth: 1, lineStyle: 2, priceFormat: sf, crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false });
    });
}

// ============================================================
// setVisibleRange — single source of truth
// ============================================================

function setVisibleRange(days) {
    currentRange = days;
    document.querySelectorAll('.range-btn').forEach((b) => b.classList.toggle('active', parseInt(b.dataset.range) === days));
    if (!ohlcData || !ohlcData.ohlc || ohlcData.ohlc.length === 0) return;

    const lastBar = ohlcData.ohlc[ohlcData.ohlc.length - 1];
    const fromTS = days === 0 ? 0 : lastBar.time - days * 24 * 3600;
    if (days === 0) candleChart.timeScale().fitContent();
    else candleChart.timeScale().setVisibleRange({ from: fromTS, to: lastBar.time });

    const rangeLabel = days === 0 ? '全部' : days <= 30 ? '1M' : days <= 90 ? '3M' : days <= 180 ? '6M' : days <= 365 ? '1Y' : '2Y';
    const eqMap = { expert: eqExpert, base: eqBase };

    ['expert', 'base'].forEach((type) => {
        const metricEl = $(`v-${type}-pnl`);
        const detailEl = $(`v-${type}-detail`);
        const simEl = $(`sim-val-${type}`);
        const eqSeries = eqMap[type];
        const eq = storeData && storeData[`_eq_${type}`];
        const trades = storeData && storeData[`trades_${type}`];

        let tradeCount = 0, tradeWins = 0;
        if (trades && trades.length > 0) {
            const filtered = days === 0 ? trades : trades.filter((t) => t.entry_time >= fromTS);
            tradeCount = filtered.length;
            tradeWins = filtered.filter((t) => t.win).length;
        }
        const wr = tradeCount > 0 ? (tradeWins / tradeCount * 100).toFixed(1) : '0.0';

        if (eq && eq.length > 0) {
            let rangeEq = days === 0 ? eq : eq.filter(pt => pt.time >= fromTS);
            if (rangeEq.length === 0) rangeEq = eq;
            const baseVal = rangeEq[0].value;
            const eqPnl = Math.round((rangeEq[rangeEq.length - 1].value - baseVal) * 100) / 100;
            setMetricPnL(`v-${type}-pnl`, eqPnl);
            detailEl.textContent = `${rangeLabel} | ${tradeCount}笔 | 胜率 ${wr}%`;
            const rebased = rangeEq.map(pt => ({ time: pt.time, value: simCapital * (1 + (pt.value - baseVal) / 100 * simLeverage) }));
            eqSeries.setData(rebased);
            const endBal = rebased[rebased.length - 1].value;
            simEl.textContent = `$${endBal.toFixed(0)}`;
            simEl.className = `sim-bal-value ${endBal >= simCapital ? 'positive' : 'negative'}`;
        } else {
            metricEl.textContent = '--';
            metricEl.className = 'metric-value';
            detailEl.textContent = '无数据';
            simEl.textContent = '--';
            simEl.className = 'sim-bal-value';
            eqSeries.setData([]);
        }
    });

    equityChart.timeScale().fitContent();
    const activeTab = document.querySelector('.tab.active');
    if (activeTab) {
        const m = activeTab.getAttribute('onclick').match(/switchTradeTab\(this,'(\w+)'\)/);
        if (m) updateCandleMarkers(m[1]);
    }
}

function setMetricPnL(id, value) {
    const el = $(id);
    el.textContent = `${value > 0 ? '+' : ''}${value}%`;
    el.className = `metric-value ${value >= 0 ? 'positive' : 'negative'}`;
}

function applySimParams() {
    simCapital = parseFloat($('sim-capital').value) || 10000;
    simLeverage = parseFloat($('sim-leverage').value) || 1;
    setVisibleRange(currentRange);
    const activeTab = document.querySelector('.tab.active');
    if (activeTab) {
        const m = activeTab.getAttribute('onclick').match(/switchTradeTab\(this,'(\w+)'\)/);
        if (m) renderTradeLog(storeData && storeData[`trades_${m[1]}`] || [], STRATEGY[m[1]].label);
    }
}

// ============================================================
// Trade log & markers
// ============================================================

function renderTradeLog(trades, label) {
    const tbody = $('trade-body');
    tbody.innerHTML = '';
    if (!trades || trades.length === 0) { tbody.innerHTML = '<tr><td colspan="8" class="empty-state">暂无交易记录</td></tr>'; return; }
    [...trades].reverse().forEach((t, i) => {
        const tr = document.createElement('tr');
        const cls = t.win ? 'positive' : 'negative';
        const usdtPnl = simCapital * (t.pnl_pct / 100) * simLeverage;
        tr.innerHTML = `<td>${trades.length - i}</td><td>${formatTS(t.entry_time)}</td><td>${formatTS(t.exit_time)}</td>
            <td>$${t.entry_price.toFixed(2)}</td><td>$${t.exit_price.toFixed(2)}</td>
            <td class="${cls}">${t.pnl_pct >= 0 ? '+' : ''}${t.pnl_pct.toFixed(3)}%</td>
            <td class="${cls}">${usdtPnl >= 0 ? '+' : ''}${usdtPnl.toFixed(2)}</td>
            <td class="${cls}">${t.win ? '盈利' : '亏损'}</td>`;
        tbody.appendChild(tr);
    });
}

function switchTradeTab(btn, type) {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    btn.classList.add('active');
    if (!storeData) return;
    renderTradeLog(storeData[`trades_${type}`] || [], STRATEGY[type].label);
    updateCandleMarkers(type);
}
window.switchTradeTab = switchTradeTab;

function updateCandleMarkers(type) {
    if (!storeData) return;
    const trades = storeData[`trades_${type}`] || [];
    const s = STRATEGY[type];
    try { document.querySelector('#chart-candle').closest('.chart-panel').querySelector('h2').innerHTML = `K线图 · SPY 1H <span style="color:${s.color};font-size:11px;margin-left:8px;">[${s.label}]</span>`; } catch(e) {}
    const markers = [];
    trades.forEach((t) => {
        markers.push({ time: t.entry_time, position: 'belowBar', color: s.color, shape: 'arrowUp', text: s.buy, size: 1 });
        markers.push({ time: t.exit_time, position: 'aboveBar', color: t.win ? s.color : s.colorLoss, shape: 'arrowDown', text: t.win ? s.win : s.loss, size: 1 });
    });
    markers.sort((a, b) => a.time - b.time);
    candleSeries.setMarkers(markers);
}

// ============================================================
// Parameter Analysis Panel
// ============================================================

function renderParams(data) {
    const el = $('param-analysis');
    $('param-total').textContent = `${data.total_params.toLocaleString()} 参数`;

    const dl = data.decision_layer;
    const fl = data.factor_layer;
    const tl = data.transformer_layer;

    let html = '';

    // ---- Layer 1: Decision ----
    html += `<div class="pa-section">
        <div class="pa-section-title">第一层：九因子 → 交易信号（决策组合层）</div>
        <div class="pa-formula">JX = J1 + J2 + <em>w_bias</em> + J2*TEMA3T3*<em>w_f1</em> + J1*TEMA3T2*<em>w_f2</em> + J3*TEMA3T1<br>
        BK2 = (JX↑) AND (C>MA_down) AND (JX > J1 * <em>w_cond3_j1</em>)</div>
        <table class="pa-table">
            <thead><tr><th>参数</th><th>作用</th><th>原始</th><th>训练后</th><th>变化量</th><th>变化率</th></tr></thead><tbody>`;
    dl.constants.forEach(c => {
        const cls = Math.abs(c.pct) > 20 ? (c.pct > 0 ? 'pa-up' : 'pa-down') : '';
        html += `<tr class="${cls}"><td class="pa-name">${c.name}</td><td class="pa-desc">${c.desc}</td>
            <td>${c.init.toFixed(2)}</td><td>${c.trained.toFixed(2)}</td>
            <td class="pa-delta">${c.delta >= 0 ? '+' : ''}${c.delta.toFixed(2)}</td>
            <td class="pa-pct">${c.pct >= 0 ? '+' : ''}${c.pct.toFixed(1)}%</td></tr>`;
    });
    html += `</tbody></table>`;

    // Temperatures
    html += `<div class="pa-sub-title">逻辑门温度（temp越大 → 判断越锐利）</div>
        <table class="pa-table">
            <thead><tr><th>逻辑门</th><th>控制条件</th><th>原始</th><th>训练后</th><th>锐化倍数</th></tr></thead><tbody>`;
    dl.temperatures.forEach(t => {
        const cls = Math.abs(t.ratio) > 5 ? 'pa-up' : t.ratio < 0 ? 'pa-down' : '';
        html += `<tr class="${cls}"><td class="pa-name">${t.gate}</td><td class="pa-desc">${t.cond}</td>
            <td>${t.init.toFixed(2)}</td><td>${t.trained.toFixed(2)}</td>
            <td class="pa-ratio">${t.ratio.toFixed(1)}x</td></tr>`;
    });
    html += `</tbody></table></div>`;

    // ---- Layer 2: Factors ----
    html += `<div class="pa-section"><div class="pa-section-title">第二层：原始OHLC → 九因子（因子计算层）</div>`;
    fl.forEach(group => {
        html += `<div class="pa-group-header"><span class="pa-group-name">${group.title}</span><span class="pa-group-desc">${group.desc}</span></div>
            <table class="pa-table pa-table-compact">
                <thead><tr><th>名称</th><th>原始周期</th><th>训练周期</th><th>周期变化</th></tr></thead><tbody>`;
        group.params.forEach(p => {
            const cls = Math.abs(p.delta_period) > 5 ? (p.delta_period > 0 ? 'pa-up' : 'pa-down') : '';
            html += `<tr class="${cls}"><td class="pa-name">${p.name}</td>
                <td>N=${p.init_period}</td><td>N=${p.trained_period}</td>
                <td class="pa-delta">${p.delta_period >= 0 ? '+' : ''}${p.delta_period.toFixed(1)}</td></tr>`;
        });
        html += `</tbody></table>`;
    });
    html += `</div>`;

    // ---- Layer 3: Transformer ----
    html += `<div class="pa-section"><div class="pa-section-title">第三层：Transformer 残差纠正层</div>
        <div class="pa-formula">final = sigmoid(logit(expert_prob) + correction)</div>
        <table class="pa-table">
            <thead><tr><th>层</th><th>参数量</th><th>mean|Δ|</th><th>max|Δ|</th><th>RMS(Δ)</th></tr></thead><tbody>`;
    tl.forEach(l => {
        html += `<tr><td class="pa-name">${l.name}</td><td>${l.count.toLocaleString()}</td>
            <td>${l.mean_delta.toFixed(4)}</td><td>${l.max_delta.toFixed(4)}</td><td>${l.rms_delta.toFixed(4)}</td></tr>`;
    });
    html += `</tbody></table></div>`;

    el.innerHTML = html;
}

// ============================================================
// Data Loading
// ============================================================

async function loadOHLC() {
    try {
        log('加载2年K线数据...');
        const res = await fetch(`${API}/ohlc`);
        const data = await res.json();
        if (!data.success) { log(`OHLC失败: ${data.error}`); return; }

        ohlcData = data;
        if (data.ohlc && data.ohlc.length > 0) {
            candleSeries.setData(data.ohlc);
            const first = new Date(data.ohlc[0].time * 1000);
            const last = new Date(data.ohlc[data.ohlc.length - 1].time * 1000);
            $('data-range').textContent = `${formatDate(first)} ~ ${formatDate(last)} (2Y)`;
        }

        if (!storeData) storeData = {};
        if (data.expert) { storeData.trades_expert = data.expert.trade_list; storeData._eq_expert = data.expert.equity; }
        if (data.base)   { storeData.trades_base = data.base.trade_list; storeData._eq_base = data.base.equity; }

        const defaultTab = data.base ? 'base' : 'expert';
        document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
        document.querySelector(`.tab[onclick*="${defaultTab}"]`).classList.add('active');
        renderTradeLog((defaultTab === 'base' && data.base ? data.base : data.expert).trade_list, STRATEGY[defaultTab].label);

        setVisibleRange(currentRange);
        updateCandleMarkers(defaultTab);

        const eI = data.expert ? `专家 ${fmtPnL(data.expert.pnl)} (${data.expert.trades}笔)` : '';
        const bI = data.base ? ` | 模型 ${fmtPnL(data.base.pnl)} (${data.base.trades}笔)` : '';
        log(`数据已加载 (${data.ohlc.length} bars) ${eI}${bI}`);
    } catch (err) { log(`OHLC失败: ${err.message}`); }
}

async function loadMyLanguage() {
    try {
        const res = await fetch(`${API}/extract`);
        const data = await res.json();
        if (data.success) {
            $('ml-code').value = data.code;
        } else {
            $('ml-code').value = `导出失败: ${data.error}`;
        }
    } catch (err) {
        $('ml-code').value = `导出失败: ${err.message}`;
    }
}

async function loadParams() {
    try {
        const res = await fetch(`${API}/params`);
        const data = await res.json();
        if (data.success) renderParams(data);
        else $('param-analysis').innerHTML = `<div class="pa-loading">参数加载失败: ${data.error}</div>`;
    } catch (err) { $('param-analysis').innerHTML = `<div class="pa-loading">参数加载失败: ${err.message}</div>`; }
}

// ============================================================
// Utils
// ============================================================

function log(msg) {
    const ts = new Date().toLocaleTimeString('zh-CN', { hour12: false });
    $('status-console').textContent = `[${ts}] ${msg}`;
    $('header-status').textContent = msg.substring(0, 50);
}
function formatTS(ts) {
    if (!ts) return '--';
    const d = new Date(ts * 1000);
    return `${(d.getMonth()+1).toString().padStart(2,'0')}/${d.getDate().toString().padStart(2,'0')} ${d.getHours().toString().padStart(2,'0')}:${d.getMinutes().toString().padStart(2,'0')}`;
}
function formatDate(d) { return `${d.getFullYear()}-${(d.getMonth()+1).toString().padStart(2,'0')}-${d.getDate().toString().padStart(2,'0')}`; }
function fmtPnL(v) { return `${v > 0 ? '+' : ''}${v}%`; }

// ============================================================
// Init
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    $('btn-sim-apply').addEventListener('click', applySimParams);
    $('btn-copy-ml').addEventListener('click', () => {
        const ta = $('ml-code');
        ta.select();
        navigator.clipboard.writeText(ta.value).then(() => {
            const btn = $('btn-copy-ml');
            btn.textContent = '已复制!';
            setTimeout(() => btn.textContent = '复制代码', 1500);
        });
    });
    document.querySelectorAll('.range-btn').forEach((btn) => btn.addEventListener('click', () => setVisibleRange(parseInt(btn.dataset.range))));
    loadOHLC();
    loadParams();
    loadMyLanguage();
});
