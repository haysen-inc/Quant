const API_BASE = window.location.origin + "/api";


// Elements
const btnPretrain = document.getElementById('btn-pretrain');
const btnRL = document.getElementById('btn-rl');
const spinPretrain = document.getElementById('spin-pretrain');
const spinRL = document.getElementById('spin-rl');
const statusBox = document.getElementById('status-console');
const mylanguageBox = document.getElementById('mylanguage-box');
const badge = document.getElementById('model-badge');

const metricLoss = document.getElementById('metric-loss');
const metricWR = document.getElementById('metric-wr');
const metricPnL = document.getElementById('metric-pnl');

// Global Chart Instances
let paramCharts = [];
let pnlChartInstance = null;
let probChartInstance = null;

// Thematic Grid Settings
const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
        duration: 2000,
        easing: 'easeOutQuart'
    },
    plugins: {
        legend: { labels: { color: '#a0aec0', font: { family: 'Inter' } } }
    },
    scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#718096' } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#718096' } }
    }
};

function drawParamChart(data) {
    paramCharts.forEach(c => c.destroy());
    paramCharts = [];

    const labels = data.history_loss ? data.history_loss.map((_, i) => `Ep ${i + 1}`) : [0, 0, 0];

    const configs = [
        { id: 'f1Chart', title: 'Factor 1: w_bias', val: data.history_w_bias || [0, 0], color: '#ef4444' },
        { id: 'f2Chart', title: 'Factor 2: w_f1', val: data.history_w_f1 || [0, 0], color: '#3b82f6' },
        { id: 'f3Chart', title: 'Factor 3: w_f2', val: data.history_w_f2 || [0, 0], color: '#10b981' },
        { id: 'f4Chart', title: 'Factor 4: w_cond3_j1', val: data.history_w_cond3_j1 || [0, 0], color: '#8b5cf6' },
        { id: 'f5Chart', title: 'Factor 5: JX > RX Temp', val: data.history_temp_1 || [0, 0], color: '#f59e0b' },
        { id: 'f6Chart', title: 'Factor 6: JX > J1 Temp', val: data.history_temp_2 || [0, 0], color: '#ec4899' },
        { id: 'f7Chart', title: 'Factor 7: C > MACD Temp', val: data.history_temp_3 || [0, 0], color: '#14b8a6' },
        { id: 'f8Chart', title: 'Factor 8: JX < RX Temp', val: data.history_temp_4 || [0, 0], color: '#6366f1' },
        { id: 'f9Chart', title: 'Factor 9: C < MACU Temp', val: data.history_temp_5 || [0, 0], color: '#fb923c' }
    ];

    configs.forEach(cfg => {
        const ctx = document.getElementById(cfg.id).getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{ label: cfg.title, data: cfg.val, borderColor: cfg.color, tension: 0.4, borderWidth: 2, pointRadius: 0 }]
            },
            options: {
                ...chartOptions,
                plugins: { legend: { display: false }, title: { display: true, text: cfg.title, color: '#a0aec0', font: { size: 10, family: 'Inter' } } },
                scales: { x: { display: false }, y: { ticks: { font: { size: 9 }, color: '#718096' }, grid: { color: 'rgba(255,255,255,0.05)' } } }
            }
        });
        paramCharts.push(chart);
    });
}

function drawProbChart(data) {
    const ctx = document.getElementById('probChart').getContext('2d');
    if (probChartInstance) probChartInstance.destroy();

    const labels = data.history_rl_decision.map((_, i) => `${i}h`);

    probChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                { label: 'Long P(Buy)', data: data.history_rl_decision.map(v => v[0]), borderColor: '#ef4444', tension: 0.1, borderWidth: 1, pointRadius: 0, fill: true, backgroundColor: 'rgba(239, 68, 68, 0.1)' },
                { label: 'Short P(Sell)', data: data.history_rl_decision.map(v => v[1]), borderColor: '#10b981', tension: 0.1, borderWidth: 1, pointRadius: 0, fill: true, backgroundColor: 'rgba(16, 185, 129, 0.1)' }
            ]
        },
        options: {
            ...chartOptions,
            scales: {
                ...chartOptions.scales,
                y: { ...chartOptions.scales.y, title: { display: true, text: 'Agent Confidence Limit', color: '#a0aec0' } }
            }
        }
    });
}

function drawPnLChart(data) {
    const ctx = document.getElementById('pnlChart').getContext('2d');
    if (pnlChartInstance) pnlChartInstance.destroy();

    const labels = data.history_base_pnl.map((_, i) => `${i}h`);

    pnlChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Human Baseline',
                    data: data.history_base_pnl.map(v => v * 100),
                    borderColor: '#ef4444',
                    borderDash: [5, 5],
                    borderWidth: 2,
                    tension: 0.1,
                    fill: false
                },
                {
                    label: 'Online RL Agent',
                    data: data.history_rl_pnl.map(v => v * 100),
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 3,
                    tension: 0.1,
                    fill: true
                }
            ]
        },
        options: {
            ...chartOptions,
            scales: {
                ...chartOptions.scales,
                y: { ...chartOptions.scales.y, title: { display: true, text: 'Cumulative PnL %', color: '#a0aec0' } }
            }
        }
    });
}

// Utility: Set Button State
function setButtonState(btn, spinner, isLoading) {
    if (isLoading) {
        btn.disabled = true;
        spinner.classList.remove('hidden');
    } else {
        btn.disabled = false;
        spinner.classList.add('hidden');
    }
}

function logToConsole(message) {
    statusBox.innerHTML = `[${new Date().toLocaleTimeString()}] ${message}`;
}

// Initial Load of Model State
async function fetchMyLanguagePayload(type) {
    try {
        const rawFeatures = document.getElementById('ast-input-features').value;
        const rawLabels = document.getElementById('ast-input-labels').value;
        const rawMyLanguage = rawFeatures + "\n" + rawLabels;

        const res = await fetch(`${API_BASE}/extract?type=${type}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mylanguage: rawMyLanguage })
        });
        const data = await res.json();
        if (data.success) {
            mylanguageBox.value = data.mylanguage_code;
            badge.innerText = `Current: ${type === 'rl' ? 'Phase 14 Online RL' : 'Base Pre-Trained'}`;
            logToConsole(`Loaded ${type} state dictionary from disk.`);
        }
    } catch (err) {
        logToConsole(`Error connecting to Backend API: ${err}`);
    }
}

// Event Listeners
document.getElementById('btn-load-base').addEventListener('click', () => fetchMyLanguagePayload('base'));
document.getElementById('btn-load-rl').addEventListener('click', () => fetchMyLanguagePayload('rl'));

document.getElementById('btn-copy').addEventListener('click', () => {
    navigator.clipboard.writeText(mylanguageBox.value);
    document.getElementById('btn-copy').innerText = "Copied!";
    setTimeout(() => { document.getElementById('btn-copy').innerText = "Copy"; }, 2000);
});

btnPretrain.addEventListener('click', async () => {
    const epochs = document.getElementById('epoch-input').value;
    const lr = document.getElementById('lr-input').value;
    const rawFeatures = document.getElementById('ast-input-features').value;
    const rawLabels = document.getElementById('ast-input-labels').value;

    // The parser backend expects a single string logic sequence to extract JX and BK2
    const rawMyLanguage = rawFeatures + "\n" + rawLabels;

    setButtonState(btnPretrain, spinPretrain, true);
    btnRL.disabled = true;
    logToConsole(`[AST Parser] Analyzing custom MyLanguage Strategy Structure...`);

    try {
        // 1. AST Tokenization
        const parseRes = await fetch(`${API_BASE}/parse`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mylanguage: rawMyLanguage })
        });
        const parseData = await parseRes.json();

        if (!parseData.success) {
            logToConsole(`[AST Parser Error] ${parseData.error}`);
            setButtonState(btnPretrain, spinPretrain, false);
            btnRL.disabled = false;
            return;
        }

        logToConsole(`[AST Success] Strategy constants tokenized. Initiating 2-Year Base Training... (Epochs: ${epochs}, LR: ${lr}).`);

        // 2. Training Execution with Payload
        const response = await fetch(`${API_BASE}/pretrain`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                epochs: parseInt(epochs),
                learning_rate: parseFloat(lr),
                ast_config: parseData.ast_config
            })
        });

        const data = await response.json();

        if (data.success) {
            const history = data.metrics.history_loss || [];

            const finishTrainingUI = () => {
                mylanguageBox.value = data.mylanguage_code;
                metricLoss.innerText = data.metrics.final_val_loss.toFixed(4);

                metricWR.innerText = `${data.metrics.best_win_rate_percent.toFixed(1)} %`;
                metricPnL.innerText = `${data.metrics.best_pnl_percent > 0 ? '+' : ''}${data.metrics.best_pnl_percent.toFixed(2)} %`;

                if (data.metrics.best_pnl_percent > 0) {
                    metricPnL.parentElement.classList.add('highlight');
                } else {
                    metricPnL.parentElement.classList.remove('highlight');
                }

                if (data.metrics.history_loss && data.metrics.history_loss.length > 0) {
                    drawParamChart(data.metrics);
                }

                badge.innerText = "Current: Base Pre-Trained";
                logToConsole(`Training Complete! Differentiable Expert parameters successfully optimized.`);
                setButtonState(btnPretrain, spinPretrain, false);
                btnRL.disabled = false;
            };

            if (history.length > 0) {
                let currentEpoch = 0;
                // Animate the epochs popping into the console over 2 seconds
                const streamInterval = setInterval(() => {
                    logToConsole(`[PyTorch GPU Engine] Epoch ${currentEpoch + 1}/${history.length} - Loss: ${history[currentEpoch].toFixed(4)}`);
                    currentEpoch++;
                    if (currentEpoch >= history.length) {
                        clearInterval(streamInterval);
                        finishTrainingUI();
                    }
                }, 40); // 40ms * 50 = 2 seconds
            } else {
                finishTrainingUI();
            }
        } else {
            logToConsole(`Error: ${data.error}`);
            setButtonState(btnPretrain, spinPretrain, false);
            btnRL.disabled = false;
        }
    } catch (err) {
        logToConsole(`Critial Error: ${err}`);
        setButtonState(btnPretrain, spinPretrain, false);
        btnRL.disabled = false;
    }
});

btnRL.addEventListener('click', async () => {
    const hold = document.getElementById('hold-input').value;
    const rawFeatures = document.getElementById('ast-input-features').value;
    const rawLabels = document.getElementById('ast-input-labels').value;
    const rawMyLanguage = rawFeatures + "\n" + rawLabels;

    setButtonState(btnRL, spinRL, true);
    btnPretrain.disabled = true;
    logToConsole(`Streamloading latest 90-days live market ticks to RL Agent...`);

    try {
        const response = await fetch(`${API_BASE}/rl`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                hold_period: parseInt(hold),
                mylanguage: rawMyLanguage
            })
        });

        const data = await response.json();

        if (data.success) {
            mylanguageBox.value = data.mylanguage_code;
            metricWR.innerText = `${data.metrics.rl_win_rate_percent.toFixed(1)} % (${data.metrics.rl_trades_count} Trades)`;

            // Calculate Absolute Outperformance against Human Base
            const rl_pnl = data.metrics.rl_agent_pnl_percent;
            const base_pnl = data.metrics.base_expert_pnl_percent;
            metricPnL.innerText = `RL: ${rl_pnl > 0 ? '+' : ''}${rl_pnl.toFixed(2)}% | Base: ${base_pnl > 0 ? '+' : ''}${base_pnl.toFixed(2)}%`;
            if (rl_pnl > 0) metricPnL.parentElement.classList.add('highlight');
            else metricPnL.parentElement.classList.remove('highlight');

            if (data.metrics.history_rl_pnl && data.metrics.history_rl_pnl.length > 0) {
                drawPnLChart(data.metrics);
                if (data.metrics.history_rl_decision) drawProbChart(data.metrics);
            }

            badge.innerText = "Current: Phase 14 Online RL";
            logToConsole(`Live Stream finished. RL Agent PnL: ${data.metrics.rl_agent_pnl_percent}% vs Riged Expert PnL: ${data.metrics.base_expert_pnl_percent}%`);
        } else {
            logToConsole(`Error: ${data.error}`);
        }
    } catch (err) {
        logToConsole(`Critical Error: ${err}`);
    } finally {
        setButtonState(btnRL, spinRL, false);
        btnPretrain.disabled = false;
    }
});

// Init load
window.onload = () => {
    // Draw Empty Graph Structural Overlays immediately so UI isn't jarringly blank
    drawParamChart({});

    drawPnLChart({
        history_base_pnl: [0, 0, 0, 0, 0],
        history_rl_pnl: [0, 0, 0, 0, 0]
    });

    drawProbChart({
        history_rl_decision: [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    });

    fetchMyLanguagePayload('rl');
};
