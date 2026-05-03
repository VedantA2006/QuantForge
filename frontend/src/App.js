import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Filler, Tooltip, Legend } from 'chart.js';
import { Line } from 'react-chartjs-2';
import './App.css';

ChartJS.register(CategoryScale, LinearScale, PointElement,
  LineElement, Filler, Tooltip, Legend);

const API = process.env.REACT_APP_API_URL || '';

/* ═══════════════════════════════════════════════════════════════
   STATUS BAR
   ═══════════════════════════════════════════════════════════════ */
function StatusBar({ status }) {
  const s = status?.status || 'idle';
  return (
    <div className={`status-pill ${s}`}>
      <span className="status-dot" />
      {s}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   METRIC CARD
   ═══════════════════════════════════════════════════════════════ */
function MetricCard({ label, value, sub, type = 'neutral' }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className={`metric-value ${type}`}>{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   LEADERBOARD
   ═══════════════════════════════════════════════════════════════ */
function Leaderboard({ strategies, onSelect }) {
  if (!strategies || strategies.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-icon">🔍</div>
        <div>No strategies discovered yet</div>
        <div style={{ fontSize: 12, marginTop: 4 }}>
          Waiting for discovery loop...
        </div>
      </div>
    );
  }

  return (
    <table className="leaderboard-table">
      <thead>
        <tr>
          <th>#</th>
          <th>ID</th>
          <th>Score</th>
          <th>Sharpe</th>
          <th>Avg/Yr</th>
          <th>DD</th>
          <th>Trades</th>
          <th>WR</th>
          <th>Avg Mo.</th>
        </tr>
      </thead>
      <tbody>
        {strategies.map((s, i) => {
          const m = s.metrics || {};
          const rankClass = i < 3 ? `rank-${i + 1}` : 'rank-n';
          return (
            <tr key={s.strategy_id || i}
                onClick={() => onSelect && onSelect(s)}
                style={{ cursor: 'pointer' }}>
              <td>
                <span className={`rank-badge ${rankClass}`}>{i + 1}</span>
              </td>
              <td style={{ color: 'var(--accent-blue)' }}>
                {(s.strategy_id || '').slice(0, 8)}
              </td>
              <td>
                <div>{(s.rank_score || 0).toFixed(4)}</div>
                <div className="score-bar">
                  <div className="score-bar-fill"
                       style={{ width: `${(s.rank_score || 0) * 100}%` }} />
                </div>
              </td>
              <td style={{ color: (m.sharpe_ratio || 0) > 0
                ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                {(m.sharpe_ratio || 0).toFixed(2)}
              </td>
              <td style={{ color: (m.avg_yearly_return || 0) > 0
                ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                {(m.avg_yearly_return || 0).toFixed(1)}%
              </td>
              <td style={{ color: 'var(--accent-red)' }}>
                {(m.max_drawdown_pct || 0).toFixed(1)}%
              </td>
              <td>{m.total_trades || 0}</td>
              <td>{(m.win_rate || 0).toFixed(1)}%</td>
              <td style={{ color: (m.avg_monthly_return || 0) > 0
                ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                {(m.avg_monthly_return || 0).toFixed(1)}%
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/* ═══════════════════════════════════════════════════════════════
   LOG VIEWER
   ═══════════════════════════════════════════════════════════════ */
function LogViewer({ logs }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  if (!logs || logs.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-icon">📋</div>
        <div>No logs yet</div>
      </div>
    );
  }

  return (
    <div className="log-viewer">
      {logs.map((log, i) => (
        <div className="log-entry" key={i}
             style={{ animationDelay: `${i * 0.02}s` }}>
          <span className="log-time">
            {(log.timestamp || '').slice(11, 19)}
          </span>
          <span className={`log-level ${log.level}`}>
            {log.level}
          </span>
          <span className="log-msg">{log.message}</span>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   EQUITY CURVE CHART
   ═══════════════════════════════════════════════════════════════ */
function EquityCurve({ equityCurve }) {
  if (!equityCurve || equityCurve.length < 2) {
    return (
      <div className="empty-state">
        <div className="empty-icon">📈</div>
        <div>Select a strategy to view equity curve</div>
      </div>
    );
  }

  const data = {
    labels: equityCurve.map((_, i) => i),
    datasets: [{
      label: 'Equity',
      data: equityCurve,
      borderColor: '#6382ff',
      backgroundColor: 'rgba(99, 130, 255, 0.08)',
      borderWidth: 2,
      fill: true,
      pointRadius: 0,
      tension: 0.3,
    }],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: 'rgba(15, 20, 34, 0.95)',
        borderColor: 'rgba(99, 130, 255, 0.3)',
        borderWidth: 1,
        titleFont: { family: 'Inter' },
        bodyFont: { family: 'JetBrains Mono', size: 12 },
        callbacks: {
          label: (ctx) => `$${ctx.parsed.y.toFixed(2)}`,
        },
      },
    },
    scales: {
      x: {
        display: false,
      },
      y: {
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#555d70', font: { family: 'JetBrains Mono', size: 11 },
          callback: (v) => `$${v.toLocaleString()}` },
      },
    },
  };

  return (
    <div className="chart-container">
      <Line data={data} options={options} />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   STRATEGY DETAILS
   ═══════════════════════════════════════════════════════════════ */
function StrategyDetails({ strategy, symbol, intervals }) {
  if (!strategy) {
    return (
      <div className="empty-state" style={{ minHeight: '120px' }}>
        <div className="empty-icon">🧩</div>
        <div>Select a strategy to view its internal logic</div>
      </div>
    );
  }

  const s = strategy.strategy || {};
  const m = strategy.metrics || {};
  const yearlyReturns = m.yearly_returns || {};
  const avgYearlyReturn = m.avg_yearly_return || 0;
  
  const readable = s.readable || "Generating strategy logic...";
  const lines = readable.split('\n');
  const riskLine = lines.find(l => l.includes('RISK:')) || '';
  const logicLines = lines.filter(l => !l.includes('RISK:') && !l.includes('gen='));

  return (
    <div className="strategy-details">
      <div className="details-header">
        <div className="details-badge">Sector: {symbol} [{intervals?.join(', ')}]</div>
        <div className="details-badge">Gen: {s.generation || 0}</div>
        <div className="details-badge">Origin: {s.origin || 'random'}</div>
        {riskLine && <div className="details-badge highlight">{riskLine.replace('RISK:', '').trim()}</div>}
        <div style={{ flexGrow: 1 }} />
        <a href={`${API}/api/strategy/${strategy.strategy_id}/code`} download>
          <button className="download-btn engine-btn">
            Download Python Engine
          </button>
        </a>
      </div>
      <div style={{ display: 'flex', gap: '16px', marginTop: '16px' }}>
        <div className="code-block" style={{ flexGrow: 1, margin: 0 }}>
          {logicLines.map((line, i) => (
            <div key={i} className="code-line">
              {line.includes('BUY:') ? <span className="keyword buy">BUY:</span> : 
               line.includes('SELL:') ? <span className="keyword sell">SELL:</span> : 
               <span className="code-text">{line}</span>}
              {line.includes('BUY:') || line.includes('SELL:') ? <span className="code-text">{line.split(/BUY:|SELL:/)[1]}</span> : null}
            </div>
          ))}
        </div>
        
        {Object.keys(yearlyReturns).length > 0 && (
          <div className="code-block" style={{ minWidth: '180px', margin: 0, fontSize: '13px' }}>
            <div style={{ color: 'var(--text-muted)', marginBottom: '8px', borderBottom: '1px solid var(--border-color)', paddingBottom: '4px', fontWeight: 600 }}>
              <span style={{ display: 'inline-block', width: '60px' }}>Year</span>
              <span>Return (%)</span>
            </div>
            {Object.entries(yearlyReturns).map(([year, ret]) => (
              <div key={year} style={{ display: 'flex', marginBottom: '4px' }}>
                <span style={{ width: '60px', color: 'var(--accent-blue)' }}>{year}</span>
                <span style={{ color: ret > 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                  {ret.toFixed(2)}
                </span>
              </div>
            ))}
            <div style={{ marginTop: '8px', borderTop: '1px solid var(--border-color)', paddingTop: '8px', display: 'flex', fontWeight: 600 }}>
              <span style={{ width: '60px', color: 'var(--text-color)' }}>Avg</span>
              <span style={{ color: avgYearlyReturn > 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                {avgYearlyReturn.toFixed(2)}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   MAIN APP
   ═══════════════════════════════════════════════════════════════ */
function App() {
  const [status, setStatus] = useState({});
  const [metrics, setMetrics] = useState({});
  const [strategies, setStrategies] = useState([]);
  const [logs, setLogs] = useState([]);
  const [selectedStrategy, setSelectedStrategy] = useState(null);

  const fetchData = useCallback(async () => {
    try {
      const [statusRes, metricsRes, bestRes, logsRes] = await Promise.all([
        fetch(`${API}/api/status`).then(r => r.json()).catch(() => ({})),
        fetch(`${API}/api/metrics`).then(r => r.json()).catch(() => ({})),
        fetch(`${API}/api/best?limit=15`).then(r => r.json()).catch(() => ({ strategies: [] })),
        fetch(`${API}/api/logs?limit=80`).then(r => r.json()).catch(() => ({ logs: [] })),
      ]);
      setStatus(statusRes);
      setMetrics(metricsRes);
      setStrategies(bestRes.strategies || []);
      setLogs(logsRes.logs || []);

      // Auto-select best if none selected
      if (!selectedStrategy && bestRes.strategies?.length > 0) {
        setSelectedStrategy(bestRes.strategies[0]);
      }
    } catch (err) {
      console.error('Fetch error:', err);
    }
  }, [selectedStrategy]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 3000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const formatUptime = (secs) => {
    if (!secs) return '0s';
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    const s = secs % 60;
    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  };

  const equity = selectedStrategy?.metrics?.equity_curve || [];

  const downloadBacktestCSV = async () => {
    if (!selectedStrategy) return;
    try {
      const res = await fetch(`${API}/api/strategy/${selectedStrategy.strategy_id}`);
      const data = await res.json();
      const trades = data.metrics?.trade_history;
      if (!trades || trades.length === 0) {
        alert("No trade history available for this strategy.");
        return;
      }
      
      const headers = ["side", "entry_bar", "exit_bar", "entry_price", "exit_price", "exit_reason", "pnl", "is_win", "holding_bars"];
      const csvContent = [
        headers.join(","),
        ...trades.map(t => headers.map(h => t[h]).join(","))
      ].join("\n");
      
      const blob = new Blob([csvContent], { type: "text/csv" });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.setAttribute("hidden", "");
      a.setAttribute("href", url);
      a.setAttribute("download", `backtest_${selectedStrategy.strategy_id}.csv`);
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } catch (err) {
      console.error("Download failed:", err);
      alert("Failed to download backtest data.");
    }
  };

  return (
    <div className="app">
      {/* ── Header ──────────────────────────────────── */}
      <header className="header">
        <div className="header-left">
          <div className="logo-icon">Q</div>
          <h1>
            QuantForge
            <span>Strategy Discovery Engine</span>
          </h1>
        </div>
        <StatusBar status={status} />
      </header>

      {/* ── Metrics Strip ──────────────────────────── */}
      <div className="metrics-grid">
        <MetricCard
          label="Cycle"
          value={status.cycle || 0}
          sub={`${formatUptime(status.uptime_seconds)} uptime`}
          type="neutral"
        />
        <MetricCard
          label="Evaluated"
          value={(status.total_evaluated || 0).toLocaleString()}
          sub={`${status.total_passed || 0} passed validation`}
          type="neutral"
        />
        <MetricCard
          label="Best Sharpe"
          value={(metrics.best_sharpe || 0).toFixed(2)}
          type={(metrics.best_sharpe || 0) > 0 ? 'positive' : 'negative'}
        />
        <MetricCard
          label="Best Avg/Yr"
          value={`${(metrics.best_avg_yearly_return || 0).toFixed(1)}%`}
          type={(metrics.best_avg_yearly_return || 0) > 0 ? 'positive' : 'negative'}
        />
        <MetricCard
          label="Best DD"
          value={`${(metrics.best_dd || 0).toFixed(1)}%`}
          type="negative"
        />
        <MetricCard
          label="GA Evolutions"
          value={metrics.ga_cycles || 0}
          sub="Genetic algorithm cycles"
          type="neutral"
        />
      </div>

      {/* ── Main Grid ──────────────────────────────── */}
      <div className="main-grid">
        {/* Leaderboard */}
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title">🏆 Strategy Leaderboard</div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              Top {strategies.length}
            </div>
          </div>
          <div className="panel-body">
            <Leaderboard
              strategies={strategies}
              onSelect={setSelectedStrategy}
            />
          </div>
        </div>

        {/* Live Logs */}
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title">📋 Live Logs</div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              Last {logs.length} entries
            </div>
          </div>
          <LogViewer logs={logs} />
        </div>

        {/* Strategy Details */}
        <div className="panel chart-section">
          <div className="panel-header">
            <div className="panel-title">
              🧩 Strategy Details
            </div>
          </div>
          <div className="panel-body" style={{ padding: '0 22px 16px' }}>
            <StrategyDetails 
              strategy={selectedStrategy} 
              symbol={status.symbol} 
              intervals={status.intervals} 
            />
          </div>
        </div>

        {/* Equity Curve */}
        <div className="panel chart-section">
          <div className="panel-header">
            <div className="panel-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
              <div>
                📈 Equity Curve
                {selectedStrategy && (
                  <span style={{ color: 'var(--accent-blue)', fontWeight: 400, fontSize: 13, marginLeft: 8 }}>
                    — {(selectedStrategy.strategy_id || '').slice(0, 8)}
                  </span>
                )}
              </div>
              {selectedStrategy && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                    Score: {(selectedStrategy.rank_score || 0).toFixed(4)}
                  </div>
                  <button onClick={downloadBacktestCSV} className="download-btn">
                    Download CSV
                  </button>
                </div>
              )}
            </div>
          </div>
          <EquityCurve equityCurve={equity} />
        </div>
      </div>
    </div>
  );
}

export default App;
