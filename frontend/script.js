// API Base URL - use empty string for relative paths when deployed on same server
const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' ? 'http://localhost:8081' : '';

// Elements
const searchInput = document.getElementById('search-input');
const recContainer = document.getElementById('rec-container');
const emptyState = document.getElementById('empty-state');
const noResults = document.getElementById('no-results');
const loader = document.getElementById('loader');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const statusPing = document.getElementById('status-ping');
const insightsSection = document.getElementById('insights-section');
const darkModeToggle = document.getElementById('dark-mode-toggle');
let salesChart = null;

// 0. Dark Mode Logic
function initDarkMode() {
    const isDark = localStorage.getItem('darkMode') === 'true' || 
                  (!('darkMode' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches);
    
    if (isDark) {
        document.documentElement.classList.add('dark');
    } else {
        document.documentElement.classList.remove('dark');
    }
}

function toggleDarkMode() {
    const isDark = document.documentElement.classList.toggle('dark');
    localStorage.setItem('darkMode', isDark);
    
    // Refresh chart to update colors if it exists
    if (salesChart) {
        getTopProducts();
    }
}

darkModeToggle.addEventListener('click', toggleDarkMode);

// 1. Check Backend Health
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        if (data.status === 'online') {
            statusDot.className = 'relative inline-flex rounded-full h-2 w-2 bg-emerald-500';
            statusPing.className = 'animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75';
            statusText.innerText = 'System Online';
            statusText.className = 'text-[11px] font-bold text-emerald-600 uppercase tracking-wider';
            // Also fetch dashboard data if online
            getTopProducts();
            getModelStats();
            getTopCombinations();
            getSmartInsights();
        }
    } catch (err) {
        statusDot.className = 'relative inline-flex rounded-full h-2 w-2 bg-rose-500';
        statusPing.className = 'absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75';
        statusText.innerText = 'System Offline';
        statusText.className = 'text-[11px] font-bold text-rose-600 uppercase tracking-wider';
    }
}

// 2. Fetch Recommendations
let debounceTimer;
async function getTopProducts() {
    try {
        const res = await fetch(`${API_BASE}/top-products`);
        const data = await res.json();
        renderSalesChart(data);
    } catch (err) {
        console.error("Error fetching top products:", err);
    }
}

async function getModelStats() {
    try {
        const res = await fetch(`${API_BASE}/model-stats`);
        const data = await res.json();
        
        // Always show 200+ for total items as requested
        document.getElementById('stat-transactions').innerText = '200+';
        
        animateValue('stat-rules', 0, data.total_rules, 1500);
        animateValue('stat-confidence', 0, data.avg_confidence, 1500, true);
        animateValue('stat-lift', 0, data.avg_lift, 1500, false, 2);
    } catch (err) {
        console.error("Error fetching model stats:", err);
    }
}

async function getTopCombinations() {
    try {
        const res = await fetch(`${API_BASE}/top-combinations`);
        const data = await res.json();
        renderCombinations(data);
    } catch (err) {
        console.error("Error fetching top combinations:", err);
    }
}

function renderCombinations(combos) {
    const container = document.getElementById('combinations-list');
    container.innerHTML = '';
    
    combos.forEach((combo, index) => {
        const div = document.createElement('div');
        div.className = "bg-white dark:bg-black p-6 rounded-3xl border border-slate-100 dark:border-slate-800 premium-shadow flex items-center justify-between group hover:border-indigo-200 dark:hover:border-indigo-900/50 transition-all duration-300";
        div.style.animationDelay = `${index * 100}ms`;
        div.classList.add('fade-in');
        
        const ants = combo.antecedents.map(a => `<span class="capitalize font-bold text-slate-700 dark:text-slate-200">${a}</span>`).join(' + ');
        const cons = combo.consequents.map(c => `<span class="capitalize font-bold text-indigo-600 dark:text-indigo-400">${c}</span>`).join(', ');
        
        div.innerHTML = `
            <div class="flex items-center gap-4">
                <div class="w-10 h-10 rounded-xl bg-slate-50 dark:bg-slate-800 flex items-center justify-center text-sm group-hover:scale-110 transition-transform">🛒</div>
                <div>
                    <p class="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Frequently Bought Together</p>
                    <p class="text-sm">${ants} <span class="mx-2 text-slate-300">➔</span> ${cons}</p>
                </div>
            </div>
            <div class="text-right">
                <p class="text-lg font-black text-slate-800 dark:text-white">${combo.confidence}%</p>
                <p class="text-[9px] font-bold text-slate-400 uppercase tracking-tighter">Confidence</p>
            </div>
        `;
        container.appendChild(div);
    });
}

async function getSmartInsights() {
    try {
        const res = await fetch(`${API_BASE}/smart-insights`);
        const data = await res.json();
        const container = document.getElementById('smart-insights-container');
        container.innerHTML = '';
        
        data.forEach(insight => {
            const p = document.createElement('p');
            p.className = "text-lg font-medium leading-relaxed flex items-start gap-3";
            p.innerHTML = `<span class="mt-1.5 w-1.5 h-1.5 rounded-full bg-indigo-300 shrink-0"></span> ${insight}`;
            container.appendChild(p);
        });
    } catch (err) {
        console.error("Error fetching smart insights:", err);
    }
}

function renderSalesChart(data) {
    const ctx = document.getElementById('salesChart').getContext('2d');
    const isDark = document.documentElement.classList.contains('dark');
    
    if (salesChart) {
        salesChart.destroy();
    }

    const labels = data.map(item => item.product.charAt(0).toUpperCase() + item.product.slice(1));
    const counts = data.map(item => item.count);

    salesChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Orders',
                data: counts,
                backgroundColor: isDark ? 'rgba(99, 102, 241, 0.2)' : 'rgba(99, 102, 241, 0.1)',
                borderColor: '#6366f1',
                borderWidth: 2,
                borderRadius: 8,
                barPercentage: 0.6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: isDark ? '#0f172a' : '#1e293b',
                    titleFont: { family: 'Inter', size: 12, weight: 'bold' },
                    bodyFont: { family: 'Inter', size: 12 },
                    padding: 12,
                    cornerRadius: 8,
                    displayColors: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { 
                        color: isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.03)', 
                        drawBorder: false 
                    },
                    ticks: { 
                        font: { family: 'Inter', size: 10, weight: 'bold' }, 
                        color: isDark ? '#64748b' : '#94a3b8' 
                    }
                },
                x: {
                    grid: { display: false },
                    ticks: { 
                        font: { family: 'Inter', size: 10, weight: 'bold' }, 
                        color: isDark ? '#64748b' : '#64748b' 
                    }
                }
            }
        }
    });
}

async function getRecommendations(item) {
    if (!item.trim()) {
        resetUI();
        return;
    }

    loader.classList.remove('hidden');
    emptyState.classList.add('hidden');
    noResults.classList.add('hidden');

    try {
        const res = await fetch(`${API_BASE}/recommend`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ item })
        });
        
        const data = await res.json();
        
        if (data.error || !data.recommendations || data.recommendations.length === 0) {
            recContainer.innerHTML = '';
            noResults.classList.remove('hidden');
            insightsSection.classList.add('opacity-0', 'translate-y-4');
        } else {
            updateUI(data.recommendations);
            updateInsights(data.metadata);
        }
    } catch (err) {
        console.error(err);
        noResults.classList.remove('hidden');
        insightsSection.classList.add('opacity-0', 'translate-y-4');
    } finally {
        loader.classList.add('hidden');
    }
}

// 3. Update UI with Premium Cards
function updateUI(recs) {
    recContainer.innerHTML = '';
    
    recs.forEach((rec, index) => {
        const isTop = index === 0;
        const card = document.createElement('div');
        
        // Premium Styling
        const baseClasses = "bg-white p-7 rounded-[2rem] border transition-all duration-500 card-hover fade-in relative overflow-hidden premium-shadow dark:bg-black dark:border-slate-800";
        const topClasses = isTop ? "glow-border !bg-indigo-50/30 ring-1 ring-indigo-100 dark:!bg-indigo-950/20 dark:ring-indigo-900/30" : "border-slate-100";
        
        card.className = `${baseClasses} ${topClasses}`;
        card.style.animationDelay = `${index * 100}ms`;

        if (isTop) {
            card.innerHTML += `
                <div class="absolute top-0 right-0 gradient-bg text-white text-[10px] font-black px-4 py-2 rounded-bl-3xl uppercase tracking-widest flex items-center gap-2 shadow-lg">
                    <span>Top Match</span>
                    <svg class="w-3.5 h-3.5 fill-current" viewBox="0 0 20 20"><path d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z"></path></svg>
                </div>
            `;
        }

        card.innerHTML += `
            <div class="flex items-start justify-between gap-6">
                <div class="flex items-center gap-5">
                    <div class="w-16 h-16 ${isTop ? 'gradient-bg text-white shadow-xl shadow-indigo-200 dark:shadow-indigo-900/20' : 'bg-slate-50 text-slate-400 dark:bg-slate-800 dark:text-slate-500'} rounded-2xl flex items-center justify-center font-black text-2xl">
                        ${rec.product.charAt(0).toUpperCase()}
                    </div>
                    <div class="space-y-1">
                        <h4 class="text-2xl font-black text-slate-800 capitalize tracking-tight dark:text-white">${rec.product}</h4>
                        <p class="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] dark:text-slate-500">${isTop ? 'AI Recommendation' : 'Frequently Linked'}</p>
                    </div>
                </div>
                <div class="text-right pt-1">
                    <div class="${isTop ? 'gradient-text' : 'text-slate-700 dark:text-slate-300'} text-4xl font-black tracking-tighter">${rec.confidence_display}%</div>
                    <p class="text-[9px] font-black text-slate-300 uppercase tracking-widest dark:text-slate-600">Match</p>
                </div>
            </div>
            <div class="mt-8 pt-6 border-t border-slate-100/80 dark:border-slate-800/50">
                <div class="flex items-start gap-3 bg-slate-50/50 p-4 rounded-2xl border border-slate-100/50 dark:bg-slate-800/30 dark:border-slate-800/50">
                    <div class="w-8 h-8 rounded-full bg-white flex items-center justify-center shadow-sm shrink-0 mt-0.5 dark:bg-slate-800 dark:shadow-none">
                        <svg class="w-4 h-4 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                    </div>
                    <p class="text-sm font-semibold text-slate-500 leading-relaxed italic dark:text-slate-400">
                        "${rec.reason}"
                    </p>
                </div>
            </div>
        `;
        
        recContainer.appendChild(card);
    });
}

// 4. Update Insights with Number Animation
function animateValue(id, start, end, duration, isPercent = false, decimals = 0) {
    const obj = document.getElementById(id);
    if (!obj) return;
    
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const currentVal = progress * (end - start) + start;
        
        if (isPercent) {
            obj.innerText = `${Math.round(currentVal)}%`;
        } else if (decimals > 0) {
            obj.innerText = currentVal.toFixed(decimals);
        } else {
            obj.innerText = Math.floor(currentVal).toLocaleString();
        }
        
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

function updateInsights(meta) {
    if (!meta) return;
    insightsSection.classList.remove('opacity-0', 'translate-y-4');
    
    animateValue('insight-rules', 0, meta.total_rules, 1000);
    animateValue('insight-conf', 0, Math.round(meta.avg_confidence * 100), 1000, true);
    animateValue('insight-lift', 0, meta.avg_lift, 1000, false, 1);
}

// 5. Reset UI
function resetUI() {
    recContainer.innerHTML = '';
    recContainer.appendChild(emptyState);
    emptyState.classList.remove('hidden');
    noResults.classList.add('hidden');
    insightsSection.classList.add('opacity-0', 'translate-y-4');
}

// Event Listeners
searchInput.addEventListener('input', (e) => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        getRecommendations(e.target.value);
    }, 400);
});

searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        getRecommendations(searchInput.value);
    }
});

document.querySelectorAll('.suggestion-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        searchInput.value = btn.innerText;
        getRecommendations(btn.innerText);
    });
});

// 6. Animation Observer
function initAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.animate-on-scroll').forEach(el => {
        observer.observe(el);
    });
}

// Init
initDarkMode();
checkHealth();
initAnimations();
setInterval(checkHealth, 10000);
