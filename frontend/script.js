const API_BASE = 'http://localhost:8080';

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
        const baseClasses = "bg-white p-7 rounded-[2rem] border transition-all duration-500 card-hover fade-in relative overflow-hidden premium-shadow";
        const topClasses = isTop ? "glow-border !bg-indigo-50/30 ring-1 ring-indigo-100" : "border-slate-100";
        
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
                    <div class="w-16 h-16 ${isTop ? 'gradient-bg text-white shadow-xl shadow-indigo-200' : 'bg-slate-50 text-slate-400'} rounded-2xl flex items-center justify-center font-black text-2xl">
                        ${rec.product.charAt(0).toUpperCase()}
                    </div>
                    <div class="space-y-1">
                        <h4 class="text-2xl font-black text-slate-800 capitalize tracking-tight">${rec.product}</h4>
                        <p class="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em]">${isTop ? 'AI Recommendation' : 'Frequently Linked'}</p>
                    </div>
                </div>
                <div class="text-right pt-1">
                    <div class="${isTop ? 'gradient-text' : 'text-slate-700'} text-4xl font-black tracking-tighter">${rec.confidence_display}%</div>
                    <p class="text-[9px] font-black text-slate-300 uppercase tracking-widest">Match</p>
                </div>
            </div>
            <div class="mt-8 pt-6 border-t border-slate-100/80">
                <div class="flex items-start gap-3 bg-slate-50/50 p-4 rounded-2xl border border-slate-100/50">
                    <div class="w-8 h-8 rounded-full bg-white flex items-center justify-center shadow-sm shrink-0 mt-0.5">
                        <svg class="w-4 h-4 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                    </div>
                    <p class="text-sm font-semibold text-slate-500 leading-relaxed italic">
                        "${rec.reason}"
                    </p>
                </div>
            </div>
        `;
        
        recContainer.appendChild(card);
    });
}

// 4. Update Insights with Animation
function updateInsights(meta) {
    if (!meta) return;
    insightsSection.classList.remove('opacity-0', 'translate-y-4');
    
    // Animate numbers if possible, otherwise just update
    document.getElementById('insight-rules').innerText = meta.total_rules.toLocaleString();
    document.getElementById('insight-conf').innerText = `${Math.round(meta.avg_confidence * 100)}%`;
    document.getElementById('insight-lift').innerText = meta.avg_lift.toFixed(1);
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

// Init
checkHealth();
setInterval(checkHealth, 10000);
