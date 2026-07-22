import React, { useState, useMemo } from 'react';
import { 
  Orbit, 
  LayoutDashboard, 
  MessageSquare, 
  GitFork, 
  Wrench, 
  Activity, 
  Award, 
  Settings, 
  Paperclip, 
  ArrowUpRight, 
  CheckCircle2, 
  Sparkles, 
  Zap, 
  ShieldCheck, 
  Flame, 
  Cpu, 
  Layers, 
  Database, 
  Trash2, 
  Maximize2, 
  TrendingUp, 
  Sliders, 
  RotateCcw,
  Search,
  ChevronDown
} from 'lucide-react';

export default function ThinkLMDashboard() {
  // Navigation State
  const [activeNav, setActiveNav] = useState('dashboard');
  const [activeMode, setActiveMode] = useState('Local Mode (Offline)');
  const [promptText, setPromptText] = useState('');
  
  // Scheduler State
  const [currentStep, setCurrentStep] = useState(55);
  const [kSteps, setKSteps] = useState(50);
  const [learningRate, setLearningRate] = useState('1e-6');
  const [klBeta, setKlBeta] = useState(0.001);

  // Derived Phase
  const isPhase1 = useMemo(() => (currentStep % (2 * kSteps)) < kSteps, [currentStep, kSteps]);
  const activePhaseTitle = isPhase1 ? 'PHASE 1: Policy Training' : 'PHASE 2: Rubric Training';
  const activePhaseSubtitle = isPhase1 
    ? 'Updating Policy Parameters (θ) • Frozen Rubric Generator (φ)' 
    : 'Updating Rubric Generator (φ) • Frozen Policy Parameters (θ)';

  // Telemetry SVG Line Generator
  const marginCurve = [0.05, 0.12, 0.18, 0.25, 0.32, 0.38, 0.42];
  const complianceCurve = [99.2, 99.2, 99.2, 99.2, 99.2, 99.2, 99.2];
  const kappaCurve = [0.25, 0.35, 0.44, 0.52, 0.58, 0.61, 0.64];

  const renderSvgLine = (data, min, max, color) => {
    const width = 280;
    const height = 70;
    const points = data.map((val, idx) => {
      const x = (idx / (data.length - 1)) * width;
      const y = height - ((val - min) / (max - min)) * (height - 16) - 8;
      return `${x},${y}`;
    }).join(' ');

    return (
      <svg width="100%" height="70" viewBox={`0 0 ${width} ${height}`} className="overflow-visible">
        <polyline fill="none" stroke={color} strokeWidth="2.5" points={points} strokeLinecap="round" strokeLinejoin="round" />
        {data.map((val, idx) => {
          const x = (idx / (data.length - 1)) * width;
          const y = height - ((val - min) / (max - min)) * (height - 16) - 8;
          return <circle key={idx} cx={x} cy={y} r="3.5" fill={color} className="transition-all hover:r-5 cursor-pointer" />;
        })}
      </svg>
    );
  };

  const handleQuickAction = (actionName) => {
    if (actionName === 'rubric') {
      setPromptText('The perimeter of a rectangle is 48. What is the largest possible area of the rectangle?');
    } else if (actionName === 'judge') {
      alert('Executing deterministic Qwen3-1.7B judge evaluation (temperature=0.0)... Score: 0.975/1.000');
    } else if (actionName === 'preference') {
      alert('Sampling on-policy preference pair (a+, a-) via Temporal Contrast (Age gap: 80 steps).');
    } else if (actionName === 'clear_graph') {
      alert('Cognitive Memory Graph consolidated and pruned in local sandbox.');
    }
  };

  return (
    <div className="relative min-h-screen w-full bg-[#030712] text-slate-100 font-sans antialiased overflow-hidden flex flex-col selection:bg-purple-500/30 selection:text-purple-200">
      
      {/* Subtle Lunar Ambient Neon Glows */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute -top-40 -left-40 w-[500px] h-[500px] bg-purple-600/10 rounded-full blur-[120px]" />
        <div className="absolute top-1/2 -right-40 w-[500px] h-[500px] bg-indigo-600/10 rounded-full blur-[120px]" />
        <div className="absolute -bottom-40 left-1/3 w-[500px] h-[500px] bg-cyan-600/10 rounded-full blur-[120px]" />
      </div>

      {/* Main Wrapper */}
      <div className="relative z-10 flex flex-1 h-screen overflow-hidden">

        {/* 1. LEFT NAVIGATION SIDEBAR (Width: 260px) */}
        <aside className="w-[260px] flex-shrink-0 bg-slate-950/80 backdrop-blur-2xl border-r border-slate-800/80 flex flex-col justify-between p-4 z-20">
          <div className="space-y-6">
            {/* Branding Header */}
            <div className="flex items-center gap-3 px-2 py-1">
              <div className="p-2 rounded-xl bg-gradient-to-br from-purple-500/20 via-indigo-500/20 to-cyan-500/20 border border-purple-500/30 text-purple-400 shadow-lg shadow-purple-500/20">
                <Orbit className="w-5 h-5 animate-spin-slow" />
              </div>
              <div>
                <span className="font-bold tracking-tight text-white text-base font-sans bg-clip-text text-transparent bg-gradient-to-r from-purple-200 via-slate-100 to-indigo-200">
                  ThinkLM AI
                </span>
                <div className="text-[10px] font-mono text-purple-400/80 font-medium">EvoLM OS v2.0</div>
              </div>
            </div>

            {/* Navigation Items */}
            <nav className="space-y-1">
              {[
                { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
                { id: 'sandbox', label: 'Chat Sandbox', icon: MessageSquare },
                { id: 'memory', label: 'Memory Graph', icon: GitFork },
                { id: 'tools', label: 'Tools (MCP)', icon: Wrench },
                { id: 'training', label: 'GRPO Training Loop', icon: Activity },
                { id: 'evaluation', label: 'Evaluation', icon: Award },
                { id: 'settings', label: 'System Settings', icon: Settings },
              ].map((item) => {
                const IconComponent = item.icon;
                const isActive = activeNav === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => setActiveNav(item.id)}
                    className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-semibold transition-all ${
                      isActive
                        ? 'bg-gradient-to-r from-purple-600/30 to-indigo-600/20 border border-purple-500/40 text-white shadow-lg shadow-purple-900/20'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60 border border-transparent'
                    }`}
                  >
                    <IconComponent className={`w-4 h-4 ${isActive ? 'text-purple-400' : 'text-slate-400'}`} />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </nav>
          </div>

          {/* Sidebar System Spec Pill */}
          <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 text-[11px] font-mono text-slate-400 space-y-1">
            <div className="flex items-center justify-between text-slate-300">
              <span>Model Pool:</span>
              <span className="text-purple-400 font-bold">Qwen3-8B</span>
            </div>
            <div className="flex items-center justify-between text-slate-300">
              <span>Local Judge:</span>
              <span className="text-emerald-400 font-bold">Qwen3-1.7B</span>
            </div>
          </div>
        </aside>

        {/* Right Section Container */}
        <div className="flex-1 flex flex-col h-screen overflow-hidden">

          {/* 2. TOP CONTROL BAR (Height: 64px) */}
          <header className="h-[64px] flex-shrink-0 px-6 bg-slate-950/70 backdrop-blur-2xl border-b border-slate-800/80 flex items-center justify-between z-20">
            {/* Left Mode Selector */}
            <div className="flex items-center gap-3">
              <div className="relative">
                <select
                  value={activeMode}
                  onChange={(e) => setActiveMode(e.target.value)}
                  className="appearance-none bg-slate-900/90 border border-slate-800 rounded-xl px-3 py-1.5 pr-8 text-xs font-medium text-slate-200 focus:outline-none focus:border-purple-500/50 cursor-pointer"
                >
                  <option value="Local Mode (Offline)">Local Mode (Offline)</option>
                  <option value="Distributed Cluster Mode">Distributed Cluster Mode</option>
                </select>
                <ChevronDown className="w-3.5 h-3.5 text-slate-400 absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
              </div>

              <div className="h-4 w-px bg-slate-800" />

              <div className="text-xs font-mono text-slate-400 flex items-center gap-2">
                <span>Session ID:</span>
                <span className="text-indigo-300 bg-slate-900 px-2 py-0.5 rounded-md border border-slate-800 font-bold">
                  thinklm-v2-run-0042
                </span>
              </div>
            </div>

            {/* Right Status Indicators */}
            <div className="flex items-center gap-4 text-xs font-medium">
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse shadow-md shadow-emerald-400/50" />
                <span>MCP Status: Connected</span>
              </div>

              {/* User Avatar */}
              <div className="relative">
                <div className="w-8 h-8 rounded-full bg-slate-900 border-2 border-purple-500/50 flex items-center justify-center font-bold text-xs text-purple-300 shadow-md shadow-purple-500/20 cursor-pointer">
                  TL
                </div>
              </div>
            </div>
          </header>

          {/* Dashboard Scrollable Body */}
          <main className="flex-1 overflow-y-auto p-6 space-y-6 pb-12">

            {/* 3. CENTRED HERO & PROMPT GATEWAY */}
            <section className="max-w-4xl mx-auto text-center space-y-4 py-4">
              <div>
                <h1 className="text-4xl font-extrabold tracking-tight text-white drop-shadow-[0_0_25px_rgba(168,85,247,0.3)] bg-clip-text text-transparent bg-gradient-to-r from-slate-100 via-purple-100 to-indigo-200">
                  ThinkLM AI
                </h1>
                <p className="mt-2 text-xs md:text-sm text-slate-400 max-w-xl mx-auto leading-relaxed">
                  Evolving natural-language discriminative rubrics to unlock self-improving policy alignment.
                </p>
              </div>

              {/* Large Centred Glassmorphic Prompt Container */}
              <div className="relative bg-slate-900/70 backdrop-blur-2xl rounded-2xl border border-purple-500/30 shadow-[0_0_30px_rgba(168,85,247,0.15)] overflow-hidden text-left transition-all focus-within:border-purple-500/60">
                <textarea
                  value={promptText}
                  onChange={(e) => setPromptText(e.target.value)}
                  placeholder="Enter a question to generate a co-evolved, schema-validated rubric..."
                  rows={3}
                  className="w-full p-4 bg-transparent text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none resize-none"
                />

                <div className="flex items-center justify-between px-4 py-2.5 bg-slate-950/60 border-t border-slate-800/80">
                  <div className="flex items-center gap-2 text-slate-400">
                    <button className="p-1.5 hover:text-slate-200 hover:bg-slate-800/60 rounded-lg transition-all">
                      <Paperclip className="w-4 h-4" />
                    </button>
                    <span className="text-[10px] font-mono text-slate-500">Pydantic v2 Validated</span>
                  </div>

                  <button
                    onClick={() => {
                      if (!promptText.trim()) return;
                      alert("Rubric generated successfully! Conforms to EvoLM schema.");
                    }}
                    className="px-4 py-1.5 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white text-xs font-semibold shadow-lg shadow-purple-600/30 flex items-center gap-1.5 transition-all"
                  >
                    <span>Execute</span>
                    <ArrowUpRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              {/* Quick-Action Grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-left">
                {[
                  {
                    id: 'rubric',
                    title: 'Generate Rubric',
                    desc: 'Isolate constraints & target expected values',
                    icon: Sparkles,
                    color: 'text-purple-400',
                  },
                  {
                    id: 'judge',
                    title: 'Run Frozen Judge',
                    desc: 'Deterministic eval using local Qwen3-1.7B',
                    icon: ShieldCheck,
                    color: 'text-emerald-400',
                  },
                  {
                    id: 'preference',
                    title: 'Sample Preference',
                    desc: 'On-policy contrast pair construction',
                    icon: Sliders,
                    color: 'text-indigo-400',
                  },
                  {
                    id: 'clear_graph',
                    title: 'Clear Graph',
                    desc: 'Consolidate memory vectors in sandbox',
                    icon: Trash2,
                    color: 'text-rose-400',
                  },
                ].map((act) => {
                  const ActIcon = act.icon;
                  return (
                    <button
                      key={act.id}
                      onClick={() => handleQuickAction(act.id)}
                      className="p-3.5 rounded-xl bg-slate-900/60 backdrop-blur-xl border border-slate-800/80 hover:border-purple-500/40 hover:bg-slate-900/90 transition-all group space-y-1.5"
                    >
                      <div className="flex items-center gap-2">
                        <ActIcon className={`w-4 h-4 ${act.color}`} />
                        <span className="text-xs font-semibold text-slate-200 group-hover:text-white">
                          {act.title}
                        </span>
                      </div>
                      <p className="text-[10px] text-slate-400 leading-tight">
                        {act.desc}
                      </p>
                    </button>
                  );
                })}
              </div>
            </section>

            {/* CORE INTERACTIVE PANELS: Phase Scheduler & Telemetry Grid */}
            <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6">

              {/* Phase Scheduler Card */}
              <div className="lg:col-span-2 p-5 rounded-2xl bg-slate-900/60 backdrop-blur-2xl border border-slate-800/80 space-y-5">
                <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
                  <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-purple-400" />
                    <h2 className="text-sm font-semibold text-slate-100">Co-Evolution Phase Scheduler</h2>
                  </div>
                  <span className={`px-2.5 py-0.5 text-[10px] font-bold rounded-full border ${
                    isPhase1
                      ? 'bg-purple-500/20 text-purple-300 border-purple-500/40'
                      : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                  }`}>
                    {activePhaseTitle}
                  </span>
                </div>

                {/* Step Slider */}
                <div className="space-y-2">
                  <div className="flex justify-between text-xs text-slate-400 font-mono">
                    <span>Step 1</span>
                    <span className="text-slate-200 font-bold">Step {currentStep} / 100</span>
                    <span>Step 100</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="100"
                    value={currentStep}
                    onChange={(e) => setCurrentStep(parseInt(e.target.value))}
                    className="w-full h-2 bg-slate-950 rounded-lg appearance-none cursor-pointer accent-purple-500"
                  />
                  <div className="text-[11px] text-slate-400 text-center font-mono">
                    {activePhaseSubtitle}
                  </div>
                </div>

                {/* Alternating Block Visualization */}
                <div className="grid grid-cols-2 gap-3 p-1.5 rounded-xl bg-slate-950/80 border border-slate-800">
                  <div className={`p-3.5 rounded-lg border text-center transition-all ${
                    isPhase1
                      ? 'bg-purple-950/40 border-purple-500/60 text-purple-200 shadow-lg shadow-purple-950/40'
                      : 'bg-slate-900/30 border-slate-800/50 text-slate-500 opacity-60'
                  }`}>
                    <div className="text-xs font-bold uppercase tracking-wider mb-1">Phase 1: Policy Training</div>
                    <div className="text-[11px]">Steps 1 - 50 • θ Active</div>
                  </div>

                  <div className={`p-3.5 rounded-lg border text-center transition-all ${
                    !isPhase1
                      ? 'bg-emerald-950/40 border-emerald-500/60 text-emerald-200 shadow-lg shadow-emerald-950/40'
                      : 'bg-slate-900/30 border-slate-800/50 text-slate-500 opacity-60'
                  }`}>
                    <div className="text-xs font-bold uppercase tracking-wider mb-1">Phase 2: Rubric Training</div>
                    <div className="text-[11px]">Steps 51 - 100 • φ Active</div>
                  </div>
                </div>
              </div>

              {/* Cognitive Memory Graph Card */}
              <div className="p-5 rounded-2xl bg-slate-900/60 backdrop-blur-2xl border border-slate-800/80 flex flex-col justify-between space-y-4">
                <div>
                  <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
                    <div className="flex items-center gap-2">
                      <GitFork className="w-4 h-4 text-cyan-400" />
                      <h3 className="text-sm font-semibold text-slate-100">Cognitive Memory Graph</h3>
                    </div>
                    <span className="text-[10px] font-mono text-cyan-400">Consolidated</span>
                  </div>

                  <div className="mt-3 p-3.5 rounded-xl bg-slate-950/80 border border-slate-800 space-y-2 text-xs">
                    <div className="flex justify-between items-center text-slate-300">
                      <span>Vector Nodes:</span>
                      <span className="font-mono text-indigo-300 font-bold">1,248</span>
                    </div>
                    <div className="flex justify-between items-center text-slate-300">
                      <span>Rollout Capacity:</span>
                      <span className="font-mono text-emerald-400 font-bold">2,048 (Capped)</span>
                    </div>
                    <div className="flex justify-between items-center text-slate-300">
                      <span>Active Sampling:</span>
                      <span className="font-mono text-purple-300 font-bold">Filtering Zero-Var</span>
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => alert("Memory Vector Graph view opened.")}
                  className="w-full py-2 px-3 text-xs font-semibold rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-200 border border-slate-800 flex items-center justify-center gap-2 transition-all"
                >
                  <Maximize2 className="w-3.5 h-3.5" />
                  Explore Full Graph View
                </button>
              </div>

            </div>

            {/* Telemetry Section: 3 Mini SVG Line Charts */}
            <div className="max-w-6xl mx-auto space-y-3">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">Real-Time GRPO Telemetry</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 rounded-xl bg-slate-900/60 backdrop-blur-2xl border border-slate-800/80 space-y-2">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-slate-400">Score Margin (m_bar)</span>
                    <span className="font-mono font-bold text-emerald-400">+0.42 Δ</span>
                  </div>
                  {renderSvgLine(marginCurve, 0.0, 0.5, '#10b981')}
                  <div className="text-[10px] text-slate-500 flex justify-between font-mono">
                    <span>Epoch 1 (+0.05)</span>
                    <span>Epoch 7 (+0.42)</span>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-slate-900/60 backdrop-blur-2xl border border-slate-800/80 space-y-2">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-slate-400">Format Compliance</span>
                    <span className="font-mono font-bold text-indigo-400">99.2%</span>
                  </div>
                  {renderSvgLine(complianceCurve, 90.0, 100.0, '#6366f1')}
                  <div className="text-[10px] text-slate-500 flex justify-between font-mono">
                    <span>Target: ≥99.0%</span>
                    <span>Locked via R_format</span>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-slate-900/60 backdrop-blur-2xl border border-slate-800/80 space-y-2">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-slate-400">Fleiss's Kappa (κ_hat)</span>
                    <span className="font-mono font-bold text-amber-400">0.64</span>
                  </div>
                  {renderSvgLine(kappaCurve, 0.0, 1.0, '#f59e0b')}
                  <div className="text-[10px] text-slate-500 flex justify-between font-mono">
                    <span>Baseline (0.25)</span>
                    <span>Consensus (0.64)</span>
                  </div>
                </div>
              </div>
            </div>

          </main>

          {/* 4. BOTTOM STATUS BAR (Height: 32px) */}
          <footer className="h-[32px] flex-shrink-0 px-6 bg-slate-950/90 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono text-slate-400 z-20">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              <span>System Operational • Local Sandboxed Execution • No External API Calls</span>
            </div>
            <div>EvoLM (arXiv:2605.03871)</div>
          </footer>

        </div>
      </div>
    </div>
  );
}
