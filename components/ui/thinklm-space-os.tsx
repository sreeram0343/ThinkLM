"use client";

import React, { useState, useMemo } from "react";
import {
  Brain,
  Cpu,
  Layers,
  Activity,
  ShieldCheck,
  GitCompare,
  TrendingUp,
  Sliders,
  Zap,
  Flame,
  CheckCircle,
  Database,
  Award,
  Terminal,
  Compass,
  Sparkles,
  Search,
  Code2,
  RefreshCw,
  Play,
  Share2,
  Lock,
  Globe,
  Radio
} from "lucide-react";
import { cn } from "@/lib/utils";

export default function ThinkLMSpaceOS() {
  const [activeModule, setActiveModule] = useState<"orchestrator" | "rubric" | "training" | "memory">("orchestrator");
  
  // Agent States
  const [agents, setAgents] = useState([
    { name: "Master Agent", role: "Goal Decomposition & Intent Routing", status: "Active", load: "14%" },
    { name: "Planner Agent", role: "DAG Graph Synthesis & Capability Bounding", status: "Active", load: "32%" },
    { name: "Executor Agent", role: "Topological MCP Tool Dispatch", status: "Active", load: "68%" },
    { name: "Writer Agent", role: "Response Synthesis & Safety Overlay", status: "Idle", load: "0%" },
  ]);

  // Telemetry metrics
  const [metrics] = useState({
    avgMargin: "+0.42",
    formatCompliance: "99.2%",
    fleissKappa: "0.64",
    trainingStep: 55,
    totalSteps: 100,
    kSteps: 50,
  });

  return (
    <div className="relative min-h-screen w-full bg-slate-950 text-slate-100 font-sans antialiased overflow-x-hidden">
      {/* Background Lunar/Space Gradient Glows */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-purple-600/15 rounded-full blur-3xl" />
        <div className="absolute top-1/3 -right-40 w-96 h-96 bg-indigo-600/15 rounded-full blur-3xl" />
        <div className="absolute -bottom-40 left-1/3 w-96 h-96 bg-blue-600/15 rounded-full blur-3xl" />
      </div>

      {/* Main Container */}
      <div className="relative z-10 flex flex-col min-h-screen">
        {/* Top OS Bar */}
        <header className="px-6 py-3 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-xl flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-gradient-to-br from-purple-500/20 to-indigo-500/20 border border-purple-500/30 text-purple-300 shadow-lg shadow-purple-500/20">
                <Brain className="w-5 h-5" />
              </div>
              <span className="font-bold tracking-tight text-white text-base bg-clip-text text-transparent bg-gradient-to-r from-purple-200 via-indigo-100 to-blue-200">
                ThinkLM AI OS
              </span>
            </div>
            <span className="px-2.5 py-0.5 text-[10px] font-mono font-bold rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              EvoLM v2.0 Online
            </span>
          </div>

          {/* Module Navigation Tabs */}
          <div className="flex items-center gap-1.5 p-1 rounded-xl bg-slate-900/90 border border-slate-800">
            <button
              onClick={() => setActiveModule("orchestrator")}
              className={cn(
                "px-3.5 py-1.5 text-xs font-semibold rounded-lg transition-all flex items-center gap-2",
                activeModule === "orchestrator"
                  ? "bg-purple-600 text-white shadow-lg shadow-purple-600/30"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              )}
            >
              <Layers className="w-3.5 h-3.5" />
              Agent Hub
            </button>

            <button
              onClick={() => setActiveModule("rubric")}
              className={cn(
                "px-3.5 py-1.5 text-xs font-semibold rounded-lg transition-all flex items-center gap-2",
                activeModule === "rubric"
                  ? "bg-purple-600 text-white shadow-lg shadow-purple-600/30"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              )}
            >
              <Zap className="w-3.5 h-3.5" />
              Rubric Playground
            </button>

            <button
              onClick={() => setActiveModule("training")}
              className={cn(
                "px-3.5 py-1.5 text-xs font-semibold rounded-lg transition-all flex items-center gap-2",
                activeModule === "training"
                  ? "bg-purple-600 text-white shadow-lg shadow-purple-600/30"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              )}
            >
              <Activity className="w-3.5 h-3.5" />
              GRPO Curriculum
            </button>
          </div>

          {/* Telemetry Status Pills */}
          <div className="flex items-center gap-3 text-xs font-mono text-slate-400">
            <div className="flex items-center gap-1.5">
              <Radio className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
              <span>GRPO Step 55/100</span>
            </div>
            <div className="h-3 w-px bg-slate-800" />
            <div className="flex items-center gap-1.5 text-slate-300">
              <ShieldCheck className="w-3.5 h-3.5 text-indigo-400" />
              <span>Format 99.2%</span>
            </div>
          </div>
        </header>

        {/* Core Content Body */}
        <main className="flex-1 p-6 max-w-7xl w-full mx-auto space-y-6">
          {/* Top Banner Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-md">
              <div className="text-[11px] font-medium text-slate-400">Active Phase State</div>
              <div className="text-sm font-bold text-emerald-400 mt-1 flex items-center gap-1.5">
                <CheckCircle className="w-4 h-4" />
                Phase 2: Rubric Training
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-md">
              <div className="text-[11px] font-medium text-slate-400">Shared Model Architecture</div>
              <div className="text-sm font-bold text-purple-300 mt-1 flex items-center gap-1.5">
                <Cpu className="w-4 h-4 text-purple-400" />
                Qwen3-8B Parameter-Sharing
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-md">
              <div className="text-[11px] font-medium text-slate-400">Discriminative Score Margin</div>
              <div className="text-sm font-bold text-indigo-300 mt-1 flex items-center gap-1.5 font-mono">
                <TrendingUp className="w-4 h-4 text-indigo-400" />
                +0.42 Δ (a+ vs a-)
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-md">
              <div className="text-[11px] font-medium text-slate-400">Multi-Judge Fleiss's Kappa</div>
              <div className="text-sm font-bold text-amber-300 mt-1 flex items-center gap-1.5 font-mono">
                <Award className="w-4 h-4 text-amber-400" />
                κ_hat = 0.64 (High Consensus)
              </div>
            </div>
          </div>

          {/* Module 1: Multi-Agent Hub */}
          {activeModule === "orchestrator" && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Agent Status Roster */}
              <div className="p-6 rounded-2xl bg-slate-900/70 border border-slate-800 backdrop-blur-xl space-y-4">
                <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                  <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-purple-400" />
                    <h3 className="text-sm font-semibold text-slate-100">Multi-Agent Roster</h3>
                  </div>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-500/10 text-purple-300 border border-purple-500/20">
                    4 Agents Active
                  </span>
                </div>

                <div className="space-y-3">
                  {agents.map((ag, i) => (
                    <div key={i} className="p-3 rounded-xl bg-slate-950/80 border border-slate-800/80 space-y-2">
                      <div className="flex justify-between items-start text-xs">
                        <div>
                          <span className="font-bold text-slate-200">{ag.name}</span>
                          <p className="text-[11px] text-slate-400">{ag.role}</p>
                        </div>
                        <span className="px-2 py-0.5 text-[10px] font-semibold rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                          {ag.status}
                        </span>
                      </div>
                      <div className="flex justify-between items-center text-[10px] text-slate-500 font-mono">
                        <span>Workload</span>
                        <span className="text-indigo-400">{ag.load}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Task Graph DAG Execution Panel */}
              <div className="lg:col-span-2 p-6 rounded-2xl bg-slate-900/70 border border-slate-800 backdrop-blur-xl space-y-4">
                <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                  <div className="flex items-center gap-2">
                    <Terminal className="w-4 h-4 text-indigo-400" />
                    <h3 className="text-sm font-semibold text-slate-100">Topological DAG Planning & Execution</h3>
                  </div>
                  <span className="text-xs text-slate-400 font-mono">BM25 ITR Capability Boundary</span>
                </div>

                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
                  <div className="text-xs font-semibold text-purple-300">Active Task Node Graph:</div>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                    <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 font-mono space-y-1">
                      <div className="text-indigo-400 font-bold">Node T1: Search</div>
                      <div className="text-slate-400">Tool: web_search</div>
                      <div className="text-[10px] text-emerald-400">Status: Complete</div>
                    </div>

                    <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 font-mono space-y-1">
                      <div className="text-indigo-400 font-bold">Node T2: Search</div>
                      <div className="text-slate-400">Tool: web_search</div>
                      <div className="text-[10px] text-emerald-400">Status: Complete</div>
                    </div>

                    <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 font-mono space-y-1">
                      <div className="text-indigo-400 font-bold">Node T3: Calculate</div>
                      <div className="text-slate-400">Tool: calculator</div>
                      <div className="text-[10px] text-purple-400">Status: Executing</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Module 2: Rubric Playground */}
          {activeModule === "rubric" && (
            <div className="p-6 rounded-2xl bg-slate-900/70 border border-slate-800 backdrop-blur-xl space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <Zap className="w-4 h-4 text-purple-400" />
                  <h3 className="text-sm font-semibold text-slate-100">Instance-Specific Rubric Evaluator</h3>
                </div>
                <span className="text-xs px-2.5 py-0.5 rounded-full bg-amber-500/10 text-amber-300 border border-amber-500/20 flex items-center gap-1 font-semibold">
                  <Flame className="w-3 h-3 text-amber-400" />
                  Dealbreaker Validation Active (Weight ≥ 0.80)
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                  <span className="font-bold text-indigo-400">Criterion #1 (Dealbreaker):</span>
                  <p className="text-slate-200">Provides the correct maximum area of 144, derived from perimeter 48.</p>
                  <div className="flex justify-between text-[11px] text-slate-400 pt-2 border-t border-slate-900">
                    <span>Weight: 0.80</span>
                    <span className="text-emerald-400">Satisfaction: 1.0</span>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                  <span className="font-bold text-indigo-400">Criterion #2:</span>
                  <p className="text-slate-200">Uses perimeter formula 2(L+W)=48 and area formula A=L*W correctly.</p>
                  <div className="flex justify-between text-[11px] text-slate-400 pt-2 border-t border-slate-900">
                    <span>Weight: 0.20</span>
                    <span className="text-emerald-400">Satisfaction: 1.0</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Module 3: GRPO Curriculum */}
          {activeModule === "training" && (
            <div className="p-6 rounded-2xl bg-slate-900/70 border border-slate-800 backdrop-blur-xl space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-emerald-400" />
                  <h3 className="text-sm font-semibold text-slate-100">GRPO Co-Evolution Scheduler (Step 55/100)</h3>
                </div>
                <span className="text-xs font-mono text-slate-400">Learning Rate: 1e-6 • β = 0.001</span>
              </div>

              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
                <div className="flex justify-between text-xs text-slate-400">
                  <span>Phase 1 (Steps 1-50): Policy Training</span>
                  <span>Phase 2 (Steps 51-100): Rubric Generator Training</span>
                </div>
                <div className="w-full h-3 bg-slate-900 rounded-full overflow-hidden flex">
                  <div className="w-1/2 h-full bg-indigo-600 opacity-60" />
                  <div className="w-1/2 h-full bg-emerald-500 shadow-lg shadow-emerald-500/50" />
                </div>
              </div>
            </div>
          )}
        </main>

        {/* Footer */}
        <footer className="px-6 py-4 border-t border-slate-800/80 bg-slate-950/80 backdrop-blur-xl text-center text-xs text-slate-500 font-mono">
          ThinkLM Platform • Research Implementation based on EvoLM (arXiv:2605.03871)
        </footer>
      </div>
    </div>
  );
}
