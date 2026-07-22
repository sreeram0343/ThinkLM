"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  ImageIcon,
  FileUp,
  MonitorIcon,
  CircleUserRound,
  ArrowUpIcon,
  Paperclip,
  PlusIcon,
  Code2,
  Palette,
  Layers,
  Rocket,
  Brain,
  Cpu,
  ShieldCheck,
  GitCompare,
  Flame,
  Activity
} from "lucide-react";

interface AutoResizeProps {
  minHeight: number;
  maxHeight?: number;
}

function useAutoResizeTextarea({ minHeight, maxHeight }: AutoResizeProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const adjustHeight = useCallback(
    (reset?: boolean) => {
      const textarea = textareaRef.current;
      if (!textarea) return;

      if (reset) {
        textarea.style.height = `${minHeight}px`;
        return;
      }

      textarea.style.height = `${minHeight}px`;
      const newHeight = Math.max(
        minHeight,
        Math.min(textarea.scrollHeight, maxHeight ?? Infinity)
      );
      textarea.style.height = `${newHeight}px`;
    },
    [minHeight, maxHeight]
  );

  useEffect(() => {
    if (textareaRef.current) textareaRef.current.style.height = `${minHeight}px`;
  }, [minHeight]);

  return { textareaRef, adjustHeight };
}

export default function RuixenMoonChat() {
  const [message, setMessage] = useState("");
  const { textareaRef, adjustHeight } = useAutoResizeTextarea({
    minHeight: 48,
    maxHeight: 150,
  });

  return (
    <div
      className="relative w-full min-h-screen bg-cover bg-center flex flex-col items-center justify-between font-sans antialiased text-white"
      style={{
        backgroundImage:
          "url('https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?q=80&w=2070&auto=format&fit=crop')",
        backgroundAttachment: "fixed",
      }}
    >
      {/* Overlay for Space/Lunar aesthetic */}
      <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm pointer-events-none" />

      {/* Top OS Navigation Header */}
      <header className="relative z-10 w-full max-w-7xl px-6 py-4 flex items-center justify-between border-b border-white/10 bg-black/40 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-purple-600/20 border border-purple-500/30 text-purple-400 shadow-lg shadow-purple-500/20">
            <Brain className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold tracking-tight text-lg text-white">ThinkLM AI OS</span>
              <span className="px-2 py-0.5 text-[10px] font-bold rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                v2.0 Self-Evolving
              </span>
            </div>
            <p className="text-xs text-neutral-400">Co-Evolved Discriminative Rubrics Engine (EvoLM)</p>
          </div>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10">
            <Cpu className="w-4 h-4 text-indigo-400" />
            <span className="text-neutral-300">Qwen3-8B (Shared)</span>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span className="text-neutral-300">Qwen3-1.7B Judge</span>
          </div>
        </div>
      </header>

      {/* Centered AI Title Section */}
      <div className="relative z-10 flex-1 w-full max-w-4xl flex flex-col items-center justify-center text-center px-4 my-8">
        <div className="p-3 rounded-2xl bg-purple-500/10 border border-purple-500/20 text-purple-300 mb-4 inline-flex items-center gap-2 text-xs font-semibold backdrop-blur-md">
          <Activity className="w-4 h-4 text-purple-400 animate-pulse" />
          Multi-Agent Orchestration • Phase 1 & Phase 2 Co-Evolution Active
        </div>
        <h1 className="text-5xl font-extrabold tracking-tight text-white drop-shadow-md bg-clip-text text-transparent bg-gradient-to-r from-purple-200 via-indigo-100 to-blue-300">
          ThinkLM Cognition Hub
        </h1>
        <p className="mt-3 text-neutral-300 max-w-xl text-sm leading-relaxed">
          Formulate queries, generate instance-specific rubrics, evaluate policy rollouts, or execute multi-agent task graphs.
        </p>
      </div>

      {/* Input Box Section */}
      <div className="relative z-10 w-full max-w-3xl px-4 mb-12">
        <div className="relative bg-black/70 backdrop-blur-xl rounded-2xl border border-neutral-700/80 shadow-2xl shadow-purple-950/40 overflow-hidden">
          <Textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => {
              setMessage(e.target.value);
              adjustHeight();
            }}
            placeholder="Ask a question or enter a math optimization problem (e.g., 'Find rectangle perimeter 48 max area')..."
            className={cn(
              "w-full px-5 py-4 resize-none border-none",
              "bg-transparent text-white text-sm",
              "focus-visible:ring-0 focus-visible:ring-offset-0",
              "placeholder:text-neutral-500 min-h-[56px]"
            )}
            style={{ overflow: "hidden" }}
          />

          {/* Footer Buttons */}
          <div className="flex items-center justify-between px-4 py-3 bg-neutral-900/50 border-t border-neutral-800">
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="icon"
                className="text-neutral-400 hover:text-white hover:bg-neutral-800 rounded-lg"
              >
                <Paperclip className="w-4 h-4" />
              </Button>
              <span className="text-[11px] text-neutral-500 font-mono">Format: Validated JSON Schema</span>
            </div>

            <div className="flex items-center gap-2">
              <Button
                disabled={!message.trim()}
                className={cn(
                  "flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold transition-all shadow-md",
                  message.trim() 
                    ? "bg-purple-600 hover:bg-purple-500 text-white shadow-purple-600/30" 
                    : "bg-neutral-800 text-neutral-500 cursor-not-allowed"
                )}
              >
                <ArrowUpIcon className="w-4 h-4" />
                <span>Execute & Evaluate</span>
              </Button>
            </div>
          </div>
        </div>

        {/* Quick Action Capsules */}
        <div className="flex items-center justify-center flex-wrap gap-2.5 mt-6">
          <QuickAction icon={<Flame className="w-3.5 h-3.5 text-amber-400" />} label="Math Optimization" />
          <QuickAction icon={<GitCompare className="w-3.5 h-3.5 text-emerald-400" />} label="Temporal Contrast" />
          <QuickAction icon={<ShieldCheck className="w-3.5 h-3.5 text-indigo-400" />} label="Frozen Qwen Judge" />
          <QuickAction icon={<Code2 className="w-3.5 h-3.5 text-purple-400" />} label="Generate Rubric" />
          <QuickAction icon={<Layers className="w-3.5 h-3.5 text-blue-400" />} label="Task DAG Planner" />
          <QuickAction icon={<Rocket className="w-3.5 h-3.5 text-pink-400" />} label="GRPO Training" />
        </div>
      </div>
    </div>
  );
}

interface QuickActionProps {
  icon: React.ReactNode;
  label: string;
}

function QuickAction({ icon, label }: QuickActionProps) {
  return (
    <Button
      variant="outline"
      className="flex items-center gap-2 rounded-full border-neutral-800 bg-black/60 backdrop-blur-md text-neutral-300 hover:text-white hover:bg-neutral-800 hover:border-purple-500/40 px-3.5 py-1.5 h-auto text-xs transition-all"
    >
      {icon}
      <span>{label}</span>
    </Button>
  );
}
