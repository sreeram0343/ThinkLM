"use client";

import React from "react";
import RuixenMoonChat from "@/components/ui/ruixen-moon-chat";

export default function DemoPage() {
  return (
    <main className="min-h-screen w-full bg-black text-white">
      {/* Chat Component */}
      <section className="flex justify-center items-start w-full">
        <RuixenMoonChat />
      </section>

      {/* Footer */}
      <footer className="text-center text-neutral-500 py-3 border-t border-neutral-900 text-xs font-mono">
        © {new Date().getFullYear()} ThinkLM Platform • Self-Evolving Language Model Research (EvoLM)
      </footer>
    </main>
  );
}
