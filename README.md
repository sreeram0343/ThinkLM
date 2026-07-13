# ThinkLM

ThinkLM is a BTech Capstone Project focused on building a cognitively inspired multi-agent reasoning system for complex task execution. The project aims to combine language-driven planning, tool-aware decision making, and memory-grounded execution into a unified architecture that can solve real-world problems more effectively than single-pass prompting alone.

## Project Vision

ThinkLM is designed to explore how large language models can behave like a structured reasoning engine rather than a passive text generator. By integrating planning, retrieval, execution, and memory, the system moves toward more reliable, interpretable, and adaptive AI workflows.

## Core Cognitive Engine

At the heart of ThinkLM lies a three-part cognitive engine that powers the system’s reasoning pipeline:

### 1. Language Self-Play (LSP)
LSP enables the model to simulate internal reasoning roles, challenge assumptions, and refine intermediate decisions through structured self-dialogue. This creates a richer deliberation process before final output generation.

### 2. Instruction-Tool Retrieval (ITR)
ITR dynamically selects the most relevant tools and execution policies based on the user’s objective and the current reasoning context. This allows the system to invoke external capabilities in a targeted, context-aware manner.

### 3. Dual-Process Graph Memory
ThinkLM maintains both fast and slow memory pathways:
- a lightweight, reactive memory layer for immediate context
- a structured graph-based memory layer for long-term dependency tracking, task relationships, and reusable knowledge

Together, these mechanisms enable better continuity, more accurate planning, and stronger reasoning over time.

## System Architecture

The architecture below illustrates how the major components collaborate during task execution:

```text
                  +----------------------+
                  |   Master Agent       |
                  |  Orchestrates flow   |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | Task Planner (DAG)   |
                  | Decomposes goals     |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  |   Executor           |
                  |   (MCP Tools)        |
                  | Runs actions/tools   |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  |       Writer         |
                  | Produces final output |
                  +----------------------+
```

### Workflow Summary

- The Master Agent receives the task and oversees the reasoning process.
- The Task Planner converts the objective into a directed acyclic graph (DAG) of subtasks.
- The Executor invokes MCP tools and performs the concrete steps needed to gather evidence or complete actions.
- The Writer synthesizes the outcomes into a coherent final response.

## Why ThinkLM Matters

ThinkLM is not just a chatbot pipeline. It represents a research-oriented approach to agentic AI, where reasoning, memory, and execution are connected in a more disciplined way. The goal is to move from prompt-driven intelligence toward a system that can plan, retrieve, act, and reflect more intentionally.

## Quick Setup

To run the project locally, use the following workflow:

1. Create a Conda environment:

```bash
conda create -n thinklm python=3.11
conda activate thinklm
```

2. Install dependencies from the project requirements file:

```bash
pip install -r requirements.txt
```

3. Launch the project using the appropriate entry command for your implementation.

> Note: If the repository evolves with a specific startup script, update this section to reflect the exact run command.

## Repository Goals

- Build a robust reasoning-first multi-agent architecture
- Demonstrate tool-use and retrieval in a structured workflow
- Explore graph-based memory for adaptive cognition
- Support the capstone goal of practical, research-driven AI development

## License

This project is developed as part of an academic capstone effort and can be adapted for further extension and collaboration.
