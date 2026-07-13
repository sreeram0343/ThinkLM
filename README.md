# ThinkLM: Self-Evolving Language Model

ThinkLM is a BTech Capstone Project focused on building a cognitively inspired multi-agent reasoning system for complex task execution. The project aims to combine language-driven planning, tool-awareness, graph-based memory, and iterative self-improvement to create an AI system that reasons, plans, and evolves.

## Project Vision

ThinkLM explores how large language models can function as structured reasoning engines rather than passive text generators. By integrating planning, retrieval, execution, and memory mechanisms, the system learns to tackle increasingly complex tasks with greater accuracy and efficiency.

## Core Cognitive Engine

At the heart of ThinkLM lies a three-part cognitive engine that powers the system's reasoning pipeline:

### 1. **Language Self-Play (LSP)**
LSP enables the model to simulate internal reasoning roles, challenge assumptions, and refine intermediate decisions through structured self-dialogue. This creates a richer deliberation process before committing to actions, ultimately improving solution quality.

### 2. **Instruction-Tool Retrieval (ITR)**
ITR dynamically selects the most relevant tools and execution policies based on the user's objective and the current reasoning context. This allows the system to invoke external capabilities in a principled way, adapting to the task at hand.

### 3. **Dual-Process Graph Memory**
ThinkLM maintains both fast and slow memory pathways:
- **Fast Memory**: A lightweight, reactive memory layer for immediate context
- **Slow Memory**: A structured graph-based memory layer for long-term dependency tracking, task relationships, and reusable knowledge

Together, these mechanisms enable better continuity, more accurate planning, and stronger reasoning over time.

## System Architecture

The architecture below illustrates how the major components collaborate during task execution:

```
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

1. **Master Agent**: Receives the task and oversees the entire reasoning process
2. **Task Planner**: Converts the objective into a directed acyclic graph (DAG) of subtasks
3. **Executor**: Invokes MCP tools and performs concrete steps needed to gather evidence or complete actions
4. **Writer**: Synthesizes the outcomes into a coherent final response

## Why ThinkLM Matters

ThinkLM is not just another chatbot pipeline. It represents a research-oriented approach to agentic AI, where reasoning, memory, and execution are connected in a more disciplined way. The goal is to move beyond in-context learning toward true adaptive cognition—a system that improves through structured self-reflection and knowledge integration.

## Quick Setup

To run the project locally, follow these steps:

### 1. Create a Conda Environment
```bash
conda create -n thinklm python=3.11
conda activate thinklm
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Project
Run the appropriate entry command for your implementation.

> **Note**: If the repository evolves with a specific startup script, update this section to reflect the exact run command.

## Repository Goals

- Build a robust reasoning-first multi-agent architecture
- Demonstrate tool-use and retrieval in a structured workflow
- Explore graph-based memory for adaptive cognition
- Support the capstone goal of practical, research-driven AI development

## Documentation

For more detailed information, check out:
- **[PRD](docs/PRD.md)**: Project Requirements Document
- **[Roadmap](docs/roadmap.md)**: Development stages and milestones
- **[Setup Guide](docs/guide.md)**: Detailed setup and usage instructions

## License

This project is developed as part of an academic capstone effort and can be adapted for further extension and collaboration.

---

Built with Self-Evolving Language Model
