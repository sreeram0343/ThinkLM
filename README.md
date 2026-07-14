# ThinkLM: Self-Evolving Language Model

ThinkLM is a capstone project focused on building a cognitively inspired multi-agent reasoning system for complex task execution. The project combines language-driven planning, tool-awareness, graph-based memory, and iterative self-improvement to create an AI system that reasons, plans, and evolves. 

Serving as an experimental platform for researching adaptive AI systems, ThinkLM explores how language models can move beyond static inference by integrating planning, memory, retrieval, tool usage, reflection, evaluation, and continual learning into a unified, modular architecture.

---

# Vision

To build a modular language model architecture capable of continuous adaptation through autonomous reasoning, memory, retrieval, evaluation, and iterative learning. 

The long-term objective is to explore the transition from static language models toward intelligent systems capable of improving their performance over time through accumulated experience and feedback rather than relying solely on one-time pretraining.

---

# Core Cognitive Engine

At the heart of ThinkLM lies a three-part cognitive engine that powers the system's reasoning pipeline:

### 1. Language Self-Play (LSP)
LSP enables the model to simulate internal reasoning roles, challenge assumptions, and refine intermediate decisions through structured self-dialogue. This creates a richer deliberation process before committing to actions, ultimately improving solution quality.

### 2. Instruction-Tool Retrieval (ITR)
ITR dynamically bounds the active tool capability dictionary based on the user's objective and query context, reducing token overhead and attention dilution during planning. ThinkLM implements this sparse indexing boundary selection using BM25 retrieval to isolate the top two most relevant tools.

### 3. Dual-Process Graph Memory
ThinkLM maintains both fast and slow memory pathways:
- Episodic Buffer: A bounded sliding window (W=10 messages) holding raw, uncompressed conversation history to preserve precise linguistic structure.
- Neocortical Semantic Memory: A structured, directed graph-based memory layer (represented via NetworkX) for long-term facts, entity relationships, and knowledge tracking.
- Spreading Activation Engine: Propagates activation energy from query keywords through the semantic graph. It models decay (decay rate of 0.01), dilution of energy based on out-degree (the ACT-R Fan Effect), lateral inhibition (retaining only the top-7 highest potential nodes), Sigmoid activation scaling, and a Feeling of Knowing (FOK) threshold gate (below which retrieval is aborted to prevent hallucinations).

---

# System Architecture

The ThinkLM architecture is composed of independent agentic components that collaborate during task execution.

```text
                        User Query
                            │
                            ▼
                       Master Agent
                  (Complexity Classifier)
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
    WRITER_ONLY     EXECUTOR_INCLUSIVE  PLANNER_ENHANCED
   (Direct Path)     (Single-Step Tool)   (Multi-Step DAG)
         │                  │                  │
         │                  │                  ▼
         │                  │            Planner Agent
         │                  │         (BM25 ITR Boundary)
         │                  │                  │
         │                  │                  ▼
         │                  └───────────► Executor Agent
         │                             (Parallel Topo-Sort)
         │                                     │
         ▼                                     ▼
   Writer Agent ◄──────────────────────────────┘
 (Synthesis & Citations)
         │
         ▼
    Final Response
```

### Core Agent Modules
1. Master Agent: Receives the user query and gates complexity into WRITER_ONLY (factual), EXECUTOR_INCLUSIVE (single-step tool), or PLANNER_ENHANCED (complex multi-step reasoning).
2. Task Planner: Decomposes complex reasoning tasks into a Directed Acyclic Graph (DAG) of sub-tasks and performs graph validation to ensure the workflow is acyclic and dependency-safe.
3. Executor Agent: Traverses the task DAG in topological order, parallelizing independent tasks at the same execution depth and utilizing fallbacks within tool clusters.
4. Writer Agent: Aggregates disparate nodes from the completed DAG trajectory and compiles a coherent final response with aligned source citations.

---

# System Workflow

```text
User Request
      │
      ▼
Task Planning (DAG Construction)
      │
      ▼
Instruction-Tool Retrieval (ITR Pruning)
      │
      ▼
Memory Retrieval (Spreading Activation)
      │
      ▼
Execute External Tools (MCP Layer)
      │
      ▼
Language Model Reasoning & Synthesis
      │
      ▼
Reflection (Self-Correction)
      │
      ▼
Evaluation (Accuracy & Alignment)
      │
      ▼
Experience Logging & Continual Learning
```

---

# Core Components Detailed

### Foundation Language Model
Responsible for reasoning, planning, natural language understanding, natural language generation, and structured self-play.

### Planner
Decomposes complex objectives into structured subtasks, retrieves relevant tools, and validates topological execution paths.

### Retrieval Engine
Integrates external knowledge using sparse indexes (BM25) and dense vector retrievals to enrich prompt contexts.

### Memory System
Maintains short-term conversational context via an episodic sliding window and structured long-term facts using a NetworkX semantic graph.

### Tool Execution Layer
Provides safe, clustered execution of APIs, computations, and local utilities via Model Context Protocol (MCP) clients.

### Reflection Module
Performs self-criticism and validation before generating responses, refining intermediate reasoning steps where anomalies are detected.

### Evaluation Framework
Automates quality checking using execution outputs, structured feedback, and scoring criteria to log trajectories.

### Continual Learning Pipeline
Collects experience logs and prepares datasets for model adaptation, fine-tuning, and alignment.

---

# Technology Stack

```text
Core Technologies
    ├── Python
    └── NetworkX (Graph Modeling)

Sparse & Dense Indexing
    ├── rank-bm25
    └── FAISS

Language Models & Training
    ├── PyTorch
    ├── Transformers
    ├── PEFT (LoRA)
    └── TRL

API & Serving
    ├── FastAPI
    └── uvicorn
```

---

# Repository Structure

```text
ThinkLM
│
├── configs/          # Configuration profiles
├── docs/             # Setup and architectural documentation
├── src/              # Source code directory
│   ├── agents/       # Agent logic (master, planner, executor, writer)
│   ├── memory/       # Cognitive memory (episodic, neocortical graph)
│   ├── training/     # Model fine-tuning and alignment logic
│   ├── utils/        # Shared helper scripts
│   └── main.py       # Main system entry point
│
├── tests/            # System test suites
├── requirements.txt  # Project dependencies
└── README.md         # Project documentation
```

---

# Quick Setup

### 1. Create a Conda Environment
```bash
conda create -n thinklm python=3.11
conda activate thinklm
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Configuration
Copy the environment template and populate the parameters:
```bash
cp .env.example .env
```

### 4. Bootstrapping
You can verify the baseline setup by running:
```bash
python src/main.py
```

---

# Research Focus

ThinkLM investigates the intersection of:
- Large Language Models as Structured Planning Engines
- Spreading Activation Networks for Context Retrieval
- Instruction-Tool Retrieval (ITR) and Capability Boundaries
- Reflection-Based Reasoning and Local Rollback Recovery
- Continual Adapter-Based Fine-Tuning and Alignment

---

# License

This project is developed as a B.Tech Capstone Project for research and educational purposes. The architecture is designed to facilitate experimentation, reproducibility, and future extensions by the AI research community.
