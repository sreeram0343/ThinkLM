import os
import sys

def create_structure(base_path="."):
    print(f"Initializing ThinkLM project structure in: {os.path.abspath(base_path)}")

    # Define directories to create
    directories = [
        "config",
        "data",
        "docs",
        "src",
        "src/agents",
        "src/memory",
        "src/training",
        "src/utils",
        "tests"
    ]

    # Create directories
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # Define templates and file contents
    requirements_content = """# PyTorch (optimized for CUDA execution where available)
# Note: For GPU execution, install matching CUDA torch binaries (e.g. pip install torch --index-url https://download.pytorch.org/whl/cu121)
torch>=2.2.0
accelerate>=0.28.0

# HuggingFace Ecosystem
transformers>=4.38.0
peft>=0.9.0
trl>=0.8.0
datasets>=2.18.0

# Vector Search & Sparse Indexes
rank-bm25>=0.2.2
faiss-cpu>=1.8.0
numpy>=1.24.0

# NetworkX (for semantic graph-based memory)
networkx>=3.2.0

# Local Tooling / API Utilities
fastapi>=0.110.0
uvicorn>=0.28.0
pydantic>=2.6.0

# General Utilities
python-dotenv>=1.0.1
requests>=2.31.0
tqdm>=4.66.0
matplotlib>=3.8.0

# Testing
pytest>=8.0.0
"""

    env_example_content = """# ThinkLM Configuration Template
# Rename this file to .env and fill in the values

# General Settings
ENVIRONMENT=development
LOG_LEVEL=INFO

# HuggingFace Configuration
HF_TOKEN=your_huggingface_token_here
HF_HOME=./data/hf_cache

# PyTorch & Device Settings
DEVICE=cuda  # options: cuda, cpu, mps
CUDA_VISIBLE_DEVICES=0

# Model Paths / Configurations
BASE_MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
FINE_TUNED_MODEL_PATH=./data/models/thinklm-lora

# Memory Settings
VECTOR_DB_PATH=./data/vector_db
SEMANTIC_GRAPH_PATH=./data/semantic_graph.json

# API Server Settings
API_HOST=0.0.0.0
API_PORT=8000
"""

    main_py_content = """import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 50)
    print("               ThinkLM Platform Bootstrapped")
    print("=" * 50)
    print("Modular, self-evolving language model platform using GRPO, ITR,")
    print("and Dual-Process (Episodic-Semantic) Memory.")
    print("-" * 50)
    print(f"Environment: {os.getenv('ENVIRONMENT', 'Not Set')}")
    print(f"Device Configured: {os.getenv('DEVICE', 'Not Set')}")
    print("=" * 50)

if __name__ == "__main__":
    main()
"""

    # File mappings (relative path inside base_path -> content)
    files_to_create = {
        "requirements.txt": requirements_content,
        ".env.example": env_example_content,
        "src/main.py": main_py_content,
        "src/__init__.py": "",
        "src/agents/__init__.py": "# ThinkLM Agents (Master, Planner, Executor, Writer)\n",
        "src/memory/__init__.py": "# ThinkLM Memory Modules (Episodic buffer & Semantic Graph)\n",
        "src/training/__init__.py": "# ThinkLM Training Pipelines (GRPO, LoRA, STABLE gating)\n",
        "src/utils/__init__.py": "# ThinkLM General Helper Functions\n",
        "tests/__init__.py": "# ThinkLM Test Suite\n",
        "docs/README.md": "# ThinkLM Documentation\n\n- PRD: Project Requirements Document\n- Roadmap: Development stages\n- Guide: Setup and usage instructions\n"
    }

    # Create files
    for file_rel_path, content in files_to_create.items():
        file_path = os.path.join(base_path, file_rel_path)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created file: {file_path}")

    print("\nProject structure initialized successfully!")

if __name__ == "__main__":
    create_structure()
