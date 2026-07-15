import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ThinkLM.Main")

# Load environment variables
load_dotenv()

# Add project root to sys.path to resolve imports cleanly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.memory.memory import DualProcessMemory
from src.agents.master import MasterAgent
from src.utils.config import ConfigManager
from src.utils.dispatcher import REPLDispatcher

def main():
    print("=" * 60)
    print("                 ThinkLM Core Platform Bootstrapped")
    print("=" * 60)
    print("Modular, self-evolving language model platform using GRPO, ITR,")
    print("and Dual-Process (Episodic-Semantic) Memory.")
    print("-" * 60)
    print(f"Environment: {os.getenv('ENVIRONMENT', 'Not Set')}")
    print(f"Device Configured: {os.getenv('DEVICE', 'Not Set')}")
    print("=" * 60)
    print("Type '/mode' to toggle planning mode, '/mcp' to inspect tools.")
    print("Type '/clear' to wipe episodic memory, '/undo' to revert.")
    print("Type 'exit' or '/exit' to quit.")
    print("=" * 60)

    # Initialize components
    memory = DualProcessMemory(episodic_limit=10)
    master_agent = MasterAgent()
    config_manager = ConfigManager()
    dispatcher = REPLDispatcher(memory, master_agent, config_manager)

    while True:
        try:
            # Display interactive prompt showing current mode
            mode_prefix = "[PLAN]" if config_manager.mode == "Plan Mode" else "[NORMAL]"
            user_input = input(f"thinklm {mode_prefix} > ")
            
            # Check exit conditions
            if user_input.strip().lower() in ["exit", "quit", "/exit", "/quit"]:
                print("Shutting down ThinkLM Platform. Goodbye!")
                break
                
            if not user_input.strip():
                continue
                
            # Process input through dual-path dispatcher
            output = dispatcher.dispatch(user_input)
            if output:
                print(output)
                print()
                
        except KeyboardInterrupt:
            print("\nInterrupt received. Type 'exit' to quit.")
        except Exception as e:
            logger.error(f"Error in main CLI loop: {e}")

if __name__ == "__main__":
    main()
