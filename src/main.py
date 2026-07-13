import os
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
