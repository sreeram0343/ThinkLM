import os
import sys
import json
import logging

# Add src to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.planner import PlannerAgent
from utils.token_counter import TokenAuditor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ThinkLM.VerifyReduction")

MOCK_TOOLS = [
    {
        "name": "web_search",
        "description": "Performs general web search using Google search API to find facts, dates, weather, and general information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculator",
        "description": "Calculates math equations, arithmetic operations, basic math formulas, and python eval expressions.",
        "parameters": {
            "type": "object",
            "properties": {
                "expr": {"type": "string", "description": "Expression to evaluate."}
            },
            "required": ["expr"]
        }
    },
    {
        "name": "wikipedia_search",
        "description": "Retrieves summaries and articles from Wikipedia about entities, historic figures, and events.",
        "parameters": {
            "type": "object",
            "properties": {
                "entity": {"type": "string", "description": "Entity to search."}
            },
            "required": ["entity"]
        }
    },
    {
        "name": "weather_lookup",
        "description": "Looks up the current weather conditions, forecast, temperature, humidity for a given city or location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City or region name."}
            },
            "required": ["location"]
        }
    },
    {
        "name": "stock_price",
        "description": "Gets current and historical stock prices, tickers, market indices, and financial market statistics.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol."}
            },
            "required": ["ticker"]
        }
    }
]

def verify_reduction_on_squad(samples_file: str, num_eval_samples: int = 5):
    logger.info(f"Loading SQuAD baseline samples from: {samples_file}")
    with open(samples_file, "r", encoding="utf-8") as f:
        samples = json.load(f)
        
    planner = PlannerAgent()
    auditor = TokenAuditor()
    
    total_mono_tokens = 0
    total_pruned_tokens = 0
    
    print("\n" + "=" * 80)
    print("                    ThinkLM Token Footprint Audit Report")
    print("=" * 80)
    
    for idx, sample in enumerate(samples[:num_eval_samples]):
        query = sample["question"]
        # Step 1: Use Planner's ITR BM25 boundary selector to get KB=2 tools
        pruned_tools = planner.restrict_boundary(query, MOCK_TOOLS)
        
        # Step 2: Audit token footprint
        audit_results = auditor.audit(query, MOCK_TOOLS, pruned_tools)
        
        mono = audit_results["monolithic_tokens"]
        pruned = audit_results["pruned_tokens"]
        reduction = audit_results["reduction"]
        percentage = audit_results["reduction_percentage"]
        
        total_mono_tokens += mono
        total_pruned_tokens += pruned
        
        print(f"\nSample #{idx + 1}:")
        print(f"  Query: '{query}'")
        print(f"  Pruned Tools: {[t['name'] for t in pruned_tools]}")
        print(f"  Monolithic Token Footprint: {mono} tokens")
        print(f"  Pruned Token Footprint:     {pruned} tokens")
        print(f"  Footprint Savings:          {reduction} tokens ({percentage:.2f}% reduction)")
        
    avg_reduction = total_mono_tokens - total_pruned_tokens
    avg_percentage = (avg_reduction / total_mono_tokens * 100) if total_mono_tokens > 0 else 0.0
    
    print("\n" + "=" * 80)
    print("                               Summary Metrics")
    print("-" * 80)
    print(f"Total Monolithic Footprint: {total_mono_tokens} tokens")
    print(f"Total Pruned Footprint:     {total_pruned_tokens} tokens")
    print(f"Aggregate Footprint Saved:  {avg_reduction} tokens ({avg_percentage:.2f}% reduction)")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    samples_file = os.path.join(base_dir, "data", "anchor_set.json")
    verify_reduction_on_squad(samples_file)
