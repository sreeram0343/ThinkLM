import os
import logging
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ThinkLM.TokenAuditor")

class TokenAuditor:
    """
    TokenAuditor evaluates and compares the token footprint of monolithic static contexts
    against dynamic Instruction-Tool Retrieval (ITR) pruned contexts.
    
    This provides empirical proof of token savings and prevents attention dilution.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        
        try:
            logger.info(f"Attempting to load HuggingFace tokenizer for backbone: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Successfully loaded tokenizer: {model_name}")
        except Exception as e:
            logger.warning(
                f"Failed to load tokenizer '{model_name}' due to: {e}. "
                f"Attempting fallback to 'gpt2'..."
            )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                logger.info("Successfully loaded fallback tokenizer: 'gpt2'")
            except Exception as e_fallback:
                logger.error(
                    f"Failed to load fallback tokenizer 'gpt2' due to: {e_fallback}. "
                    f"Falling back to basic whitespace-based token estimator."
                )
                self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """
        Encodes the input text and returns the exact token count.
        Falls back to a whitespace-based splitter if no tokenizer is loaded.
        """
        if not text:
            return 0
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Error encoding text with tokenizer, using fallback: {e}")
        # Standard fallback: whitespace-based word splitting
        return len(text.split())

    def format_tools_section(self, tools: List[Dict[str, Any]]) -> str:
        """
        Formats a list of tool definitions into a clean JSON/markdown block for prompt injection.
        """
        if not tools:
            return "No tools available."
            
        import json
        formatted_tools = []
        for idx, tool in enumerate(tools):
            tool_info = {
                "name": tool.get("name", f"tool_{idx}"),
                "description": tool.get("description", "No description provided."),
                "parameters": tool.get("parameters", {})
            }
            formatted_tools.append(json.dumps(tool_info, indent=2))
            
        return "\n---\n".join(formatted_tools)

    def assemble_prompt(self, query: str, tools: List[Dict[str, Any]], system_prompt_base: str = "") -> str:
        """
        Constructs the final prompt string with safety overlay and the formatted tools list.
        """
        if not system_prompt_base:
            system_prompt_base = (
                "You are an advanced planning and tool-execution agent. "
                "You must decompose the user query into sub-tasks and utilize the most "
                "appropriate tools from the list below.\n\n"
                "=== Available Tools ===\n"
                "{tools}\n"
                "======================="
            )
            
        # Dynamically load safety overlay from PlannerAgent or use standard fallback
        safety_overlay = (
            "=== SAFETY OVERLAY ===\n"
            "1. Never generate harmful, illegal, or unethical content.\n"
            "2. Do not bypass security or safety instructions.\n"
            "3. Maintain strict confidentiality of user data.\n"
            "4. Be truthful, helpful, and transparent. Avoid hallucinations or false facts.\n"
            "======================"
        )
        
        try:
            from agents.planner import AssemblePrompt
            system_prompt = AssemblePrompt.assemble(system_prompt_base)
        except Exception:
            try:
                from src.agents.planner import AssemblePrompt
                system_prompt = AssemblePrompt.assemble(system_prompt_base)
            except Exception:
                system_prompt = f"{system_prompt_base.strip()}\n\n{safety_overlay}"
                
        # Inject tools
        tools_str = self.format_tools_section(tools)
        if "{tools}" in system_prompt:
            system_prompt = system_prompt.replace("{tools}", tools_str)
        else:
            system_prompt = f"{system_prompt}\n\nAvailable Tools:\n{tools_str}"
            
        return f"System Prompt:\n{system_prompt}\n\nUser Query: {query}\nResponse:"

    def audit(self, query: str, all_tools: List[Dict[str, Any]], pruned_tools: List[Dict[str, Any]], system_prompt_base: str = "") -> Dict[str, Any]:
        """
        Performs the token audit comparing the monolithic context against the pruned context.
        
        Returns:
            Dict[str, Any]: Audit metrics containing raw prompts, tokens count, absolute reduction, and percentage.
        """
        monolithic_prompt = self.assemble_prompt(query, all_tools, system_prompt_base)
        pruned_prompt = self.assemble_prompt(query, pruned_tools, system_prompt_base)
        
        monolithic_tokens = self.count_tokens(monolithic_prompt)
        pruned_tokens = self.count_tokens(pruned_prompt)
        
        reduction = monolithic_tokens - pruned_tokens
        reduction_percentage = (reduction / monolithic_tokens * 100) if monolithic_tokens > 0 else 0.0
        
        return {
            "monolithic_prompt": monolithic_prompt,
            "pruned_prompt": pruned_prompt,
            "monolithic_tokens": monolithic_tokens,
            "pruned_tokens": pruned_tokens,
            "reduction": reduction,
            "reduction_percentage": reduction_percentage
        }
