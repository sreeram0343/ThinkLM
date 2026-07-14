import logging
import copy
from typing import Dict, Any, List, Optional
from src.memory.memory import DualProcessMemory
from src.agents.master import MasterAgent
from src.utils.config import ConfigManager

logger = logging.getLogger("ThinkLM.Dispatcher")

class REPLDispatcher:
    """
    REPLDispatcher acts as the dual-path input dispatcher at the REPL boundary.
    It routes slash commands to local handlers and plain queries to MasterAgent.
    """
    def __init__(self, memory: DualProcessMemory, master_agent: MasterAgent, config_manager: ConfigManager):
        self.memory = memory
        self.master_agent = master_agent
        self.config_manager = config_manager
        
        # Undo stack of state snapshots
        self.undo_stack: List[Dict[str, Any]] = []
        
        # Default registered MCP tools
        self.registered_mcp_tools = [
            {"name": "web_search", "description": "Clustered search engines", "status": "active"},
            {"name": "calculator", "description": "Python math subprocessor", "status": "active"}
        ]
        
    def save_snapshot(self) -> None:
        """Saves a deep snapshot of memory buffers and configurations for /undo support."""
        snapshot = {
            "episodic_buffer": copy.deepcopy(self.memory.episodic_buffer),
            "semantic_graph": self.memory.semantic_graph.copy(),
            "mode": self.config_manager.mode,
            "mcp_tools": copy.deepcopy(self.registered_mcp_tools)
        }
        self.undo_stack.append(snapshot)
        logger.info(f"State snapshot saved. Undo stack depth: {len(self.undo_stack)}")

    def handle_command(self, user_input: str) -> str:
        """
        Intercepts and processes local slash commands deterministically.
        """
        parts = user_input.strip().split()
        if not parts:
            return ""
            
        command = parts[0].lower()
        args = parts[1:]
        
        if command == "/clear":
            self.save_snapshot()
            self.memory.episodic_buffer.clear()
            logger.info("Episodic buffer cleared via /clear command.")
            return "Session history and sliding episodic memory buffer cleared successfully."
            
        elif command == "/undo":
            if not self.undo_stack:
                logger.warning("Attempted /undo with an empty undo stack.")
                return "No state snapshots available to undo."
                
            snapshot = self.undo_stack.pop()
            self.memory.episodic_buffer = snapshot["episodic_buffer"]
            self.memory.semantic_graph = snapshot["semantic_graph"]
            self.config_manager.mode = snapshot["mode"]
            self.registered_mcp_tools = snapshot["mcp_tools"]
            logger.info("State snapshot restored via /undo command.")
            return f"Reverted to the last state snapshot. (System Mode: '{self.config_manager.mode}')"
            
        elif command == "/mode":
            self.save_snapshot()
            new_mode = self.config_manager.toggle_mode()
            return f"System mode toggled. Config set to: '{new_mode}'."
            
        elif command == "/mcp":
            if not args:
                return self._list_mcp()
                
            subcmd = args[0].lower()
            if subcmd in ["list", "show"]:
                return self._list_mcp()
            elif subcmd in ["register", "add"]:
                if len(args) < 3:
                    return "Usage: /mcp register <name> <description>"
                name = args[1]
                desc = " ".join(args[2:])
                self.save_snapshot()
                self.registered_mcp_tools.append({
                    "name": name,
                    "description": desc,
                    "status": "active"
                })
                logger.info(f"Registered new MCP tool: {name}")
                return f"Successfully registered MCP tool '{name}'."
            elif subcmd == "evaluate":
                if len(args) < 2:
                    return "Usage: /mcp evaluate <name>"
                name = args[1]
                tool = next((t for t in self.registered_mcp_tools if t["name"] == name), None)
                if not tool:
                    return f"MCP tool '{name}' not found."
                return f"Evaluating MCP tool '{name}': Status is '{tool['status']}', description: '{tool['description']}'."
            else:
                return f"Unknown MCP subcommand: '{subcmd}'. Available: list, register, evaluate."
                
        else:
            return f"Unknown command: '{command}'"

    def _list_mcp(self) -> str:
        if not self.registered_mcp_tools:
            return "No Model Context Protocol (MCP) tools registered."
        lines = ["Registered Model Context Protocol (MCP) Tools:"]
        for tool in self.registered_mcp_tools:
            lines.append(f"- {tool['name']}: {tool['description']} [{tool['status']}]")
        return "\n".join(lines)

    def dispatch(self, user_input: str) -> str:
        """
        Routes user input.
        If starting with '/', handles as a slash command.
        Otherwise, routes to MasterAgent.
        """
        cleaned = user_input.strip()
        if not cleaned:
            return ""
            
        if cleaned.startswith("/"):
            return self.handle_command(cleaned)
            
        # Standard query path: routes to collaborative loop
        self.save_snapshot()
        
        # Log entry in episodic memory buffer
        import time
        self.memory.add_episodic_message("user", cleaned, time.time())
        
        # Execute routing via MasterAgent
        result = self.master_agent.run_collaborative_loop(cleaned, self.memory)
        final_answer = result.get("final_answer", "Error: No response generated.")
        
        # Log response in episodic memory buffer
        self.memory.add_episodic_message("assistant", str(final_answer), time.time())
        return str(final_answer)
