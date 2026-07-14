import logging
import json
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ThinkLM.Planner")

class PlannerAgent:
    """
    PlannerAgent is responsible for query decomposition and dynamic tool-task binding.
    
    Ref: 'Towards AI Search Paradigm' (Baidu, 2025) [1] and 'Instruction-Tool Retrieval (ITR)' (Franko, 2025) [5, 6].
    
    Key Functions:
    1. Dynamic Capability Boundary: Restricts full tool list into query-oriented candidates using ITR [1, 5, 6].
    2. DAG Planning: Maps complex queries to Directed Acyclic Graphs containing atomic sub-tasks [1].
    3. Graph Validation: Ensures the resulting task flow is acyclic, dependency-safe, and topologically sortable.
    """
    
    def __init__(self, mcp_client: Optional[Any] = None):
        self.mcp_client = mcp_client
        logger.info("PlannerAgent initialized.")

    def restrict_boundary(self, query: str, active_mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Implements Instruction-Tool Retrieval (ITR) to bound the active tool dictionary,
        reducing token overhead and attention dilution during planning.
        
        Ref: 'Dynamic System Instructions and Tool Exposure for Efficient Agentic LLMs' [5, 6].
        
        Args:
            query (str): The input user query.
            active_mcp_tools (List[Dict[str, Any]]): The full catalog of available tools.
            
        Returns:
            List[Dict[str, Any]]: A narrowed, task-relevant subset of tools (e.g. top KB=2 tools) [1, 6].
        """
        logger.info(f"Applying ITR pruning to {len(active_mcp_tools)} tools based on query context...")
        
        narrowed_tools = []
        query_words = set(query.lower().replace("?", "").split())
        for tool in active_mcp_tools:
            desc = tool.get("description", "").lower()
            name = tool.get("name", "").lower()
            if any(word in name or word in desc for word in query_words):
                narrowed_tools.append(tool)
                
        # Limit to top-K tools to protect the context window [6].
        kb_limit = 3
        narrowed_tools = narrowed_tools[:kb_limit]
        logger.info(f"Dynamic Capability Boundary restricted to {len(narrowed_tools)} tools.")
        return narrowed_tools

    def create_task_dag(self, query: str, active_mcp_tools: List[Dict[str, Any]]) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """
        Generates a verified Directed Acyclic Graph (DAG) for multi-step reasoning.
        
        Args:
            query (str): The raw input query.
            active_mcp_tools (List[Dict[str, Any]]): Available MCP tools.
            
        Returns:
            Tuple[nx.DiGraph, Dict[str, Any]]: A NetworkX DiGraph instance and its raw JSON representation.
        """
        logger.info(f"Planning sub-tasks for query: '{query}'")
        
        mock_dag_json = {
            "tasks": [
                {
                    "id": "T1",
                    "description": "Search the birthdate of Emperor Han-Wu",
                    "tool": "web_search",
                    "parameters": {"query": "Emperor Han-Wu birth date"}
                },
                {
                    "id": "T2",
                    "description": "Search the birthdate of Julius Caesar",
                    "tool": "web_search",
                    "parameters": {"query": "Julius Caesar birth date"}
                },
                {
                    "id": "T3",
                    "description": "Calculate age difference between Han-Wu and Julius Caesar",
                    "tool": "calculator",
                    "parameters": {
                        "expr": "T1.birth_year - T2.birth_year",
                        "dependencies": ["T1", "T2"]
                    }
                }
            ],
            "dependencies": [
                {"source": "T1", "target": "T3"},
                {"source": "T2", "target": "T3"}
            ]
        }
        
        # NetworkX verification layer: Ensure it is a valid, acyclic graph
        g = nx.DiGraph()
        for task in mock_dag_json["tasks"]:
            g.add_node(
                task["id"],
                description=task["description"],
                tool=task["tool"],
                parameters=task["parameters"]
            )
            
        for edge in mock_dag_json["dependencies"]:
            g.add_edge(edge["source"], edge["target"])
            
        if not nx.is_directed_acyclic_graph(g):
            raise ValueError("Task structure contains circular dependencies! Aborting execution.")
            
        logger.info("Task planning complete. Graph validated successfully as a DAG.")
        return g, mock_dag_json