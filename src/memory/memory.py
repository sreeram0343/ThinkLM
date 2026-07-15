import logging
import math
import re
import collections
from typing import Dict, Any, List, Optional, Tuple, Set
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ThinkLM.Memory")


class DualProcessMemory:
    """
    DualProcessMemory implements the unified cognitive memory engine for ThinkLM.
    
    Architecture:
    1. Short-term Episodic Buffer: Strict sliding window (W = 10 messages by default)
       holding raw conversational history. Preserves precise linguistic structure.
    2. Long-term Semantic Graph: Consolidated semantic knowledge profile represented
       as a directed, weighted semantic graph (NetworkX) of entities and concepts.
    3. Spreading Activation Engine: Propagates activation energy from lexical/semantic triggers
       through relation edges to retrieve implicit context, incorporating decay, fan effect,
       and lateral inhibition.
    """

    def __init__(
        self,
        episodic_limit: int = 10,
        decay_rate: float = 0.01,
        spreading_factor: float = 0.8,
    ):
        """
        Initializes the memory system.

        Args:
            episodic_limit (int): Strict sliding window size. Defaults to 10.
            decay_rate (float): Activation decay per propagation step. Defaults to 0.01.
            spreading_factor (float): Gating propagation ratio. Defaults to 0.8.
        """
        self.episodic_limit = episodic_limit
        self.decay_rate = decay_rate
        self.spreading_factor = spreading_factor

        # 1. Episodic Buffer: List of raw message dicts. Managed as a FIFO sliding window.
        self.episodic_buffer: List[Dict[str, Any]] = []

        # 2. Neocortical Semantic Graph
        self.semantic_graph = nx.DiGraph()
        logger.info(
            f"DualProcessMemory initialized. Episodic Buffer Limit W={self.episodic_limit}"
        )

    def add_message(self, role: str, content: str, timestamp: Optional[float] = None) -> None:
        """
        Appends raw message to the episodic sliding buffer. Automatically handles
        eviction internally when W (episodic_limit) is exceeded. Alias of add_episodic_message.
        """
        self.add_episodic_message(role=role, content=content, timestamp=timestamp)

    def add_episodic_message(self, role: str, content: str, timestamp: Optional[float] = None) -> None:
        """
        Appends raw message to the episodic sliding buffer, evicting the oldest message when W is exceeded.
        """
        # Basic input validation
        if not role or not isinstance(role, str):
            raise ValueError("Role must be a non-empty string.")
        if content is None or not isinstance(content, str):
            raise ValueError("Content must be a string.")

        import time
        if timestamp is None:
            timestamp = time.time()
        else:
            try:
                timestamp = float(timestamp)
            except (TypeError, ValueError):
                raise ValueError("Timestamp must be numeric or None.")

        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp,
        }

        self.episodic_buffer.append(message)
        logger.info(
            f"Added message to Episodic Buffer. Current size: {len(self.episodic_buffer)}"
        )

        # Enforce constant-bound sliding window
        if len(self.episodic_buffer) > self.episodic_limit:
            evicted = self.episodic_buffer.pop(0)
            logger.info(
                f"Episodic Buffer limit exceeded. Evicted oldest message from time {evicted.get('timestamp')}"
            )

    def get_recent_messages(self) -> List[Dict[str, Any]]:
        """
        Returns the current buffer contents as a list.
        """
        return list(self.episodic_buffer)

    def clear_episodic_buffer(self) -> None:
        """
        Clears the episodic memory buffer.
        """
        self.episodic_buffer.clear()
        logger.info("Episodic Buffer cleared.")

    def add_semantic_fact(
        self, src: str, tgt: str, rel: str, weight: float = 1.0
    ) -> None:
        """
        Adds a directed edge representing a relationship from concept node src to concept node tgt.
        Does not duplicate existing nodes. If the exact (src, rel, tgt) triple already exists
        or edge exists, updates the weight attribute.
        
        Saves both key 'relation' and 'rel' to guarantee backward compatibility with consolidator.
        """
        # Basic input validation
        if not isinstance(src, str) or not src.strip():
            raise ValueError("Source concept/node name must be a non-empty string.")
        if not isinstance(tgt, str) or not tgt.strip():
            raise ValueError("Target concept/node name must be a non-empty string.")
        if not isinstance(rel, str) or not rel.strip():
            raise ValueError("Relationship label must be a non-empty string.")
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            raise ValueError("Weight must be a numeric value.")

        # Ensure node existence without duplicating attributes unnecessarily
        if not self.semantic_graph.has_node(src):
            self.semantic_graph.add_node(
                src, type="concept", embedding_label=src.lower()
            )
        if not self.semantic_graph.has_node(tgt):
            self.semantic_graph.add_node(
                tgt, type="concept", embedding_label=tgt.lower()
            )

        # DiGraph allows only one edge between src and tgt.
        # If the edge already exists, update/overwrite the weight and properties.
        self.semantic_graph.add_edge(
            src, tgt, relation=rel, rel=rel, weight=weight
        )
        logger.info(
            f"Semantic Graph: added/updated edge '{src}' --({rel})--> '{tgt}' (weight: {weight})"
        )

    def retrieve_context(
        self,
        query: str,
        n_anchors: int = 2,
        steps: int = 3,
        gamma: float = 0.8,
        inhibition_top_m: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        A high-level context retrieval utility wrapper using standard Spreading Activation.
        
        Args:
            query (str): The search query.
            n_anchors (int): Maximum number of trigger node anchors to activate initially.
            steps (int): The propagation depth.
            gamma (float): Sigmoid scaling coefficient for node firing.
            inhibition_top_m (int): Limits number of active nodes retained during propagation.

        Returns:
            List[Tuple[str, float]]: List of retrieved nodes and their activation weights, sorted decending.
        """
        if not self.semantic_graph or len(self.semantic_graph) == 0:
            logger.info("Semantic graph is empty. Retrieval aborted.")
            return []

        # Find anchors
        query_words = set(re.findall(r"\w+", query.lower()))
        if not query_words:
            return []

        anchors_scored = []
        for node in self.semantic_graph.nodes:
            node_label = self.semantic_graph.nodes[node].get("embedding_label", str(node)).lower()
            node_words = set(re.findall(r"\w+", node_label))
            node_key_words = set(re.findall(r"\w+", str(node).lower()))
            intersection = query_words.intersection(node_words) or query_words.intersection(node_key_words)
            if intersection:
                score = len(intersection) / max(len(query_words), 1)
                anchors_scored.append((node, score))

        anchors_scored.sort(key=lambda x: x[1], reverse=True)
        selected_anchors = anchors_scored[:n_anchors]
        if not selected_anchors:
            logger.info("No semantic anchors found in memory.")
            return []

        activation_state = {node: 0.0 for node in self.semantic_graph.nodes}
        for node, _ in selected_anchors:
            activation_state[node] = 1.0

        for step in range(steps):
            new_potentials = {node: 0.0 for node in self.semantic_graph.nodes}
            # Evaporation / Decay and Outward flow
            for u in self.semantic_graph.nodes:
                new_potentials[u] = (1.0 - self.decay_rate) * activation_state[u]
                
                # Flow from predecessors (incoming edges)
                for v in self.semantic_graph.predecessors(u):
                    edge_data = self.semantic_graph.edges[v, u]
                    w_ji = edge_data.get("weight", 1.0)
                    fan_v = self.semantic_graph.out_degree(v)
                    if fan_v > 0:
                        new_potentials[u] += (self.spreading_factor * w_ji * activation_state[v]) / fan_v
            
            # Apply lateral inhibition (only keep top_m potentials)
            sorted_nodes = sorted(new_potentials.items(), key=lambda x: x[1], reverse=True)
            inhibited_potentials = {node: 0.0 for node in self.semantic_graph.nodes}
            for idx, (node, score) in enumerate(sorted_nodes):
                if idx < inhibition_top_m:
                    inhibited_potentials[node] = score

            # Apply sigmoid firing threshold
            for node in self.semantic_graph.nodes:
                v = inhibited_potentials[node]
                if v <= 0.0:
                    activation_state[node] = 0.0
                else:
                    # Math sigmoid thresholded around 0.3
                    activation_state[node] = 1.0 / (1.0 + math.exp(-gamma * (v - 0.3)))

        sorted_results = sorted(activation_state.items(), key=lambda x: x[1], reverse=True)
        return [node_pair for node_pair in sorted_results if node_pair[1] > 0.0]

    def retrieve_spreading_activation(
        self, query: str, steps: int = 3, threshold: float = 0.12
    ) -> List[Tuple[str, float]]:
        """
        Original spreading activation implementation.
        """
        if not self.semantic_graph or len(self.semantic_graph) == 0:
            logger.info("Semantic graph is empty. Spreading activation bypassed.")
            return []

        # 1. Dual-Trigger Anchor Initialization
        query_words = set(re.findall(r"\w+", query.lower()))
        if not query_words:
            logger.info("Empty query context. Spreading activation bypassed.")
            return []

        activation_state: Dict[str, float] = {
            node: 0.0 for node in self.semantic_graph.nodes
        }
        has_anchors = False
        for node in self.semantic_graph.nodes:
            node_label = (
                self.semantic_graph.nodes[node]
                .get("embedding_label", str(node))
                .lower()
            )
            node_words = set(re.findall(r"\w+", node_label))
            node_key_words = set(re.findall(r"\w+", str(node).lower()))

            if query_words.intersection(node_words) or query_words.intersection(
                node_key_words
            ):
                activation_state[node] = 1.0
                has_anchors = True

        if not has_anchors:
            logger.info(
                "No memory anchors activated for query. Feeling of Knowing (FOK) indicates non-retrieval."
            )
            return []

        logger.info(
            f"Spreading Activation initiated with Anchors: {[n for n, a in activation_state.items() if a > 0.0]}"
        )

        # 2. Propagation Steps
        for step in range(steps):
            new_potentials = {node: 0.0 for node in self.semantic_graph.nodes}

            for u in self.semantic_graph.nodes:
                # Potentials decay
                new_potentials[u] = (1.0 - self.decay_rate) * activation_state[u]

                for v in self.semantic_graph.predecessors(u):
                    edge_data = self.semantic_graph.edges[v, u]
                    w_ji = edge_data.get("weight", 1.0)

                    # Fan Effect: Dilute energy based on the out-degree of the sender node v
                    fan_v = self.semantic_graph.out_degree(v)
                    if fan_v > 0:
                        new_potentials[u] += (
                            self.spreading_factor * w_ji * activation_state[v]
                        ) / fan_v

            # 3. Lateral Inhibition (top 7)
            new_potentials = self._apply_lateral_inhibition(new_potentials, top_m=7)

            # 4. Non-Linear Sigmoid Activation Firing
            for node in self.semantic_graph.nodes:
                activation_state[node] = self._sigmoid_fire(new_potentials[node])

        # 5. Uncertainty-Aware Rejection
        sorted_nodes = sorted(
            activation_state.items(), key=lambda x: x[1], reverse=True
        )
        if not sorted_nodes:
            return []

        top_node, top_energy = sorted_nodes[0]
        logger.info(
            f"Activation propagation finished. Top node: '{top_node}' with energy: {top_energy:.4f}"
        )

        if top_energy < threshold:
            logger.warning(
                f"Confidence score {top_energy:.4f} below FOK threshold {threshold}. Preemptively rejecting query."
            )
            return []

        return [
            node_pair for node_pair in sorted_nodes if node_pair[1] >= threshold
        ]

    def _apply_lateral_inhibition(
        self, potentials: Dict[str, float], top_m: int = 7
    ) -> Dict[str, float]:
        sorted_nodes = sorted(
            potentials.items(), key=lambda x: x[1], reverse=True
        )
        inhibited = {}
        for idx, (node, score) in enumerate(sorted_nodes):
            if idx < top_m:
                inhibited[node] = score
            else:
                inhibited[node] = 0.0
        return inhibited

    def _sigmoid_fire(
        self, potential: float, gamma: float = 5.0, theta: float = 0.3
    ) -> float:
        if potential <= 0.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-gamma * (potential - theta)))

    def summary(self) -> str:
        """
        Returns a debugging summary string of memory status.
        """
        return (
            f"DualProcessMemory Summary:\n"
            f"- Episodic Buffer size: {len(self.episodic_buffer)} / {self.episodic_limit}\n"
            f"- Semantic Graph: {self.semantic_graph.number_of_nodes()} nodes, "
            f"{self.semantic_graph.number_of_edges()} edges"
        )

    def __repr__(self) -> str:
        return (
            f"<DualProcessMemory episodic_size={len(self.episodic_buffer)} "
            f"graph_nodes={self.semantic_graph.number_of_nodes()} "
            f"graph_edges={self.semantic_graph.number_of_edges()}>"
        )


if __name__ == "__main__":
    # Internal validation/testing run
    print("--- Testing DualProcessMemory ---")
    memory = DualProcessMemory(episodic_limit=10)

    # 1. Adding 12 messages to episodic_buffer and confirming only the last 10 remain.
    print("\n1. Appending 12 messages to episodic buffer (limit = 10)...")
    for i in range(12):
        memory.add_message(
            role="user" if i % 2 == 0 else "assistant",
            content=f"Message index {i}",
            timestamp=float(i),
        )

    recent = memory.get_recent_messages()
    print(f"Total buffer size: {len(recent)} (expected 10)")
    assert len(recent) == 10
    print(f"Oldest remaining: '{recent[0]['content']}' (expected 'Message index 2')")
    assert recent[0]["content"] == "Message index 2"
    print(f"Newest remaining: '{recent[-1]['content']}' (expected 'Message index 11')")
    assert recent[-1]["content"] == "Message index 11"

    # 2. Calling add_semantic_fact twice with the same src/tgt and confirming no duplicate nodes.
    print(
        "\n2. Adding duplicate nodes/edges to long-term semantic knowledge graph..."
    )
    memory.add_semantic_fact("Han-Wu", "Wu of Han", "alias", weight=1.0)
    # Check node and edge count
    nodes1 = memory.semantic_graph.number_of_nodes()
    edges1 = memory.semantic_graph.number_of_edges()
    print(f"Initial: {nodes1} nodes, {edges1} edges (expected 2 nodes, 1 edge)")
    assert nodes1 == 2
    assert edges1 == 1

    # Add again with different relation/weight to see updates
    memory.add_semantic_fact("Han-Wu", "Wu of Han", "alias", weight=2.0)
    nodes2 = memory.semantic_graph.number_of_nodes()
    edges2 = memory.semantic_graph.number_of_edges()
    print(
        f"After duplicate call: {nodes2} nodes, {edges2} edges (expected 2 nodes, 1 edge)"
    )
    assert nodes2 == 2
    assert edges2 == 1

    # 3. Verifying edge attributes (rel, weight) are stored and retrievable via semantic_graph.edges(data=True)
    print("\n3. Verifying edge attributes (rel, weight)...")
    edges_data = list(memory.semantic_graph.edges(data=True))
    print(f"Retrieved edge data: {edges_data}")
    src_node, tgt_node, attrs = edges_data[0]
    print(f"Relation label: {attrs.get('rel')} (expected 'alias')")
    assert attrs.get("rel") == "alias"
    print(f"Weight value: {attrs.get('weight')} (expected 2.0)")
    assert attrs.get("weight") == 2.0

    print("\nAll internal sanity tests passed!")
