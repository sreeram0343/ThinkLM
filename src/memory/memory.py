import logging
import math
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple, Set

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ThinkLM.Memory")

class DualProcessMemory:
    """
    DualProcessMemory implements the unified cognitive memory engine for ThinkLM.
    
    Ref: 'Synapse: Spreading Activation' [4], 'Episodic-Semantic Memory' [3, 8, 9],
    and 'Multi-Layer Memory Framework (MLMF)' [10, 11].
    
    Architecture:
    1. Episodic Buffer (Synchronous): Bounded sliding window (W=10 messages) holding raw, uncompressed 
       history. Preserves precise linguistic structure for pronoun and discourse resolution [3, 9].
    2. Neocortical Memory (Asynchronous): Consolidated semantic knowledge profile represented 
       as a directed, weighted semantic graph (NetworkX) of entities and abstract concepts [3, 9].
    3. Spreading Activation Engine: Propagates activation energy from lexical/semantic triggers 
       through temporal, association, and abstraction edges to retrieve implicit contexts,
       incorporating decay, the fan effect, and lateral inhibition [4, 12].
    """
    
    def __init__(self, episodic_limit: int = 10, decay_rate: float = 0.01, spreading_factor: float = 0.8):
        self.episodic_limit = episodic_limit
        self.decay_rate = decay_rate
        self.spreading_factor = spreading_factor
        
        # 1. Episodic Buffer: List of raw message dicts
        self.episodic_buffer: List[Dict[str, Any]] = []
        
        # 2. Neocortical Semantic Graph
        self.semantic_graph = nx.DiGraph()
        logger.info(f"DualProcessMemory initialized. Episodic Buffer Limit W={self.episodic_limit}")

    def add_episodic_message(self, role: str, content: str, timestamp: float) -> None:
        """
        Appends raw message to episodic sliding buffer, evicting the oldest message when W is exceeded.
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        self.episodic_buffer.append(message)
        logger.info(f"Added message to Episodic Buffer. Current size: {len(self.episodic_buffer)}")
        
        # Enforce constant-bound sliding window (W=10)
        # Ref: 'Episodic-Semantic Memory Architecture' Section 3.1 [3].
        if len(self.episodic_buffer) > self.episodic_limit:
            evicted = self.episodic_buffer.pop(0)
            logger.info(f"Episodic Buffer W reached. Evicted oldest message from time {evicted['timestamp']}")

    def add_semantic_fact(self, source_entity: str, target_entity: str, relation: str, weight: float = 1.0) -> None:
        """
        Adds facts/relationships to the neocortical semantic graph.
        """
        if not self.semantic_graph.has_node(source_entity):
            self.semantic_graph.add_node(source_entity, type="concept", embedding_label=source_entity.lower())
        if not self.semantic_graph.has_node(target_entity):
            self.semantic_graph.add_node(target_entity, type="concept", embedding_label=target_entity.lower())
            
        self.semantic_graph.add_edge(source_entity, target_entity, relation=relation, weight=weight)
        logger.info(f"Semantic Graph: added edge '{source_entity}' --({relation})--> '{target_entity}' (weight: {weight})")

    def retrieve_spreading_activation(self, query: str, steps: int = 3, threshold: float = 0.12) -> List[Tuple[str, float]]:
        """
        Retrieves relevant context nodes using spreading activation over the semantic graph.
        
        Ref: 'Synapse: Spreading Activation' Section 3.2 [4] and 'MLMF adaptive gating' [12].
        """
        if not self.semantic_graph or len(self.semantic_graph) == 0:
            logger.info("Semantic graph is empty. Spreading activation bypassed.")
            return []
            
        # 1. Dual-Trigger Anchor Initialization
        anchors = self._initialize_anchors(query)
        if not anchors:
            logger.info("No memory anchors activated for query. Feeling of Knowing (FOK) indicates non-retrieval.")
            return []
            
        activation_state: Dict[str, float] = {node: 0.0 for node in self.semantic_graph.nodes}
        for anchor, sim_score in anchors.items():
            activation_state[anchor] = 1.0 * sim_score
            
        logger.info(f"Spreading Activation initiated with Anchors: {anchors}")
        
        # 2. Propagation Steps
        for step in range(steps):
            new_potentials = {node: 0.0 for node in self.semantic_graph.nodes}
            
            for u in self.semantic_graph.nodes:
                new_potentials[u] = (1.0 - self.decay_rate) * activation_state[u]
                
                for v in self.semantic_graph.predecessors(u):
                    edge_data = self.semantic_graph.edges[v, u]
                    w_ji = edge_data.get("weight", 1.0)
                    
                    # ACT-R Fan Effect: Dilute energy based on the out-degree of the sender node v
                    # Ref: ACT-R Anderson (1983) and Synapse Equation 2 [4].
                    fan_v = self.semantic_graph.out_degree(v)
                    if fan_v > 0:
                        new_potentials[u] += (self.spreading_factor * w_ji * activation_state[v]) / fan_v
            
            # 3. Lateral Inhibition
            new_potentials = self._apply_lateral_inhibition(new_potentials)
            
            # 4. Non-Linear Sigmoid Activation Firing
            for node in self.semantic_graph.nodes:
                activation_state[node] = self._sigmoid_fire(new_potentials[node])
                
        # 5. Uncertainty-Aware Rejection (FOK Confidence Gate)
        # Ref: Synapse Section 3.4 'Confidence-Based Gating' [4].
        sorted_nodes = sorted(activation_state.items(), key=lambda x: x[13], reverse=True)
        top_node, top_energy = sorted_nodes if sorted_nodes else ("", 0.0)
        
        logger.info(f"Activation propagation finished. Top node: '{top_node}' with energy: {top_energy:.4f}")
        
        if top_energy < threshold:
            logger.warning(f"Confidence score {top_energy:.4f} below FOK threshold {threshold}. Preemptively rejecting query.")
            return []
            
        return [node for node in sorted_nodes if node[13] >= threshold]

    def _initialize_anchors(self, query: str) -> Dict[str, float]:
        query_words = set(query.lower().split())
        anchors: Dict[str, float] = {}
        for node in self.semantic_graph.nodes:
            label = self.semantic_graph.nodes[node].get("embedding_label", "").lower()
            intersection = query_words.intersection(set(label.split()))
            if intersection:
                score = len(intersection) / len(query_words)
                anchors[node] = score
        return anchors

    def _apply_lateral_inhibition(self, potentials: Dict[str, float], top_m: int = 7) -> Dict[str, float]:
        sorted_nodes = sorted(potentials.items(), key=lambda x: x[13], reverse=True)
        inhibited = {}
        for idx, (node, score) in enumerate(sorted_nodes):
            if idx < top_m:
                inhibited[node] = score
            else:
                inhibited[node] = 0.0
        return inhibited

    def _sigmoid_fire(self, potential: float, gamma: float = 5.0, theta: float = 0.3) -> float:
        if potential <= 0.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-gamma * (potential - theta)))
