import logging
import math
import re
import collections
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import networkx as nx

from src.memory.memory import DualProcessMemory

logger = logging.getLogger("ThinkLM.SpreadingActivation")


class SpreadingActivationEngine:
    """
    SpreadingActivationEngine implements the Synapse Spreading Activation retrieval mechanism.
    It traverses a semantic knowledge graph (networkx.DiGraph) to find multi-hop,
    implicitly related context derived from a search query.
    """

    def __init__(
        self,
        semantic_graph: nx.DiGraph,
        emb_model: Optional[Any] = None,
        T: int = 3,
        rho: float = 0.05,
        M: int = 7,
        tau: float = 0.12,
        k: float = 5.0,
        x0: float = 0.5,
        spreading_factor: float = 0.8,
    ):
        """
        Initializes the spreading activation engine.

        Args:
            semantic_graph: Reference to the shared networkx.DiGraph to query.
            emb_model: Dense embedding model (e.g. SentenceTransformer) injected as a dependency.
            T: Number of propagation cycles/steps. Default is 3.
            rho: Global decay rate per step (ρ). Default is 0.05.
            M: Lateral inhibition top-M cutoff limit. Default is 7.
            tau: Feeling of Knowing (FOK) threshold (τ). Default is 0.12.
            k: Sigmoid steepness parameter. Default is 5.0.
            x0: Sigmoid midpoint parameter. Default is 0.5.
            spreading_factor: Optional scaling constant for propagation transfer. Default is 0.8.
        """
        self.semantic_graph = semantic_graph
        self.emb_model = emb_model
        self.T = T
        self.rho = rho
        self.M = M
        self.tau = tau
        self.k = k
        self.x0 = x0
        self.spreading_factor = spreading_factor

    def activate(self, query: str) -> List[Tuple[str, float]]:
        """
        Orchestrates the entire activation workflow:
        1. Find query anchor nodes via hybrid search (BM25 + Cosine similarity).
        2. Perform exactly T propagation loop steps (fan-effect, global decay, lateral inhibition).
        3. Apply sigmoid firing function to active nodes.
        4. Apply FOK threshold gate and return sorted results.

        Args:
            query (str): Lexical/semantic query string.

        Returns:
            List[Tuple[str, float]]: List of (node_id, fired_score) tuples, sorted descending.
        """
        if not self.semantic_graph or len(self.semantic_graph) == 0:
            logger.info("Semantic graph is empty. Activation returning empty results.")
            return []

        # Step 1: Initialize anchors
        initial_energy = self._find_anchor_nodes(query)
        if not initial_energy:
            logger.info("No query anchor nodes found in graph.")
            return []

        # Step 2: Propagate energy over T steps
        final_energy = self._propagate(initial_energy)

        # Step 3 & 4: Applying sigmoid and filtering by FOK threshold (tau)
        fired_nodes = []
        for node, score in final_energy.items():
            if score > 0.0:
                f_x = self._sigmoid(score)
                if f_x > self.tau:
                    fired_nodes.append((node, f_x))

        # Sort descending by fired_score
        fired_nodes.sort(key=lambda x: x[1], reverse=True)
        return fired_nodes

    def _find_anchor_nodes(self, query: str, top_k: int = 3) -> Dict[str, float]:
        """
        Finds the initial matching anchor nodes by combining:
        - Sparse lexical score via basic BM25-like/word-overlap search.
        - Dense semantic representation via Cosine Similarity between embeddings.

        Chosen fusion method: Equal weighted average of normalized sparse score and
        normalized dense similarity score.

        Args:
            query (str): query string.
            top_k (int): maximum number of anchor nodes to return.

        Returns:
            Dict[str, float]: Initial energy state dictionary mapping anchor node ID to 1.0.
                              All non-anchor nodes are implicitly 0.0.
        """
        query_words = re.findall(r"\w+", query.lower())
        if not query_words:
            return {}

        nodes_list = list(self.semantic_graph.nodes)
        if not nodes_list:
            return {}

        # 1. Compute Lexical/Sparse scores (BM25 fallback or actual BM25Okapi)
        texts = [
            self.semantic_graph.nodes[n].get("embedding_label", str(n))
            for n in nodes_list
        ]
        tokenized_corpus = [re.findall(r"\w+", t.lower()) for t in texts]

        bm25_scores = np.zeros(len(nodes_list))
        try:
            from rank_bm25 import BM25Okapi
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = np.array(bm25.get_scores(query_words))
        except Exception:
            # Fallback simple Jaccard / Token overlap
            logger.debug("rank_bm25 fallback lexical search active.")
            q_set = set(query_words)
            for idx, tc in enumerate(tokenized_corpus):
                tc_set = set(tc)
                if tc_set:
                    intersect = q_set.intersection(tc_set)
                    bm25_scores[idx] = len(intersect) / len(q_set)

        max_bm25 = np.max(bm25_scores) if len(bm25_scores) > 0 else 0.0
        bm25_norm = (
            bm25_scores / max_bm25 if max_bm25 > 0.0 else bm25_scores
        )

        # 2. Compute Dense Similarity scores
        dense_scores = np.zeros(len(nodes_list))
        if self.emb_model is not None:
            try:
                query_emb = self.emb_model.encode(query, convert_to_numpy=True)
                q_norm = np.linalg.norm(query_emb)

                for idx, node in enumerate(nodes_list):
                    node_data = self.semantic_graph.nodes[node]
                    node_emb = node_data.get("embedding")
                    if node_emb is None:
                        # Generate on the fly
                        node_text = node_data.get("embedding_label", str(node))
                        node_emb = self.emb_model.encode(
                            node_text, convert_to_numpy=True
                        )

                    n_norm = np.linalg.norm(node_emb)
                    if q_norm > 0 and n_norm > 0:
                        sim = np.dot(query_emb, node_emb) / (q_norm * n_norm)
                        # scale cosine [-1, 1] -> [0, 1]
                        dense_scores[idx] = (sim + 1.0) / 2.0
            except Exception as e:
                logger.warning(
                    f"Dense scoring failed: {e}. Falling back to lexical resemblance."
                )
                dense_scores = self._fallback_dense_resemblance(query, texts)
        else:
            dense_scores = self._fallback_dense_resemblance(query, texts)

        # 3. Fuse Scores
        fusion_scores = {}
        for idx, node in enumerate(nodes_list):
            # Combined relevance score (alpha = 0.5)
            f_score = 0.5 * dense_scores[idx] + 0.5 * bm25_norm[idx]
            fusion_scores[node] = float(f_score)

        # Filter nodes with score > 0.0 and sort
        valid_anchors = [
            (node, score)
            for node, score in fusion_scores.items()
            if score > 0.0
        ]
        valid_anchors.sort(key=lambda x: x[1], reverse=True)

        selected = valid_anchors[:top_k]
        energy = collections.defaultdict(float)
        for node, _ in selected:
            energy[node] = 1.0

        return energy

    def _fallback_dense_resemblance(
        self, query: str, texts: List[str]
    ) -> np.ndarray:
        q_words = set(re.findall(r"\w+", query.lower()))
        scores = []
        for text in texts:
            t_words = set(re.findall(r"\w+", text.lower()))
            if not q_words or not t_words:
                scores.append(0.0)
                continue
            intersection = q_words.intersection(t_words)
            scores.append(len(intersection) / len(q_words))
        return np.array(scores)

    def _propagate(self, energy: Dict[str, float]) -> Dict[str, float]:
        """
        Runs exactly T propagation steps. At each t step, a deep snapshot config is built
        so updates are synchronous and order-independent.

        Each step consists of:
        (a) Compute outbound diluted contributions from currently active nodes.
        (b) Sum those contributions into receiving nodes.
        (c) Apply global decay to the resulting total energy at every node.
        (d) Apply lateral inhibition filter (keep top-M nodes, reset rest to 0.0).
        """
        current_energy = collections.defaultdict(float, energy)

        # Ensure all nodes in graph are represented in state representation
        for node in self.semantic_graph.nodes:
            if node not in current_energy:
                current_energy[node] = 0.0

        for step in range(self.T):
            step_contributions = collections.defaultdict(float)

            # Step (a) & (b): Compute outbound diluted flow towards neighbors
            for u in self.semantic_graph.nodes:
                val_u = current_energy[u]
                if val_u <= 0.0:
                    continue

                out_deg = self.semantic_graph.out_degree(u)
                if out_deg > 0:
                    # Diluted potential: scaled by degree and weight multiplier representation
                    diluted = val_u / float(out_deg)
                    for s in self.semantic_graph.successors(u):
                        edge_data = self.semantic_graph[u][s]
                        weight = edge_data.get("weight", 1.0)
                        # Scale optionally by edge weight and spreading factor
                        flow = diluted * weight * self.spreading_factor
                        step_contributions[s] += flow

            # Step (c): Apply global decay (1 - rho) to the accumulated total energy values
            next_step_energy = collections.defaultdict(float)
            for node in self.semantic_graph.nodes:
                total_energy = current_energy[node] + step_contributions[node]
                next_step_energy[node] = (1.0 - self.rho) * total_energy

            # Step (d): Lateral Inhibition - Sort & keep top-M = 7 active elements
            sorted_nodes = sorted(
                next_step_energy.items(), key=lambda x: x[1], reverse=True
            )

            current_energy = collections.defaultdict(float)
            for idx, (node, score) in enumerate(sorted_nodes):
                if idx < self.M and score > 0.0:
                    current_energy[node] = score
                else:
                    current_energy[node] = 0.0

            logger.debug(
                f"Activation Step {step+1} finished. Top items: "
                f"{[(k, round(v, 4)) for k, v in current_energy.items() if v > 0.0]}"
            )

        return current_energy

    def _sigmoid(self, x: float) -> float:
        """
        Applies a numerically stable sigmoid fire logic.
        f(x) = 1 / (1 + exp(-k * (x - x0)))
        """
        # Limit exponential input range to prevent overflows
        val = -self.k * (x - self.x0)
        if val >= 0:
            return 1.0 / (1.0 + math.exp(val))
        else:
            e = math.exp(-val)
            return e / (1.0 + e)


if __name__ == "__main__":
    print("--- Running Synapse Spreading Activation Engine Demo ---")

    # 1. Setup a toy DualProcessMemory and add facts
    mem = DualProcessMemory(episodic_limit=5)
    mem.add_semantic_fact("Liu Che", "Han Dynasty", "emperor_of", weight=1.0)
    mem.add_semantic_fact("Emperor Han-Wu", "Liu Che", "birth_name", weight=1.0)
    mem.add_semantic_fact("Liu Che", "Changan", "residence", weight=0.8)
    mem.add_semantic_fact("Changan", "Han Empire", "capital_of", weight=1.0)
    mem.add_semantic_fact("Julius Caesar", "Roman Republic", "dictator_of", weight=0.9)
    mem.add_semantic_fact("Roman Republic", "Rome", "capital_of", weight=1.0)

    # 2. Instantiate SpreadingActivationEngine
    engine = SpreadingActivationEngine(
        semantic_graph=mem.semantic_graph,
        T=3,
        rho=0.05,
        M=7,
        tau=0.12,
        k=5.0,
        x0=0.5,
    )

    # 3. Run activate on query
    query = "Where did Liu Che live?"
    print(f"Target query: '{query}'")

    # Run anchor checking first
    anchors = engine._find_anchor_nodes(query)
    print(f"Detected anchors: {dict(anchors)}")

    # Execute activate
    retrieved = engine.activate(query)
    print(f"\nFinal activated nodes clearing threshold (tau=0.12):\n{retrieved}")

    # Assertion checks
    # Assert loop properties
    assert len(retrieved) > 0, "Should retrieve at least one node"
    
    # Assert top-limit restriction (not more than M=7 nodes returned)
    assert len(retrieved) <= 7, "Output exceeds lateral inhibition limit"

    # Assert that all scores clear FOK threshold tau = 0.12
    for node, score in retrieved:
        assert score > 0.12, f"Node {node} failed threshold check with score {score}"

    print("\nhybrid anchor propagation tests successfully completed!")
