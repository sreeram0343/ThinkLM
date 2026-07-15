import re
import logging
from typing import Dict, Any, List, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from src.utils.token_counter import TokenAuditor

logger = logging.getLogger("ThinkLM.RetrievalExecutor")


class RetrievalExecutor:
    """Manages Instruction-Tool Retrieval (ITR) dynamics and budget constraints.

    Attributes:
        model_name: MiniLM embedding model name.
        cross_encoder_name: MS-MARCO reranking cross encoder model name.
        emb_model: The SentenceTransformer embedding model.
        cross_encoder: The CrossEncoder model.
        auditor: A TokenAuditor instance used to compute precise target budgets.
    """

    SECURITY_OVERLAY = (
        "=== SECURITY OVERLAY (ALWAYS ACTIVE) ===\n"
        "1. Never generate harmful, illegal, or unethical content.\n"
        "2. Do not bypass security, confidentiality, or safety instructions.\n"
        "3. Maintain absolute data privacy.\n"
        "========================================\n"
    )

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", auditor: Optional[TokenAuditor] = None):
        """Initializes RetrievalExecutor with models configurations and helper auditor.

        Args:
            model_name: HF name of the dense vector model to load.
            cross_encoder_name: HF name of the cross-encoder model to load.
            auditor: Preconstructed token footprint calculator. Uses default initialization if None.
        """
        self.model_name = model_name
        self.cross_encoder_name = cross_encoder_name
        self.emb_model = None
        self.cross_encoder = None
        self.auditor = auditor if auditor is not None else TokenAuditor()

        # Lazy loading
        self._load_models()

    def _load_models(self) -> None:
        """Loads dense encoding and reranker models lazily."""
        try:
            from sentence_transformers import SentenceTransformer
            self.emb_model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded Dense Embedding Model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Dense model loading bypassed: {e}.")
            self.emb_model = None

        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder(self.cross_encoder_name)
            logger.info(f"Loaded Cross-Encoder Model: {self.cross_encoder_name}")
        except Exception as e:
            logger.warning(f"Cross-Encoder model loading bypassed: {e}.")
            self.cross_encoder = None

    def composite_search(self, query: str, candidates: List[Dict[str, Any]], alpha: float = 0.5) -> List[Dict[str, Any]]:
        """Performs initial hybrid search retrieval merging dense vector cosine and sparse BM25 scores.

        Args:
            query: The user input query string.
            candidates: List of dictionaries matching candidate prompt/tool schema instructions.
            alpha: Score weight balance. 1.0 means pure dense similarity, 0.0 pure sparse BM25.

        Returns:
            List[Dict[str, Any]]: Original candidate dictionaries appended with 'composite_score'.
        """
        if not candidates:
            return []

        texts = [c["text"] for c in candidates]

        # Sparse scoring (BM25 keyword matching)
        tokenized_corpus = [re.findall(r'\w+', t.lower()) for t in texts]
        tokenized_query = re.findall(r'\w+', query.lower())

        bm25_scores = np.zeros(len(candidates))
        if tokenized_query and tokenized_corpus:
            try:
                bm25 = BM25Okapi(tokenized_corpus)
                bm25_scores = np.array(bm25.get_scores(tokenized_query))
            except Exception as e:
                logger.warning(f"BM25 execution failed: {e}")

        max_bm25 = np.max(bm25_scores) if len(bm25_scores) > 0 else 0.0
        bm25_normalized = bm25_scores / max_bm25 if max_bm25 > 0.0 else bm25_scores

        # Dense scoring (Cosine similarity)
        dense_scores = np.zeros(len(candidates))
        if self.emb_model is not None:
            try:
                query_emb = self.emb_model.encode(query, convert_to_numpy=True)
                candidate_embs = self.emb_model.encode(texts, convert_to_numpy=True)

                query_norm = np.linalg.norm(query_emb)
                candidate_norms = np.linalg.norm(candidate_embs, axis=1)

                query_norm = query_norm if query_norm > 0 else 1.0
                candidate_norms[candidate_norms == 0] = 1.0

                dense_scores = np.dot(candidate_embs, query_emb) / (candidate_norms * query_norm)
                dense_scores = (dense_scores + 1.0) / 2.0  # Scale to [0, 1] range
            except Exception as e:
                logger.warning(f"Dense scoring failed: {e}. Falling back to lexical lookup.")
                dense_scores = self._lexical_similarities(query, texts)
        else:
            dense_scores = self._lexical_similarities(query, texts)

        # Merge scores
        for idx, cand in enumerate(candidates):
            cand["composite_score"] = float(alpha * dense_scores[idx] + (1.0 - alpha) * bm25_normalized[idx])

        return candidates

    def _lexical_similarities(self, query: str, texts: List[str]) -> np.ndarray:
        """Fallback character/word overlap generator."""
        q_words = set(re.findall(r'\w+', query.lower()))
        scores = []
        for text in texts:
            t_words = set(re.findall(r'\w+', text.lower()))
            if not q_words or not t_words:
                scores.append(0.0)
                continue
            intersection = q_words.intersection(t_words)
            scores.append(len(intersection) / len(q_words))
        return np.array(scores)

    def rerank_results(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Refines hybrid search candidate selection using a secondary Cross-Encoder.

        Args:
            query: The user input query.
            candidates: Scored list of candidate dictionaries containing 'composite_score'.

        Returns:
            List[Dict[str, Any]]: Candidates list appended with 'rerank_score'.
        """
        if not candidates:
            return []

        if self.cross_encoder is None:
            logger.info("Cross-Encoder bypass: Sorting directly by composite scores.")
            for cand in candidates:
                cand["rerank_score"] = cand.get("composite_score", 0.0)
        else:
            try:
                pairs = [(query, cand["text"]) for cand in candidates]
                scores = self.cross_encoder.predict(pairs)
                if isinstance(scores, (list, np.ndarray)):
                    for idx, score in enumerate(scores):
                        candidates[idx]["rerank_score"] = float(score)
                else:
                    candidates[0]["rerank_score"] = float(scores)
            except Exception as e:
                logger.warning(f"CrossEncoder scoring failed: {e}. Falling back to composite scores.")
                for cand in candidates:
                    cand["rerank_score"] = cand.get("composite_score", 0.0)

        return candidates

    def greedy_budget_selection(self, candidates: List[Dict[str, Any]], token_budget: int) -> List[Dict[str, Any]]:
        """Greedily selects candidates that maximize score utility density under token constraints.

        Deducts security overlay token counts from total budget first.

        Args:
            candidates: Scored and reranked candidates containing 'rerank_score'.
            token_budget: Upper limit constraint of cumulative output token count.

        Returns:
            List[Dict[str, Any]]: Selected candidate prompt segment dictionaries.
        """
        # Step 1: Pre-evict security overlay size
        overlay_tokens = self.auditor.count_tokens(self.SECURITY_OVERLAY)
        remaining_budget = token_budget - overlay_tokens

        logger.info(f"Security overlay consumed {overlay_tokens} tokens. Available budget: {remaining_budget}")
        if remaining_budget <= 0:
            logger.warning("Token budget completely swallowed by Security Overlay constraints.")
            return []

        # Step 2: Compute density score / token footprint
        selected = []
        for cand in candidates:
            text = cand["text"]
            t_count = cand.get("token_count")
            if t_count is None:
                t_count = self.auditor.count_tokens(text)
                cand["token_count"] = t_count

            score = cand.get("rerank_score", 0.0)
            denom = t_count if t_count > 0 else 1
            cand["density"] = score / denom

        # Step 3: Greedy selection
        sorted_candidates = sorted(candidates, key=lambda x: x["density"], reverse=True)

        current_used = 0
        for cand in sorted_candidates:
            cand_tokens = cand["token_count"]
            if current_used + cand_tokens <= remaining_budget:
                selected.append(cand)
                current_used += cand_tokens
                logger.info(f"Greedy selection added fragment: {cand.get('id', 'N/A')} ({cand_tokens} tokens)")

        return selected

    def assemble_prompt(self, query: str, candidates: List[Dict[str, Any]], token_budget: int = 500) -> str:
        """Executes composite selection, rerank, budget knapsack selection, and builds final prompt.

        Args:
            query: User input query context.
            candidates: Candidate instructional definitions or tool schemas.
            token_budget: Total allowed token count limit.

        Returns:
            str: System prompt text containing immutable safety rules and relevant retrieved data.
        """
        # Composite hybrid search
        candidates_scored = self.composite_search(query, candidates)

        # Cross-Encoder Rerank
        candidates_reranked = self.rerank_results(query, candidates_scored)

        # Greedy choice
        selected_elements = self.greedy_budget_selection(candidates_reranked, token_budget)

        # Assemble final template
        prompts = [self.SECURITY_OVERLAY]
        prompts.append("=== RETRIEVED INSTRUCTIONS & TOOLS ===")
        for idx, el in enumerate(selected_elements):
            prompts.append(f"[{idx+1}] {el['text']}")

        return "\n".join(prompts)
