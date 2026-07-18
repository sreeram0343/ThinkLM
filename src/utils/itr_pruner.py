import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Callable
from rank_bm25 import BM25Okapi

logger = logging.getLogger("ThinkLM.ITRPruner")


@dataclass
class ToolSchema:
    name: str
    schema: dict  # JSON schema / function-calling spec
    description: str  # text used for BM25 indexing


@dataclass
class PromptFragment:
    id: str
    text: str  # the instruction fragment itself, also used for BM25 indexing


@dataclass
class PruneResult:
    selected_tools: List[ToolSchema]
    selected_fragments: List[PromptFragment]
    tool_scores: Dict[str, float]      # RAW BM25 scores, for debugging
    fragment_scores: Dict[str, float]  # RAW BM25 scores, for debugging
    total_tokens: int
    discovery_flag: bool               # triggers expanded search if target tool missing


class ITRPruner:
    """
    Instruction-Tool Retrieval (ITR) Prompt Pruner.
    Selects only the most relevant tool schemas and system prompt fragments
    for incoming queries based on lexical overlap using independent BM25 indices.
    """

    def __init__(
        self,
        KB: int = 2,
        KA: int = 3,
        TAU_GATE: float = 0.12,
        drop_fragments_first: bool = True,
        tokenizer_fn: Optional[Callable[[str], int]] = None,
    ):
        """
        Initializes the pruner.

        Args:
            KB (int): Max number of tool schemas to retain. Default is 2.
            KA (int): Max number of system prompt fragments to retain. Default is 3.
            TAU_GATE (float): Confidence threshold for discovery gating (τ_gate). Default is 0.12.
            drop_fragments_first (bool): If True, lowest-scoring fragments are dropped before tools when budget is exceeded.
            tokenizer_fn (Optional[Callable[[str], int]]): Pluggable token-counting function.
                                                           If None, instantiates TokenAuditor or falls back to word splitting.
        """
        self.KB = KB
        self.KA = KA
        self.TAU_GATE = TAU_GATE
        self.drop_fragments_first = drop_fragments_first

        # Injected tokenizer
        if tokenizer_fn is not None:
            self.count_tokens = tokenizer_fn
        else:
            self.count_tokens = self._setup_default_token_counter()

        self.tools: List[ToolSchema] = []
        self.fragments: List[PromptFragment] = []

        self.tool_bm25: Optional[BM25Okapi] = None
        self.fragment_bm25: Optional[BM25Okapi] = None

    def _setup_default_token_counter(self) -> Callable[[str], int]:
        """Loads TokenAuditor or falls back to basic whitespace splitting."""
        try:
            from src.utils.token_counter import TokenAuditor
            auditor = TokenAuditor()
            return auditor.count_tokens
        except Exception as e:
            logger.warning(f"Could not load TokenAuditor: {e}. Falling back to basic word counter.")
            return lambda text: len(text.split())

    def build_index(
        self, tool_schemas: List[ToolSchema], prompt_fragments: List[PromptFragment]
    ) -> None:
        """
        populates the pruner registry and builds distinct lexical indices.

        Args:
            tool_schemas: List of available ToolSchema specifications.
            prompt_fragments: List of prompt instruction/rule fragments.
        """
        self.tools = list(tool_schemas)
        self.fragments = list(prompt_fragments)

        # Build tools index
        if self.tools:
            tool_corpus_tokens = [self.tokenize(t.description) for t in self.tools]
            self.tool_bm25 = BM25Okapi(tool_corpus_tokens)
        else:
            self.tool_bm25 = None

        # Build fragments index
        if self.fragments:
            frag_corpus_tokens = [self.tokenize(f.text) for f in self.fragments]
            self.fragment_bm25 = BM25Okapi(frag_corpus_tokens)
        else:
            self.fragment_bm25 = None

        logger.info(
            f"Indexed {len(self.tools)} tool schemas and {len(self.fragments)} prompt fragments."
        )

    def rebuild_index(self) -> None:
        """Rebuilds the BM25 indexes with the current set of tools and fragments."""
        self.build_index(self.tools, self.fragments)

    def tokenize(self, text: str) -> List[str]:
        """
        Simple regex tokenizer matching alphanumeric words, excluding common stopwords.
        """
        tokens = re.findall(r"\w+", text.lower())
        stopwords = {
            "a", "an", "the", "and", "or", "but", "if", "then", "of", "to", "for",
            "with", "in", "on", "at", "by", "from", "as", "about", "is", "are",
            "was", "were", "be", "been", "being", "this", "that", "it", "they",
            "he", "she", "you", "me", "my"
        }
        return [t for t in tokens if t not in stopwords]


    def serialize_tool(self, tool: ToolSchema) -> str:
        """Serializes the ToolSchema into a structured format for prompt injection."""
        return json.dumps(
            {
               "name": tool.name,
               "description": tool.description,
               "parameters": tool.schema,
            },
            indent=2,
        )

    def serialize_fragment(self, fragment: PromptFragment) -> str:
        """Serializes the PromptFragment into a structured format for prompt injection."""
        return fragment.text

    def prune(self, query: str, token_budget: int = 400) -> PruneResult:
        """
        Filters tools and prompt fragments for relevance, capping size to token_budget.
        If confidence falls below TAU_GATE (0.12 after saturating normalization),
        signals need for tool discovery.

        Args:
            query (str): The search query.
            token_budget (int): Maximum allotted token footprint constraint.

        Returns:
            PruneResult: Dataclass summarizing chosen assets, scores, token counts, and discovery flag.
        """
        q_tokens = self.tokenize(query)

        # 1. Score and sort tools
        tool_scores = {}
        sorted_tools = []
        if self.tool_bm25 is not None and self.tools:
            raw_scores = self.tool_bm25.get_scores(q_tokens)
            for idx, tool in enumerate(self.tools):
                raw_score = float(raw_scores[idx])
                tool_scores[tool.name] = raw_score
                sorted_tools.append((tool, raw_score))
            sorted_tools.sort(key=lambda x: x[1], reverse=True)
        else:
            # Empty scores
            sorted_tools = [(t, 0.0) for t in self.tools]

        # 2. Score and sort fragments
        frag_scores = {}
        sorted_frags = []
        if self.fragment_bm25 is not None and self.fragments:
            raw_scores = self.fragment_bm25.get_scores(q_tokens)
            for idx, frag in enumerate(self.fragments):
                raw_score = float(raw_scores[idx])
                frag_scores[frag.id] = raw_score
                sorted_frags.append((frag, raw_score))
            sorted_frags.sort(key=lambda x: x[1], reverse=True)
        else:
            sorted_frags = [(f, 0.0) for f in self.fragments]

        # 3. Select Initial Candidates
        selected_tools = [item[0] for item in sorted_tools[:self.KB]]
        selected_fragments = [item[0] for item in sorted_frags[:self.KA]]

        # Calculate individual token sizes
        tool_sizes = {t.name: self.count_tokens(self.serialize_tool(t)) for t in selected_tools}
        frag_sizes = {f.id: self.count_tokens(self.serialize_fragment(f)) for f in selected_fragments}

        total_tokens = sum(tool_sizes.values()) + sum(frag_sizes.values())

        # 4. Enforce Token Budget
        if total_tokens > token_budget:
            logger.info(
                f"Initial token size {total_tokens} exceeds target budget {token_budget}. Commencing pruning."
            )

            # Drop items until total_tokens fits budget
            while total_tokens > token_budget and (selected_tools or selected_fragments):
                if selected_fragments and (self.drop_fragments_first or not selected_tools):
                    # Drop lowest-scoring fragment (which is at the end of the list)
                    dropped = selected_fragments.pop()
                    total_tokens -= frag_sizes[dropped.id]
                    logger.info(f"Dropped fragment '{dropped.id}' under budget constraint.")
                elif selected_tools:
                    # Drop lowest-scoring tool
                    dropped = selected_tools.pop()
                    total_tokens -= tool_sizes[dropped.name]
                    logger.info(f"Dropped tool '{dropped.name}' under budget constraint.")

            # Truncation logic as a last resort: if we have 1 fragment remaining but it still exceeds budget
            if total_tokens > token_budget and selected_fragments:
                frag = selected_fragments[0]
                words = frag.text.split()
                left, right = 0, len(words)
                best_text = ""
                while left <= right:
                    mid = (left + right) // 2
                    candidate_text = " ".join(words[:mid])
                    tok_count = self.count_tokens(candidate_text)
                    if tok_count <= token_budget:
                        best_text = candidate_text
                        left = mid + 1
                    else:
                        right = mid - 1
                frag.text = best_text
                total_tokens = self.count_tokens(best_text)
                logger.info(f"As a last resort, truncated fragment '{frag.id}' to fit budget.")

            if total_tokens > token_budget:
                logger.warning(
                    f"Warning: Even after complete pruning/trimming, footprint ({total_tokens}) exceeds budget ({token_budget})."
                )

        # 5. Discovery Gating Fallback Check
        # Confidence Metric: Max normalized score among all tools
        max_raw_tool_score = 0.0
        if sorted_tools:
            max_raw_tool_score = max(score for _, score in sorted_tools)

        # Design Decision: We use a saturating transform to normalize raw BM25 scores:
        # s_norm = raw / (raw + 2.0). Raw scores of 0 map to 0.0, and high scores saturate towards 1.0.
        # This keeps the confidence measure bounded within [0, 1) consistently.
        normalized_confidence = max_raw_tool_score / (max_raw_tool_score + 2.0) if max_raw_tool_score > 0.0 else 0.0

        discovery_flag = normalized_confidence < self.TAU_GATE
        if discovery_flag:
            logger.info(
                f"Confidence {normalized_confidence:.4f} is below Gating threshold {self.TAU_GATE}. "
                f"Discovery flag set to True."
            )

        return PruneResult(
            selected_tools=selected_tools,
            selected_fragments=selected_fragments,
            tool_scores=tool_scores,
            fragment_scores=frag_scores,
            total_tokens=total_tokens,
            discovery_flag=discovery_flag,
        )

    def assemble_prompt(
        self, prune_result: PruneResult, base_system_prompt: str = ""
    ) -> str:
        """
        Assembles a ready-to-use prompt containing:
        - Base system prompt
        - Selected prompt fragments
        - Serialized selected tool schemas
        - Conditional [TOOL_DISCOVERY] token if discovery flag is true.

        Args:
            prune_result: The result of prune().
            base_system_prompt: Optional base guidelines context.

        Returns:
            str: Assembled prompt markdown block.
        """
        parts = []
        if base_system_prompt:
            parts.append(base_system_prompt.strip())

        if prune_result.selected_fragments:
            parts.append("=== INSTRUCTION FRAGMENTS ===")
            for f in prune_result.selected_fragments:
                parts.append(f.text.strip())

        if prune_result.selected_tools:
            parts.append("=== AVAILABLE TOOLS ===")
            for t in prune_result.selected_tools:
                parts.append(self.serialize_tool(t))

        if prune_result.discovery_flag:
            # Specialized fallback token
            parts.append("[TOOL_DISCOVERY]")

        return "\n\n".join(parts)


if __name__ == "__main__":
    print("--- Running ITRPruner Demo ---")

    # 1. Register Toy tools
    toy_tools = [
        ToolSchema(
            name="calculator",
            schema={"type": "object", "properties": {"expression": {"type": "string"}}},
            description="Evaluates mathematical arithmetic expressions and processes standard calculations.",
        ),
        ToolSchema(
            name="web_search",
            schema={"type": "object", "properties": {"query": {"type": "string"}}},
            description="Searches the web for public online articles, news reports, and webpages.",
        ),
        ToolSchema(
            name="database_lookup",
            schema={"type": "object", "properties": {"client_id": {"type": "integer"}}},
            description="Queries the relational database for specific client information and records.",
        ),
        ToolSchema(
            name="code_executor",
            schema={"type": "object", "properties": {"code": {"type": "string"}}},
            description="Compiles and runs Python code within a safe subprocess sandbox environment.",
        ),
    ]

    # 2. Register Toy prompt fragments
    toy_fragments = [
        PromptFragment(
            id="safety_overlay",
            text="Rule 1: Do not execute dangerous actions or leak confidential parameters.",
        ),
        PromptFragment(
            id="response_format",
            text="Rule 2: Provide short summaries of results in JSON format.",
        ),
        PromptFragment(
            id="math_spec",
            text="Rule 3: Always check for divide by zero before calling the calculator.",
        ),
    ]

    # 3. Instantiate pruner
    pruner = ITRPruner(KB=2, KA=2, TAU_GATE=0.12)
    pruner.build_index(toy_tools, toy_fragments)

    # 4. Query that closely matches tools (should NOT trigger discovery flag)
    query_match = "Do a quick web search on math equations and execute code."
    print(f"\nEvaluating Query 1: '{query_match}'")
    result_match = pruner.prune(query_match, token_budget=500)
    print(f"Selected Tools: {[t.name for t in result_match.selected_tools]}")
    print(f"Selected Fragments: {[f.id for f in result_match.selected_fragments]}")
    print(f"Discovery Flag: {result_match.discovery_flag} (Expected: False)")
    print(f"Total tokens: {result_match.total_tokens}")
    assert len(result_match.selected_tools) <= 2
    assert len(result_match.selected_fragments) <= 2
    assert not result_match.discovery_flag, "Should not trigger discovery flag."

    # 5. Query that matches nothing (should trigger discovery flag)
    query_miss = "Paint a watercolor picture of sunset clouds."
    print(f"\nEvaluating Query 2: '{query_miss}'")
    result_miss = pruner.prune(query_miss, token_budget=500)
    print(f"Discovery Flag: {result_miss.discovery_flag} (Expected: True)")
    assert result_miss.discovery_flag, "Should trigger tool discovery."

    # 6. Test Token budget enforcement pruning
    print(f"\nEvaluating Query 1 with a tight budget (B=60)")
    result_tight = pruner.prune(query_match, token_budget=60)
    print(f"Budget restricted Selected Tools: {[t.name for t in result_tight.selected_tools]}")
    print(f"Budget restricted Selected Fragments: {[f.id for f in result_tight.selected_fragments]}")
    print(f"Tight budget total tokens: {result_tight.total_tokens} (Expected <= 60)")
    assert result_tight.total_tokens <= 60

    print("\nVisual Inspection of Assembled Prompt (High match case):")
    final_prompt = pruner.assemble_prompt(result_match, base_system_prompt="Base System Instructions")
    print(final_prompt)

    print("\nVisual Inspection of Assembled Prompt (Gated miss case):")
    gated_prompt = pruner.assemble_prompt(result_miss, base_system_prompt="Base System Instructions")
    print(gated_prompt)

    print("\nAll ITRPruner demo tests successfully completed!")
