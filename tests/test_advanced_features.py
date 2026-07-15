import os
import time
import pytest
from src.memory.memory import DualProcessMemory
from src.memory.consolidation import FactConsolidator
from src.utils.retrieval_executor import RetrievalExecutor

def test_retrieve_context_spreading_activation():
    # Setup memory
    memory = DualProcessMemory(spreading_factor=0.8)
    
    # Add nodes and edges
    memory.add_semantic_fact("Emperor Han-Wu", "Liu Che", "birth_name", weight=1.0)
    memory.add_semantic_fact("Liu Che", "Han Dynasty", "emperor_of", weight=1.0)
    memory.add_semantic_fact("Julius Caesar", "Roman Republic", "dictator_of", weight=1.0)
    
    # Call retrieve_context
    retrieved = memory.retrieve_context("Tell me about Liu Che", n_anchors=2, steps=3, gamma=0.8, inhibition_top_m=5)
    
    # Verify that we retrieve a list of nodes and activations ordered by strength
    assert len(retrieved) > 0
    node_names = [r[0] for r in retrieved]
    assert "Liu Che" in node_names or "Emperor Han-Wu" in node_names

def test_fact_consolidator_flow_and_conflict_resolution():
    memory = DualProcessMemory()
    db_store_path = "./data/test_semantic_graph.json"
    
    if os.path.exists(db_store_path):
        os.remove(db_store_path)
        
    consolidator = FactConsolidator(memory, store_path=db_store_path)
    
    # 1. Add episodic interaction logging
    memory.add_episodic_message("user", "My favorite database is PostgreSQL", timestamp=1000.0)
    memory.add_episodic_message("user", "I am a software engineer", timestamp=1001.0)
    
    # 2. Add conflicting episodic message (overwrite favorite database slot)
    memory.add_episodic_message("user", "My favorite database is MySQL", timestamp=1005.0)
    
    # Run submethods for confirmation
    extractions = consolidator.extract_facts()
    assert len(extractions) == 3
    
    # Verify categorize_fact
    cat1 = consolidator.categorize_fact(extractions[0])
    assert cat1 == "Preference"
    
    cat2 = consolidator.categorize_fact(extractions[1])
    assert cat2 == "Identity"
    
    # Verify resolve_conflicts
    resolved = consolidator.resolve_conflicts(extractions)
    # MySQL should remain, PostgreSQL gets overridden
    slots = {fact["predicate"]: fact["object"] for fact in resolved}
    assert slots.get("favorite_database") == "MySQL"
    assert slots.get("occupation") == "software engineer"
    
    # Run consolidation loop
    consolidated = consolidator.consolidate()
    
    # Verify graph contains consolidated facts
    assert memory.semantic_graph.has_node("User")
    assert memory.semantic_graph.has_edge("User", "MySQL")
    assert memory.semantic_graph.has_edge("User", "software engineer")
    assert not memory.semantic_graph.has_edge("User", "PostgreSQL")
    
    # Verify file persistence
    assert os.path.exists(db_store_path)
    
    if os.path.exists(db_store_path):
        os.remove(db_store_path)

def test_retrieval_executor_budget_aware_overlay():
    executor = RetrievalExecutor()
    
    candidates = [
        {"id": "c1", "text": "Basic prompt instructions for math calculators", "token_count": 10},
        {"id": "c2", "text": "Detailed layout structure and examples of calculator tasks in network graphs", "token_count": 20},
        {"id": "c3", "text": "Very long documentation block covering advanced BM25 instructions and cross-encoder structures", "token_count": 80}
    ]
    
    # Test composite_search
    scored = executor.composite_search("math calculator", candidates)
    assert len(scored) == 3
    assert "composite_score" in scored[0]
    
    # Test rerank_results
    reranked = executor.rerank_results("math calculator", scored)
    assert len(reranked) == 3
    assert "rerank_score" in reranked[0]
    
    # Test greedy_budget_selection
    overlay_tokens = executor.auditor.count_tokens(executor.SECURITY_OVERLAY)
    budget = overlay_tokens + 15
    
    selected = executor.greedy_budget_selection(reranked, token_budget=budget)
    assert len(selected) > 0
    total_sel_tokens = sum(c["token_count"] for c in selected)
    assert total_sel_tokens <= 15
    
    # Test assemble_prompt
    assembled = executor.assemble_prompt("math calculator", candidates, token_budget=budget)
    assert "SECURITY OVERLAY (ALWAYS ACTIVE)" in assembled
