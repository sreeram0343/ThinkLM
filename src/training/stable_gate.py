from typing import List
from pathlib import Path
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class STABLEGate:
    """
    STABLE Gate for preventing catastrophic forgetting.

    Evaluates a LoRA adapter against an anchor dataset before merging.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.device = next(
            model.parameters()
        ).device

        print("=" * 60)
        print(" STABLE Gate Initialized")
        print("=" * 60)
        print(f"Device : {self.device}")
        print("=" * 60)
    def load_anchor_set(
        self,
        anchor_path: str
    ):
        """
        Load the cached SQuAD anchor dataset.

        Returns:
            List of question-answer pairs.
        """

        anchor_path = Path(anchor_path)

        if not anchor_path.exists():
            raise FileNotFoundError(
                f"Anchor set not found: {anchor_path}"
            )

        with open(anchor_path, "r", encoding="utf-8") as f:
            anchors = json.load(f)

        print(f"\nLoaded {len(anchors)} anchor samples.")

        return anchors
    def evaluate_em(
        self,
        anchor_path: str
    ) -> float:
        """
        Evaluate Exact Match accuracy on the anchor set.
        """

        anchors = self.load_anchor_set(anchor_path)[:5]

        correct = 0

        total = len(anchors)

        print("\nEvaluating Anchor Set...\n")

        for sample in anchors:

            prompt = (
                f"Context:\n{sample['context']}\n\n"
                f"Question: {sample['question']}\n"
                "Answer:"
            )

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            gold_answers = [
                ans.lower().strip()
                for ans in sample["answers"]
            ]

            prediction = generated.lower().strip()

            if prediction in gold_answers:
                correct += 1

        em = correct / total

        print(f"\nExact Match: {em:.3f}")

        return em
    def evaluate_em(
        self,
        anchor_path: str,
        max_samples: int = 5
    ) -> float:
        """
        Evaluate Exact Match (EM) accuracy on the anchor set.

        Args:
            anchor_path: Path to the anchor dataset.
            max_samples: Number of samples to evaluate (for faster testing).

        Returns:
            Exact Match (EM) accuracy.
        """

        anchors = self.load_anchor_set(anchor_path)[:max_samples]

        correct = 0

        print("\nEvaluating Exact Match...\n")

        for i, sample in enumerate(anchors, start=1):

            prompt = (
                f"Context:\n{sample['context']}\n\n"
                f"Question: {sample['question']}\n"
                "Answer:"
            )

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            prediction = (
                generated
                .split("\n")[0]
                .strip()
                .lower()
            )
            gold_answers = [
                ans.lower().strip()
                for ans in sample["answers"]
            ]

            is_correct = any(
                answer in prediction
                for answer in gold_answers
            )
            if is_correct:
                correct += 1

            print(f"[{i}]")
            print("Question :", sample["question"])
            print("Prediction:", generated)
            print("Expected :", gold_answers[0])
            print("Correct  :", is_correct)
            print("-" * 60)

        em = correct / len(anchors)

        print(f"\nExact Match Accuracy: {em:.3f}")

        return em
    def compute_em_drop(
        self,
        baseline_em: float,
        candidate_em: float,
    ) -> float:
        """
        Compute the Exact Match (EM) accuracy drop.
        """

        em_drop = baseline_em - candidate_em

        print("\n" + "=" * 60)
        print("EM DROP ANALYSIS")
        print("=" * 60)
        print(f"Baseline EM : {baseline_em:.3f}")
        print(f"Candidate EM: {candidate_em:.3f}")
        print(f"EM Drop     : {em_drop:.3f}")
        print("=" * 60)

        return em_drop


    # ----------------------------------------------------
    # ADD THIS ENTIRE FUNCTION BELOW compute_em_drop()
    # ----------------------------------------------------

    def binary_search_clip(
        self,
        baseline_em: float,
        candidate_em: float,
        threshold: float = 0.07,
        max_steps: int = 5,
    ):
        """
        Binary search for the maximum safe LoRA scale.

        Returns:
            float
        """

        # Step 3: Initialize search range
        low = 0.0
        high = 1.0
        best_scale = 0.0

        # Step 4: Binary search loop
        for step in range(max_steps):

            scale = (low + high) / 2

            # Step 5: Simulate EM drop
            scaled_drop = (
                baseline_em - candidate_em
            ) * scale

            # Step 6: Print progress
            print(
                f"Step {step+1}: "
                f"Scale={scale:.3f} "
                f"Drop={scaled_drop:.3f}"
            )

            # Step 7: Update search bounds
            if scaled_drop <= threshold:

                best_scale = scale
                low = scale

            else:

                high = scale

        # Step 8: Return best scale
        print("\nBest Safe Scale:", best_scale)

        return best_scale
    def gate_merge_with_clip(
        self,
        baseline_em: float,
        candidate_em: float,
        threshold: float = 0.07,
    ):
        """
        Decide whether a LoRA update is safe.
        """

        em_drop = self.compute_em_drop(
            baseline_em,
            candidate_em
        )

        print("\nChecking if clipping is required...\n")

        if em_drop <= threshold:

            print("LoRA update is SAFE.")
            print("Using full scale = 1.0")

            return {
                "accepted": True,
                "scale": 1.0,
                "em_drop": em_drop,
            }

        print("LoRA update exceeds threshold.")
        print("Running binary search...\n")

        scale = self.binary_search_clip(
            baseline_em,
            candidate_em,
            threshold
        )

        return {
            "accepted": False,
            "scale": scale,
            "em_drop": em_drop,
        }