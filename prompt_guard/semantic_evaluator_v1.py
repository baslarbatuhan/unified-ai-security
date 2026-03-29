"""
prompt_guard/semantic_evaluator_v1.py
======================================
Semantic similarity-based prompt injection detector.

Purpose:
    - Encode prompts using sentence-transformer (BGE-M3)
    - Compare against known attack pattern embeddings (cosine similarity)
    - Detect injection attempts that evade rule-based filters
    - Known injection examples MUST produce high risk scores

How it works:
    1. Load known attack prompts from injection_dataset_v1.csv (label=1)
    2. Pre-compute embeddings for all known attacks (attack signature bank)
    3. For each incoming prompt, compute cosine similarity to all signatures
    4. Return max similarity as the semantic risk score

Dependencies:
    pip install sentence-transformers numpy pandas

Usage:
    python prompt_guard/semantic_evaluator_v1.py                    # Full eval
    python prompt_guard/semantic_evaluator_v1.py --prompt "test"    # Single prompt
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    import sys
    sys.exit("sentence-transformers not installed. Run:  pip install sentence-transformers")

import torch

def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FILE_DIR.parent if _FILE_DIR.name == "prompt_guard" else _FILE_DIR
_DATASET_PATH = _PROJECT_ROOT / "datasets" / "injection_prompts" / "injection_dataset_v1.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
FALLBACK_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.65


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SemanticScore:
    """Evaluation result for a single prompt."""
    prompt: str
    semantic_score: float           # max cosine sim to known attacks (0-1)
    is_suspicious: bool
    matched_category: Optional[str] = None
    matched_technique: Optional[str] = None
    matched_prompt: Optional[str] = None
    top_k_similarities: List[float] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class EvaluationResult:
    """Batch evaluation result."""
    total_prompts: int
    flagged_count: int
    scores: List[SemanticScore] = field(default_factory=list)
    latency_ms: int = 0
    threshold: float = DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# Semantic Evaluator
# ---------------------------------------------------------------------------
class SemanticEvaluator:
    """
    Prompt injection detector using semantic similarity.

    Builds an "attack signature bank" from known injection prompts,
    then scores new prompts based on their cosine similarity to the
    nearest known attack.

    This goes beyond rule-based filters by catching:
    - Paraphrased attacks
    - Novel attack variations
    - Obfuscated injection attempts
    """

    def __init__(
        self,
        dataset_path: str | Path = _DATASET_PATH,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        top_k: int = 3,
    ):
        """
        Args:
            dataset_path:    Path to injection_dataset_v1.csv
            embedding_model: Sentence-transformer model name
            threshold:       Similarity score above which a prompt is flagged
            top_k:           Number of top similar attacks to track
        """
        self.dataset_path = Path(dataset_path)
        self.threshold = threshold
        self.top_k = top_k

        # Load embedding model
        device = _get_device()
        print(f"[SemanticEvaluator] Loading model: {embedding_model}  (device={device})")
        try:
            self.model = SentenceTransformer(embedding_model, device=device)
        except Exception:
            print(f"[SemanticEvaluator] Fallback to {FALLBACK_EMBEDDING_MODEL}")
            self.model = SentenceTransformer(FALLBACK_EMBEDDING_MODEL, device=device)

        dim = self.model.get_sentence_embedding_dimension()
        print(f"[SemanticEvaluator] Model ready. Dimension: {dim}, Device: {device}")

        # Load and encode attack signatures
        self._attack_prompts: List[Dict] = []
        self._attack_embeddings: Optional[np.ndarray] = None
        self._load_attack_signatures()

    # ------------------------------------------------------------------
    # Load known attack patterns
    # ------------------------------------------------------------------
    def _load_attack_signatures(self):
        """Load attack prompts from dataset and pre-compute embeddings."""
        if not self.dataset_path.exists():
            print(f"[SemanticEvaluator] WARNING: Dataset not found at {self.dataset_path}")
            print("[SemanticEvaluator] Running without attack signatures (all scores will be 0)")
            return

        # Read CSV
        attacks = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Strip whitespace/quotes from field names
            if reader.fieldnames:
                reader.fieldnames = [fn.strip().strip('"') for fn in reader.fieldnames]
            for row in reader:
                cleaned = {k.strip().strip('"'): v.strip().strip('"') for k, v in row.items()}
                if cleaned.get("label", "0") == "1":
                    attacks.append({
                        "prompt": cleaned.get("prompt", ""),
                        "category": cleaned.get("category", "unknown"),
                        "technique": cleaned.get("technique", "unknown"),
                    })

        if not attacks:
            print("[SemanticEvaluator] WARNING: No attack prompts found in dataset")
            return

        self._attack_prompts = attacks
        print(f"[SemanticEvaluator] Loaded {len(attacks)} attack signatures")

        # Pre-compute embeddings
        print(f"[SemanticEvaluator] Encoding attack signatures...")
        t0 = time.time()
        attack_texts = [a["prompt"] for a in attacks]
        self._attack_embeddings = self.model.encode(
            attack_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        elapsed = time.time() - t0
        print(f"[SemanticEvaluator] Signatures encoded in {elapsed:.2f}s")

    # ------------------------------------------------------------------
    # Score a single prompt
    # ------------------------------------------------------------------
    def evaluate(self, prompt: str) -> SemanticScore:
        """
        Compute semantic similarity of a prompt to known attack patterns.

        Returns:
            SemanticScore with similarity score and match info.
        """
        if self._attack_embeddings is None or len(self._attack_prompts) == 0:
            return SemanticScore(
                prompt=prompt,
                semantic_score=0.0,
                is_suspicious=False,
                confidence=0.0,
            )

        # Encode incoming prompt
        prompt_embedding = self.model.encode(
            [prompt], normalize_embeddings=True, show_progress_bar=False
        )

        # Cosine similarity (normalized embeddings -> dot product)
        similarities = np.dot(self._attack_embeddings, prompt_embedding[0])

        # Top-k matches
        top_k_indices = np.argsort(similarities)[-self.top_k:][::-1]
        top_k_sims = [float(similarities[idx]) for idx in top_k_indices]

        # Best match
        best_idx = top_k_indices[0]
        best_sim = top_k_sims[0]
        best_match = self._attack_prompts[best_idx]

        # Is suspicious?
        is_suspicious = best_sim >= self.threshold

        # Confidence based on score distribution
        if best_sim >= 0.85:
            confidence = 0.95
        elif best_sim >= 0.70:
            confidence = 0.85
        elif best_sim >= self.threshold:
            confidence = 0.70
        elif best_sim >= self.threshold - 0.10:
            confidence = 0.55   # borderline
        else:
            confidence = 0.90   # confident it's benign

        return SemanticScore(
            prompt=prompt,
            semantic_score=round(float(best_sim), 4),
            is_suspicious=is_suspicious,
            matched_category=best_match["category"] if is_suspicious else None,
            matched_technique=best_match["technique"] if is_suspicious else None,
            matched_prompt=best_match["prompt"] if is_suspicious else None,
            top_k_similarities=[round(s, 4) for s in top_k_sims],
            confidence=round(confidence, 2),
        )

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------
    def evaluate_batch(self, prompts: List[str]) -> EvaluationResult:
        """Evaluate a batch of prompts."""
        t0 = time.time()

        scores = []
        flagged = 0
        for prompt in prompts:
            score = self.evaluate(prompt)
            scores.append(score)
            if score.is_suspicious:
                flagged += 1

        latency_ms = int((time.time() - t0) * 1000)

        return EvaluationResult(
            total_prompts=len(prompts),
            flagged_count=flagged,
            scores=scores,
            latency_ms=latency_ms,
            threshold=self.threshold,
        )

    # ------------------------------------------------------------------
    # Full dataset evaluation (for metrics)
    # ------------------------------------------------------------------
    def evaluate_dataset(self, threshold: Optional[float] = None) -> Dict:
        """
        Evaluate the ENTIRE dataset (both benign and attack) and compute metrics.

        Uses leave-one-out for attack prompts: when evaluating an attack prompt,
        its own embedding is excluded from the signature bank to avoid trivial
        self-matching.

        Returns dict with TP, FP, TN, FN, precision, recall, F1, FPR.
        """
        if not self.dataset_path.exists():
            return {"error": "Dataset not found"}

        thresh = threshold or self.threshold

        # Load full dataset
        all_prompts = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Strip whitespace/quotes from field names
            if reader.fieldnames:
                reader.fieldnames = [fn.strip().strip('"') for fn in reader.fieldnames]
            for row in reader:
                # Strip keys and values for robustness
                cleaned = {k.strip().strip('"'): v.strip().strip('"') for k, v in row.items()}
                all_prompts.append({
                    "prompt": cleaned.get("prompt", ""),
                    "label": int(cleaned.get("label", "0")),
                    "category": cleaned.get("category", "unknown"),
                    "technique": cleaned.get("technique", "unknown"),
                })

        # Encode ALL prompts at once for efficiency
        all_texts = [p["prompt"] for p in all_prompts]
        print(f"[Eval] Encoding {len(all_texts)} prompts...")
        all_embeddings = self.model.encode(
            all_texts, normalize_embeddings=True,
            show_progress_bar=True, batch_size=32,
        )

        # Separate attack embeddings/indices
        attack_indices = [i for i, p in enumerate(all_prompts) if p["label"] == 1]
        attack_embeddings = all_embeddings[attack_indices]

        tp = fp = tn = fn = 0
        results = []

        # Build a quick lookup: attack index in attack_indices -> original prompt data
        attack_prompt_data = [all_prompts[idx] for idx in attack_indices]

        for i, prompt_data in enumerate(all_prompts):
            prompt_emb = all_embeddings[i]
            actual_label = prompt_data["label"]
            best_match_idx_in_attacks = 0

            if actual_label == 1:
                # Leave-one-out: exclude self from attack bank
                mask = np.array(attack_indices) != i
                if mask.sum() == 0:
                    max_sim = 0.0
                else:
                    filtered_attacks = attack_embeddings[mask]
                    sims = np.dot(filtered_attacks, prompt_emb)
                    max_sim = float(np.max(sims))
                    best_match_idx_in_attacks = int(np.where(mask)[0][np.argmax(sims)])
            else:
                # Benign: compare against all attacks
                sims = np.dot(attack_embeddings, prompt_emb)
                if len(sims) > 0:
                    max_sim = float(np.max(sims))
                    best_match_idx_in_attacks = int(np.argmax(sims))
                else:
                    max_sim = 0.0

            predicted = 1 if max_sim >= thresh else 0

            if actual_label == 1 and predicted == 1:
                tp += 1
            elif actual_label == 0 and predicted == 1:
                fp += 1
            elif actual_label == 0 and predicted == 0:
                tn += 1
            else:
                fn += 1

            # Get the best matching attack's info
            if len(attack_prompt_data) > best_match_idx_in_attacks:
                best_attack = attack_prompt_data[best_match_idx_in_attacks]
                matched_category = best_attack.get("category", "attack")
                matched_technique = best_attack.get("technique", "-")
                matched_prompt = best_attack.get("prompt", "")
            else:
                matched_category = "unknown"
                matched_technique = "unknown"
                matched_prompt = ""

            results.append({
                "prompt": prompt_data["prompt"][:80],
                "actual": actual_label,
                "predicted": predicted,
                "similarity": round(max_sim, 4),
                "category": prompt_data.get("category", "attack" if actual_label == 1 else "benign"),
                "technique": prompt_data.get("technique", "-"),
                "matched_category": matched_category,
                "matched_technique": matched_technique,
                "matched_prompt": matched_prompt[:80],
            })

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        accuracy = (tp + tn) / len(all_prompts) if len(all_prompts) > 0 else 0.0

        return {
            "threshold": thresh,
            "total": len(all_prompts),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "fpr": round(fpr, 4),
            "accuracy": round(accuracy, 4),
            "details": results,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Semantic Prompt Injection Evaluator")
    parser.add_argument("--dataset", type=str, default=str(_DATASET_PATH))
    parser.add_argument("--model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--prompt", type=str, default=None, help="Evaluate a single prompt")
    args = parser.parse_args()

    evaluator = SemanticEvaluator(
        dataset_path=args.dataset,
        embedding_model=args.model,
        threshold=args.threshold,
    )

    if args.prompt:
        # Single prompt mode
        score = evaluator.evaluate(args.prompt)
        status = "SUSPICIOUS" if score.is_suspicious else "CLEAN"
        print(f"\nPrompt:     \"{score.prompt}\"")
        print(f"Score:      {score.semantic_score:.4f}")
        print(f"Status:     {status}")
        print(f"Confidence: {score.confidence}")
        if score.matched_category:
            print(f"Matched:    {score.matched_category} / {score.matched_technique}")
            print(f"Similar to: \"{score.matched_prompt[:80]}...\"")
        print(f"Top-{evaluator.top_k} sims: {score.top_k_similarities}")
    else:
        # Full dataset evaluation
        print(f"\n[Eval] Running full dataset evaluation (threshold={args.threshold})...")
        metrics = evaluator.evaluate_dataset(threshold=args.threshold)

        print(f"\n{'='*60}")
        print(f"  SEMANTIC EVALUATOR METRICS (threshold={metrics['threshold']})")
        print(f"{'='*60}")
        print(f"  Total: {metrics['total']}")
        print(f"  TP: {metrics['tp']}  FP: {metrics['fp']}  TN: {metrics['tn']}  FN: {metrics['fn']}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  FPR:       {metrics['fpr']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"{'='*60}")

        # Show false negatives (missed attacks)
        fn_list = [r for r in metrics["details"] if r["actual"] == 1 and r["predicted"] == 0]
        if fn_list:
            print(f"\n  FALSE NEGATIVES ({len(fn_list)} missed attacks):")
            for r in fn_list[:10]:
                print(f"    sim={r['similarity']:.4f} | {r['prompt']}")
                if r.get("matched_prompt"):
                    print(f"      nearest: \"{r['matched_prompt'][:70]}...\"")

        # Show false positives (wrongly flagged benign)
        fp_list = [r for r in metrics["details"] if r["actual"] == 0 and r["predicted"] == 1]
        if fp_list:
            print(f"\n  FALSE POSITIVES ({len(fp_list)} wrongly flagged):")
            for r in fp_list[:10]:
                print(f"    sim={r['similarity']:.4f} | {r['prompt']}")
                if r.get("matched_prompt"):
                    print(f"      matched to: \"{r['matched_prompt'][:70]}...\"")


if __name__ == "__main__":
    main()
