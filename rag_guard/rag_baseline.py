"""
rag_guard/rag_baseline.py
==========================
Baseline (VULNERABLE) RAG retrieval pipeline.

Purpose:
    - Index documents (clean + poisoned) into ChromaDB
    - Perform embedding-based retrieval WITHOUT any defense
    - Demonstrate that poisoned documents are retrieved for target queries
    - Serve as the "before defense" baseline for evaluation

Dependencies:
    pip install chromadb sentence-transformers

Usage:
    python rag_guard/rag_baseline.py                       # Full demo
    python rag_guard/rag_baseline.py --query "password policy"  # Single query

Architecture:
    poison_samples.json ──► ChromaDB (persistent) ──► top-k retrieval
                                                          │
                                                     No filtering
                                                     No poison check
                                                          │
                                                     Raw results returned
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

# ---------------------------------------------------------------------------
# Lazy imports – give clear error messages if deps are missing
# ---------------------------------------------------------------------------
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    sys.exit("chromadb not installed. Run:  pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("sentence-transformers not installed. Run:  pip install sentence-transformers")

import os
import torch

def _get_device() -> str:
    if not os.environ.get("EMBEDDING_DEVICE"):
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        except ImportError:
            pass
    forced = os.environ.get("EMBEDDING_DEVICE", "").lower()
    if forced in ("cpu", "cuda"):
        return forced
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# ---------------------------------------------------------------------------
# Project paths (works from repo root or from rag_guard/)
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FILE_DIR.parent if _FILE_DIR.name == "rag_guard" else _FILE_DIR
_DATASET_PATH = _PROJECT_ROOT / "datasets" / "poisoned_corpus" / "poison_samples.json"
_CHROMA_DIR = _PROJECT_ROOT / "data" / "chroma_baseline"


# ---------------------------------------------------------------------------
# Config defaults (mirrors configs/secure_balanced.yaml)
# ---------------------------------------------------------------------------
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"          # multilingual, dense+sparse
FALLBACK_EMBEDDING_MODEL = "all-MiniLM-L6-v2"    # lightweight fallback
COLLECTION_NAME = "rag_baseline_corpus"
TOP_K = 5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RetrievedDoc:
    """A single retrieved document with metadata."""
    doc_id: str
    title: str
    content: str
    distance: float
    is_poisoned: bool
    poison_type: Optional[str] = None
    source: Optional[str] = None


@dataclass
class RetrievalResult:
    """Full retrieval result for a single query."""
    query: str
    top_k: int
    results: List[RetrievedDoc] = field(default_factory=list)
    latency_ms: int = 0
    poisoned_count: int = 0
    total_retrieved: int = 0

    @property
    def poison_ratio(self) -> float:
        if self.total_retrieved == 0:
            return 0.0
        return self.poisoned_count / self.total_retrieved

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["poison_ratio"] = self.poison_ratio
        return d


# ---------------------------------------------------------------------------
# Embedding wrapper
# ---------------------------------------------------------------------------
class EmbeddingEngine:
    """Wraps SentenceTransformer for document/query embedding."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        device = _get_device()
        print(f"[EmbeddingEngine] Loading model: {model_name}  (device={device})")
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception:
            print(f"[EmbeddingEngine] Failed to load {model_name}, falling back to {FALLBACK_EMBEDDING_MODEL}")
            self.model = SentenceTransformer(FALLBACK_EMBEDDING_MODEL, device=device)
        self.model_name = self.model.get_sentence_embedding_dimension()
        print(f"[EmbeddingEngine] Ready. Dimension: {self.model.get_sentence_embedding_dimension()}, Device: {device}")

    def encode(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def encode_single(self, text: str) -> List[float]:
        return self.encode([text])[0]


# ---------------------------------------------------------------------------
# Baseline RAG Pipeline (VULNERABLE - no defenses)
# ---------------------------------------------------------------------------
class RAGBaselinePipeline:
    """
    Vulnerable RAG pipeline for baseline evaluation.

    This pipeline intentionally has NO security measures:
    - No poison detection
    - No content filtering
    - No risk scoring
    - No input validation

    It serves as the "before" state to compare with defended pipelines.
    """

    def __init__(
        self,
        dataset_path: str | Path = _DATASET_PATH,
        chroma_dir: str | Path = _CHROMA_DIR,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        collection_name: str = COLLECTION_NAME,
        top_k: int = TOP_K,
    ):
        self.dataset_path = Path(dataset_path)
        self.chroma_dir = Path(chroma_dir)
        self.top_k = top_k
        self.collection_name = collection_name

        # Initialize embedding engine
        self.embedder = EmbeddingEngine(embedding_model)

        # Initialize ChromaDB (persistent)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._documents_meta: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def load_and_index(self, force_reindex: bool = False) -> int:
        """
        Load documents from poison_samples.json and index into ChromaDB.

        Returns:
            Number of documents indexed.
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        # Skip if already indexed (unless forced)
        existing_count = self.collection.count()
        if existing_count > 0 and not force_reindex:
            print(f"[Index] Collection already has {existing_count} documents. Use force_reindex=True to re-index.")
            self._load_metadata()
            return existing_count

        # Clear collection if re-indexing
        if force_reindex and existing_count > 0:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        # Load dataset
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        documents = dataset["documents"]
        print(f"[Index] Loading {len(documents)} documents from {self.dataset_path.name}")

        # Prepare for batch indexing
        ids = []
        texts = []
        metadatas = []

        for doc in documents:
            doc_id = doc["doc_id"]
            content = doc["content"]
            title = doc.get("title", "")

            # Combine title + content for richer embedding
            full_text = f"{title}. {content}" if title else content

            ids.append(doc_id)
            texts.append(full_text)
            metadatas.append({
                "doc_id": doc_id,
                "title": title,
                "source": doc.get("source", "unknown"),
                "is_poisoned": str(doc.get("is_poisoned", False)),
                "poison_type": doc.get("poison_type") or "none",
                "poison_technique": doc.get("poison_technique") or "none",
                "target_query": doc.get("target_query") or "none",
            })

            # Store full metadata for result enrichment
            self._documents_meta[doc_id] = doc

        # Create embeddings
        print(f"[Index] Generating embeddings for {len(texts)} documents...")
        t0 = time.time()
        embeddings = self.embedder.encode(texts)
        embed_time = time.time() - t0
        print(f"[Index] Embeddings generated in {embed_time:.2f}s")

        # Index into ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        print(f"[Index] Indexed {len(ids)} documents into ChromaDB collection '{self.collection_name}'")

        # Print summary
        poisoned = sum(1 for d in documents if d.get("is_poisoned"))
        clean = len(documents) - poisoned
        print(f"[Index] Clean: {clean} | Poisoned: {poisoned}")

        return len(ids)

    def _load_metadata(self):
        """Load metadata from dataset file for result enrichment."""
        if self._documents_meta:
            return
        if self.dataset_path.exists():
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            for doc in dataset["documents"]:
                self._documents_meta[doc["doc_id"]] = doc

    # ------------------------------------------------------------------
    # Retrieval (VULNERABLE - no filtering/defense)
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """
        Retrieve top-k documents for a query. NO SECURITY CHECKS.

        This is intentionally vulnerable:
        - No poison detection on retrieved documents
        - No content filtering
        - No risk scoring
        - Returns raw results from vector similarity search
        """
        k = top_k or self.top_k
        t0 = time.time()

        # Embed query
        query_embedding = self.embedder.encode_single(query)

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        latency_ms = int((time.time() - t0) * 1000)

        # Build result objects
        retrieved_docs = []
        poisoned_count = 0

        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0

                is_poisoned = meta.get("is_poisoned", "False") == "True"
                if is_poisoned:
                    poisoned_count += 1

                # Get full content from metadata store
                full_meta = self._documents_meta.get(doc_id, {})

                retrieved_docs.append(RetrievedDoc(
                    doc_id=doc_id,
                    title=meta.get("title", "Unknown"),
                    content=full_meta.get("content", results["documents"][0][i] if results["documents"] else ""),
                    distance=round(distance, 4),
                    is_poisoned=is_poisoned,
                    poison_type=meta.get("poison_type", "none") if is_poisoned else None,
                    source=meta.get("source", "unknown"),
                ))

        return RetrievalResult(
            query=query,
            top_k=k,
            results=retrieved_docs,
            latency_ms=latency_ms,
            poisoned_count=poisoned_count,
            total_retrieved=len(retrieved_docs),
        )

    # ------------------------------------------------------------------
    # Convenience: build context string (what an LLM would receive)
    # ------------------------------------------------------------------
    def build_context(self, query: str, top_k: Optional[int] = None) -> tuple[str, RetrievalResult]:
        """
        Retrieve docs and build a context string for LLM prompting.
        Returns (context_string, retrieval_result).
        """
        result = self.retrieve(query, top_k)

        context_parts = []
        for i, doc in enumerate(result.results, 1):
            context_parts.append(f"[Document {i}]: {doc.content}")

        context = "\n\n".join(context_parts)
        return context, result

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        """Return collection statistics."""
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "chroma_dir": str(self.chroma_dir),
            "embedding_model": self.embedder.model_name,
        }


# ---------------------------------------------------------------------------
# Vulnerability demonstration
# ---------------------------------------------------------------------------
def demonstrate_vulnerability(pipeline: RAGBaselinePipeline):
    """
    Run target queries from poisoned documents and show that
    poisoned content is retrieved WITHOUT any defense.
    """
    print("\n" + "=" * 70)
    print("  VULNERABILITY DEMONSTRATION")
    print("  Baseline RAG pipeline - NO defenses active")
    print("=" * 70)

    # Collect target queries from poisoned docs
    target_queries = []
    for doc_id, meta in pipeline._documents_meta.items():
        if meta.get("is_poisoned") and meta.get("target_query"):
            target_queries.append({
                "query": meta["target_query"],
                "expected_poison_type": meta["poison_type"],
                "expected_doc_id": doc_id,
            })

    if not target_queries:
        print("[!] No target queries found in poisoned documents.")
        return

    total_poisoned_retrieved = 0
    total_queries = len(target_queries)

    # Top-k poison ratio trackers
    topk_levels = [1, 3, 5]
    topk_poison_counts = {k: 0 for k in topk_levels}   # queries where at least 1 poisoned in top-k
    topk_ratio_sums = {k: 0.0 for k in topk_levels}    # sum of poison ratios for averaging

    for i, tq in enumerate(target_queries, 1):
        query = tq["query"]
        result = pipeline.retrieve(query)

        print(f"\n--- Query {i}/{total_queries}: \"{query}\" ---")
        print(f"    Expected attack: {tq['expected_poison_type']}")
        print(f"    Retrieved {result.total_retrieved} docs | "
              f"Poisoned: {result.poisoned_count} | "
              f"Poison ratio: {result.poison_ratio:.0%} | "
              f"Latency: {result.latency_ms}ms")

        for j, doc in enumerate(result.results, 1):
            status = "POISONED" if doc.is_poisoned else "CLEAN"
            print(f"    [{j}] {status} | dist={doc.distance:.4f} | {doc.doc_id}: {doc.title}")

        if result.poisoned_count > 0:
            total_poisoned_retrieved += 1

        # Compute top-k poison ratios
        for k in topk_levels:
            top_k_docs = result.results[:k]
            poisoned_in_k = sum(1 for d in top_k_docs if d.is_poisoned)
            if poisoned_in_k > 0:
                topk_poison_counts[k] += 1
            topk_ratio_sums[k] += poisoned_in_k / k if k <= len(result.results) else 0.0

    # Summary
    attack_success_rate = total_poisoned_retrieved / total_queries if total_queries > 0 else 0
    print("\n" + "=" * 70)
    print("  BASELINE VULNERABILITY SUMMARY")
    print("=" * 70)
    print(f"  Total target queries:         {total_queries}")
    print(f"  Queries with poisoned results: {total_poisoned_retrieved}")
    print(f"  Attack Success Rate (ASR):     {attack_success_rate:.1%}")

    # Top-k poison ratio table
    print(f"\n  Top-k Poison Ratio Analysis:")
    print(f"  {'k':>5} | {'ASR@k':>8} | {'Avg Poison Ratio':>18}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*18}")
    topk_results = {}
    for k in topk_levels:
        asr_at_k = topk_poison_counts[k] / total_queries if total_queries > 0 else 0
        avg_ratio = topk_ratio_sums[k] / total_queries if total_queries > 0 else 0
        print(f"  {k:>5} | {asr_at_k:>7.1%} | {avg_ratio:>17.3f}")
        topk_results[f"asr_top{k}"] = round(asr_at_k, 3)
        topk_results[f"avg_poison_ratio_top{k}"] = round(avg_ratio, 3)

    print(f"\n  VERDICT: {'VULNERABLE' if attack_success_rate > 0.5 else 'PARTIALLY VULNERABLE'}")
    print("=" * 70)

    return {
        "total_queries": total_queries,
        "poisoned_retrieved": total_poisoned_retrieved,
        "asr": round(attack_success_rate, 3),
        **topk_results,
    }


# ---------------------------------------------------------------------------
# General query examples (non-targeted, to check normal behavior)
# ---------------------------------------------------------------------------
def demo_general_queries(pipeline: RAGBaselinePipeline):
    """Run general queries to show normal retrieval behavior."""
    print("\n" + "-" * 70)
    print("  GENERAL QUERY EXAMPLES (non-targeted)")
    print("-" * 70)

    general_queries = [
        "How do I deploy a machine learning model?",
        "What is the best branching strategy for Git?",
        "Explain microservices architecture",
        "How does Kubernetes handle scaling?",
        "What monitoring tools should I use?",
    ]

    for query in general_queries:
        result = pipeline.retrieve(query, top_k=3)
        print(f"\nQ: \"{query}\"")
        for j, doc in enumerate(result.results, 1):
            status = "POISONED" if doc.is_poisoned else "CLEAN"
            print(f"  [{j}] {status} | dist={doc.distance:.4f} | {doc.title}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RAG Baseline Pipeline (Vulnerable)")
    parser.add_argument("--dataset", type=str, default=str(_DATASET_PATH),
                        help="Path to poison_samples.json")
    parser.add_argument("--chroma-dir", type=str, default=str(_CHROMA_DIR),
                        help="ChromaDB persistent storage directory")
    parser.add_argument("--model", type=str, default=DEFAULT_EMBEDDING_MODEL,
                        help=f"Embedding model (default: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"Number of documents to retrieve (default: {TOP_K})")
    parser.add_argument("--query", type=str, default=None,
                        help="Run a single query instead of full demo")
    parser.add_argument("--force-reindex", action="store_true",
                        help="Force re-indexing even if collection exists")
    args = parser.parse_args()

    # Initialize pipeline
    pipeline = RAGBaselinePipeline(
        dataset_path=args.dataset,
        chroma_dir=args.chroma_dir,
        embedding_model=args.model,
        top_k=args.top_k,
    )

    # Index documents
    count = pipeline.load_and_index(force_reindex=args.force_reindex)
    print(f"\n[Ready] {count} documents in collection")

    if args.query:
        # Single query mode
        result = pipeline.retrieve(args.query)
        print(f"\nQuery: \"{args.query}\"")
        print(f"Retrieved: {result.total_retrieved} | Poisoned: {result.poisoned_count} | Latency: {result.latency_ms}ms")
        for j, doc in enumerate(result.results, 1):
            status = "POISONED" if doc.is_poisoned else "CLEAN"
            print(f"  [{j}] {status} | dist={doc.distance:.4f} | {doc.doc_id}: {doc.title}")
            print(f"       {doc.content[:120]}...")
    else:
        # Full demo
        vulnerability_stats = demonstrate_vulnerability(pipeline)
        demo_general_queries(pipeline)

        # Save results
        results_dir = _PROJECT_ROOT / "runs"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / "baseline_vulnerability_report.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(vulnerability_stats, f, indent=2)
        print(f"\n[Saved] Vulnerability report: {results_path}")


if __name__ == "__main__":
    main()
