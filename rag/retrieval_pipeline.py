from functools import lru_cache
from pathlib import Path

try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:  # pragma: no cover - exercised in bootstrap environments
    try:
        from langchain.retrievers import EnsembleRetriever
    except ImportError:
        EnsembleRetriever = None

try:
    from langchain_community.retrievers import BM25Retriever
except ImportError:  # pragma: no cover - exercised in bootstrap environments
    BM25Retriever = None

try:
    from langchain_community.vectorstores import FAISS
except ImportError:  # pragma: no cover - exercised in bootstrap environments
    FAISS = None

try:
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover - exercised in bootstrap environments
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        Chroma = None

from rag.context_packer import pack_retrieved_context, reorder_for_long_context
from rag.document_index import load_document_summary_index, select_document_ids
from rag.embeddings import EMBEDDING_MAP, embeddings_available
from rag.query_planner import build_query_plan
from rag.reranker import chunk_content, rerank_chunks

BASE_DIR = Path(__file__).resolve().parents[1]
CHROMA_DIR = BASE_DIR / "chroma"
FAISS_DIR = BASE_DIR / "faiss"
DOCUMENT_INDEX_FILE = FAISS_DIR / "document_index" / "document_summaries.json"

RETRIEVAL_CONFIG = {
    "rag_enrich_node": {
        "knowledge_db": {"mode": "hybrid", "k": 6},
        "reasoning_db": {"mode": "mmr", "k": 4, "fetch_k": 25, "lambda_mult": 0.65},
    },
    "argue_node": {
        "knowledge_db": {"mode": "hybrid", "k": 3},
        "reasoning_db": {"mode": "mmr", "k": 5, "fetch_k": 25, "lambda_mult": 0.65},
        "style_db": {"mode": "similarity_score_threshold", "k": 2, "score_threshold": 0.72},
    },
    "coach_node": {
        "knowledge_db": {"mode": "hybrid", "k": 2},
        "reasoning_db": {"mode": "mmr", "k": 3, "fetch_k": 25, "lambda_mult": 0.65},
        "style_db": {"mode": "similarity_score_threshold", "k": 5, "score_threshold": 0.72},
    },
    "english_coach_node": {
        "english_db": {"mode": "similarity", "k": 6},
    },
}


@lru_cache(maxsize=8)
def _load_vector_store(store_name: str):
    if store_name in BROKEN_STORES:
        return None

    embedding = EMBEDDING_MAP.get(store_name)
    if embedding is None:
        return None

    faiss_dir = FAISS_DIR / store_name
    if FAISS is not None and faiss_dir.exists() and any(faiss_dir.iterdir()):
        try:
            return FAISS.load_local(
                str(faiss_dir),
                embedding,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            pass

    if Chroma is not None:
        chroma_dir = CHROMA_DIR / store_name
        if chroma_dir.exists():
            return Chroma(
                persist_directory=str(chroma_dir),
                embedding_function=embedding,
            )

    return None


@lru_cache(maxsize=1)
def _load_document_index() -> dict[str, list[dict]]:
    return load_document_summary_index(DOCUMENT_INDEX_FILE)


STORE_SECTION_LABELS = {
    "knowledge_db": "DOMAIN KNOWLEDGE",
    "reasoning_db": "DEBATE THEORY AND REASONING",
    "style_db": "YOUR DEBATE STYLE",
    "english_db": "ENGLISH VOCABULARY (Word Power Made Easy)",
}

BROKEN_STORES: set[str] = set()


def _vector_store_corpus(vector_store) -> tuple[list[str], list[dict]]:
    """Pull (texts, metadatas) from a FAISS or Chroma vector store for BM25."""
    # FAISS — read from docstore
    docstore = getattr(vector_store, "docstore", None)
    if docstore is not None and hasattr(docstore, "_dict"):
        texts, metadatas = [], []
        for document in docstore._dict.values():
            texts.append(document.page_content)
            metadatas.append(getattr(document, "metadata", {}) or {})
        return texts, metadatas

    # Chroma — get() returns {documents, metadatas}
    if hasattr(vector_store, "get"):
        try:
            data = vector_store.get()
            return list(data.get("documents") or []), list(data.get("metadatas") or [])
        except Exception:
            return [], []

    return [], []


def build_hybrid_retriever(store_name: str, k: int = 6):
    if EnsembleRetriever is None or BM25Retriever is None:
        return None

    vector_store = _load_vector_store(store_name)
    if vector_store is None:
        return None

    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    texts, metadatas = _vector_store_corpus(vector_store)
    if not texts:
        # No corpus available — degrade gracefully to vector-only retrieval.
        return vector_retriever

    bm25_retriever = BM25Retriever.from_texts(texts, metadatas=metadatas or None)
    bm25_retriever.k = k
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],
    )


def _build_retriever(store_name: str, config: dict):
    if store_name in BROKEN_STORES or not embeddings_available(store_name):
        return None

    vector_store = _load_vector_store(store_name)
    if vector_store is None:
        return None

    mode = config["mode"]
    if mode == "hybrid":
        return build_hybrid_retriever(store_name, config["k"])

    search_kwargs = {key: value for key, value in config.items() if key != "mode"}
    return vector_store.as_retriever(search_type=mode, search_kwargs=search_kwargs)


def _oversampled_config(config: dict) -> dict:
    expanded = dict(config)
    base_k = int(config.get("k") or 4)
    expanded["k"] = max(base_k * 4, int(config.get("fetch_k") or 0), 12)
    if expanded.get("mode") == "mmr":
        expanded["fetch_k"] = max(int(config.get("fetch_k") or 0), expanded["k"])
    return expanded


def _filter_chunks_to_documents(chunks: list, candidate_document_ids: list[str]) -> list:
    if not candidate_document_ids:
        return list(chunks or [])

    allowed = set(candidate_document_ids)
    filtered = []
    for chunk in chunks or []:
        _, metadata = chunk_content(chunk)
        if str(metadata.get("document_id") or "") in allowed:
            filtered.append(chunk)
    return filtered


def retrieve_for_node(node_name: str, query: str, state: dict | None = None, plan: dict | None = None) -> dict:
    """Returns {store_name: [chunks, ...]} so callers can preserve lane labels.

    Plain list iteration still works because `format_retrieved_context` accepts
    either a dict-of-lanes or a flat list.
    """
    if plan is None and state is not None:
        try:
            plan = build_query_plan(node_name, state)
        except Exception as exc:
            print(f"[Retrieval] query plan failed for {node_name}: {exc}")
            plan = None

    node_config = RETRIEVAL_CONFIG.get(node_name, {})
    retrieved: dict[str, list] = {}

    for store_name, config in node_config.items():
        preferred_stores = (plan or {}).get("preferred_stores", []) or []
        if preferred_stores and store_name not in preferred_stores:
            continue

        retriever = _build_retriever(store_name, _oversampled_config(config))
        if retriever is None:
            continue

        try:
            store_query = ((plan or {}).get("store_queries", {}) or {}).get(store_name, query)
            candidate_document_ids = select_document_ids(
                _load_document_index(),
                store_name,
                store_query,
                plan=plan,
                limit=max(int(config.get("k") or 4), 4),
            )
            raw_chunks = retriever.invoke(store_query)
            filtered_chunks = _filter_chunks_to_documents(raw_chunks, candidate_document_ids)
            if filtered_chunks:
                raw_chunks = filtered_chunks
            limit = int(config.get("k") or len(raw_chunks) or 0)
            reranked = rerank_chunks(raw_chunks, store_query, plan, store_name, limit or len(raw_chunks))
            retrieved[store_name] = reorder_for_long_context(reranked, store_query, plan, store_name)
        except Exception as exc:
            print(f"[Retrieval] {store_name} failed for {node_name}: {exc}")
            BROKEN_STORES.add(store_name)
            retrieved[store_name] = []

    return retrieved


def build_retrieval_trace(chunks: dict[str, list], query_plan: dict | None = None) -> dict:
    trace: dict[str, list[dict]] = {}
    queries = (query_plan or {}).get("store_queries", {}) or {}

    for store_name, store_chunks in (chunks or {}).items():
        store_trace: list[dict] = []
        for chunk in store_chunks[:5]:
            content, metadata = chunk_content(chunk)
            store_trace.append(
                {
                    "source_ref": metadata.get("source_path")
                    or metadata.get("url")
                    or metadata.get("video_id")
                    or "unknown",
                    "document_id": metadata.get("document_id"),
                    "source_class": metadata.get("source_class"),
                    "topic_family": metadata.get("topic_family"),
                    "debate_utility": metadata.get("debate_utility"),
                    "query": queries.get(store_name, ""),
                    "preview": " ".join(content.split())[:180],
                }
            )
        trace[store_name] = store_trace
    return trace


def retrieve_bundle_for_node(node_name: str, query: str, state: dict | None = None, plan: dict | None = None) -> dict:
    if plan is None and state is not None:
        try:
            plan = build_query_plan(node_name, state)
        except Exception as exc:
            print(f"[Retrieval] query plan failed for {node_name}: {exc}")
            plan = None

    chunks = retrieve_for_node(node_name, query, state=state, plan=plan)
    trace = build_retrieval_trace(chunks, query_plan=plan)
    return {"chunks": chunks, "plan": plan, "trace": trace}


def _chunk_content(chunk) -> tuple[str, dict]:
    if hasattr(chunk, "page_content"):
        return chunk.page_content, getattr(chunk, "metadata", {}) or {}
    return chunk.get("page_content", ""), chunk.get("metadata", {}) or {}


def _format_lane(label: str, chunks: list) -> str:
    if not chunks:
        return ""
    lines = [f"{label}:"]
    for chunk in chunks:
        content, metadata = _chunk_content(chunk)
        source = (
            metadata.get("source_path")
            or metadata.get("url")
            or metadata.get("video_id")
            or "unknown"
        )
        lines.append(f"[source: {source}]\n{content}")
    return "\n---\n".join(lines)


def format_retrieved_context(chunks) -> str:
    """Format retrieved chunks for LLM prompts.

    Accepts the dict shape returned by `retrieve_for_node` (preferred — it
    keeps the per-lane DOMAIN KNOWLEDGE / DEBATE THEORY / YOUR DEBATE STYLE
    headers the prompts depend on) or a flat list (legacy callers).
    """
    if isinstance(chunks, dict):
        return pack_retrieved_context(chunks, labels=STORE_SECTION_LABELS)

    return _format_lane("RETRIEVED CONTEXT", list(chunks)).strip()
