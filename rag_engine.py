import time
import math
import tiktoken
import logging
import sys
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
import functools

# =============================================================================
# Terminal Logger — coloured, structured, human-readable
# =============================================================================

class _ColouredFormatter(logging.Formatter):
    """Adds ANSI colour codes keyed to log level for fast visual scanning."""

    COLOURS = {
        logging.DEBUG:    "\033[0;36m",   # Cyan
        logging.INFO:     "\033[0;32m",   # Green
        logging.WARNING:  "\033[0;33m",   # Yellow
        logging.ERROR:    "\033[0;31m",   # Red
        logging.CRITICAL: "\033[1;31m",   # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record):
        colour = self.COLOURS.get(record.levelno, self.RESET)
        record.levelname = f"{colour}{record.levelname:<8}{self.RESET}"
        return super().format(record)


def _build_logger(name: str) -> logging.Logger:
    """Build (or return existing) coloured stream logger that writes to stdout."""
    log = logging.getLogger(name)
    if not log.handlers:                    # guard against Streamlit rerun duplication
        log.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(_ColouredFormatter(
            fmt="%(asctime)s  %(levelname)s  %(message)s",
            datefmt="%H:%M:%S"
        ))
        log.addHandler(handler)
        log.propagate = False
    return log


logger = _build_logger("rag_pipeline")

# Visual separators
_MAJOR = "=" * 72
_MINOR = "-" * 60


def _banner(title: str) -> str:
    return f"\n{_MAJOR}\n  {title}\n{_MAJOR}"


def _section(title: str) -> str:
    return f"\n{_MINOR}\n  {title}\n{_MINOR}"


# =============================================================================
# Trace decorator
# =============================================================================

def smart_trace(span_type, name=None, include_io=False):
    """Declarative trace decorator that finds the tracer on the instance (self)."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, "tracer") and self.tracer:
                span_name = name or func.__name__
                return self.tracer.trace(
                    name=span_name,
                    span_type=span_type,
                    include_io=include_io
                )(func)(self, *args, **kwargs)
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# RAGEngine
# =============================================================================

class RAGEngine:

    def __init__(self, llm, vector_store, tracer=None, k=8, max_context_tokens=2000,
                 distance_threshold=0.50, routing_threshold=0.50):
        self.llm = llm
        self.vector_store = vector_store
        self.tracer = tracer
        self.k = k
        self.max_context_tokens = max_context_tokens
        # distance_threshold : minimum sigmoid score a doc must have to enter the
        #                       LLM context window (context quality gate).
        self.distance_threshold = distance_threshold
        # routing_threshold  : minimum best-doc score required to route to "rag"
        #                       instead of "out_of_scope".
        self.routing_threshold = routing_threshold

        self.enc = tiktoken.get_encoding("cl100k_base")

        # ── Startup log ───────────────────────────────────────────────────────
        logger.info(_banner("RAGEngine — INITIALISING"))
        logger.info(f"  k                  = {k}")
        logger.info(f"  max_context_tokens = {max_context_tokens}")
        logger.info(f"  distance_threshold = {distance_threshold}")
        logger.info(f"  routing_threshold  = {routing_threshold}")

        all_docs = []
        if hasattr(self.vector_store, "docstore"):
            all_docs = list(self.vector_store.docstore._dict.values())

        logger.info(f"  Vector store loaded  —  {len(all_docs)} documents in docstore")

        if all_docs:
            self.bm25 = BM25Retriever.from_documents(all_docs)
            self.bm25.k = 8
            logger.info("  BM25 index built     —  k=8")
        else:
            self.bm25 = None
            logger.warning("  BM25 index SKIPPED   —  docstore is empty")

        logger.info("  Loading CrossEncoder reranker (ms-marco-MiniLM-L-6-v2) …")
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu",
            max_length=512
        )
        logger.info("  CrossEncoder ready")
        logger.info(_MAJOR + "\n")

    # =========================================================================
    # STEP 1 — Intent Classification
    # =========================================================================

    @smart_trace(span_type="intent-classification", name="intent-classifier")
    def classify_intent(self, query):
        logger.info(_section("STEP 1  ·  Intent Classification"))
        logger.info(f"  Input query : \"{query}\"")

        prompt = f"""Classify the following user message into exactly one of these two categories:
- greeting
- search_query

Rules:
- "greeting" means a simple salutation: hi, hello, hey, good morning, etc.
- "search_query" means any question or request for information.
- Output ONLY the category label. No explanation, no punctuation, nothing else.

User message: "{query}"

Category:"""

        t0 = time.perf_counter()
        response = self.llm.invoke(prompt)
        elapsed = (time.perf_counter() - t0) * 1000

        metadata = getattr(response, "response_metadata", {})
        usage = (
            metadata.get("token_usage")
            or metadata.get("usage_metadata")
            or metadata
        )

        content = response.content.strip().lower()
        intent  = "greeting" if (content.startswith("greeting") or content == "greeting") \
                  else "search_query"

        logger.info(f"  LLM raw response : \"{response.content.strip()}\"")
        logger.info(f"  Classified intent: \033[1m{intent.upper()}\033[0m   ({elapsed:.1f} ms)\n")

        return intent, usage

    # =========================================================================
    # STEP 2 — Query Rewrite
    # =========================================================================

    @smart_trace(span_type="chain", name="query-rewrite")
    def rewrite_query(self, query):
        logger.info(_section("STEP 2  ·  Query Rewrite"))
        logger.info(f"  Original : \"{query}\"")

        prompt = f"""INSTRUCTION: Rewrite the user's query into a descriptive natural language search phrase to improve vector document retrieval.
RULES:
1. Do NOT generate SQL, code, or structured database queries.
2. Use synonyms and technical variations of the terms.
3. Return ONLY the rewritten natural language phrase.

User Query:
{query}

Rewritten Phrase:"""

        t0 = time.perf_counter()
        response = self.llm.invoke(prompt)
        elapsed  = (time.perf_counter() - t0) * 1000

        rewritten = response.content.strip()
        logger.info(f"  Rewritten: \"{rewritten}\"   ({elapsed:.1f} ms)\n")

        return rewritten

    # =========================================================================
    # Helpers — RRF & MMR
    # =========================================================================

    def reciprocal_rank_fusion(self, vector_docs, bm25_docs, k=60):
        scores = {}
        for rank, (doc, _) in enumerate(vector_docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1 / (k + rank)

        seen = {}
        for doc, _ in vector_docs:
            seen[doc.page_content] = doc
        for doc in bm25_docs:
            seen[doc.page_content] = doc

        merged = [
            seen[c]
            for c, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]

        logger.debug(f"    RRF merged {len(merged)} unique docs "
                     f"(vector={len(vector_docs)}, bm25={len(bm25_docs)})")
        return merged

    def mmr_filter(self, docs, query):
        selected, seen = [], set()
        for doc in docs:
            key = doc.page_content[:200]
            if key in seen:
                continue
            seen.add(key)
            selected.append(doc)
            if len(selected) >= self.k * 3:
                break
        logger.debug(f"    MMR diversity filter: {len(docs)} → {len(selected)} docs")
        return selected

    # =========================================================================
    # STEP 3 — Retrieval
    # =========================================================================

    @smart_trace(span_type="retrieval", name="vector-search")
    def retrieve_documents(self, query):
        logger.info(_section("STEP 3  ·  Document Retrieval"))
        logger.info(f"  Search query : \"{query}\"")

        # ── Vector similarity search ──────────────────────────────────────────
        t0 = time.perf_counter()
        vector_docs = self.vector_store.similarity_search_with_score(
            query, k=self.k * 2
        )
        vec_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"  Vector search : {len(vector_docs)} hits  ({vec_ms:.1f} ms)")

        for i, (doc, raw_score) in enumerate(vector_docs[:5]):
            src     = doc.metadata.get("source", "unknown")
            snippet = doc.page_content[:70].replace("\n", " ")
            logger.debug(f"    [vec #{i+1:02d}]  raw_score={raw_score:.4f}  src={src}  \"{snippet}…\"")

        # ── BM25 keyword search ───────────────────────────────────────────────
        bm25_docs = []
        if self.bm25:
            t0 = time.perf_counter()
            bm25_docs = self.bm25.invoke(query)
            bm25_ms   = (time.perf_counter() - t0) * 1000
            logger.info(f"  BM25 search   : {len(bm25_docs)} hits  ({bm25_ms:.1f} ms)")
            for i, doc in enumerate(bm25_docs[:5]):
                src     = doc.metadata.get("source", "unknown")
                snippet = doc.page_content[:70].replace("\n", " ")
                logger.debug(f"    [bm25 #{i+1:02d}]  src={src}  \"{snippet}…\"")
        else:
            logger.warning("  BM25 search   : SKIPPED (index not available)")

        # ── Fusion + diversity ────────────────────────────────────────────────
        fused_docs      = self.reciprocal_rank_fusion(vector_docs, bm25_docs)
        diversified_docs = self.mmr_filter(fused_docs, query)
        logger.info(f"  After RRF+MMR : {len(diversified_docs)} candidate docs")

        # ── CrossEncoder reranking ────────────────────────────────────────────
        docs_with_scores = []

        if diversified_docs:
            logger.info(f"  Reranking {len(diversified_docs)} docs with CrossEncoder …")
            # Truncate each chunk to 400 chars to stay safely within CrossEncoder's 512-token limit
            pairs = [[query[:300], doc.page_content[:400]] for doc in diversified_docs]

            try:
                t0 = time.perf_counter()
                # Use convert_to_numpy=True (not convert_to_tensor) to avoid nan from tensor ops
                raw_scores = self.reranker.predict(
                    pairs, batch_size=32, convert_to_numpy=True, show_progress_bar=False
                )
                rerank_ms = (time.perf_counter() - t0) * 1000

                # Guard: use None sentinel for nan/inf so we can detect total failure
                cleaned = [
                    float(s) if (not math.isnan(float(s)) and not math.isinf(float(s))) else None
                    for s in raw_scores
                ]

                if all(v is None for v in cleaned):
                    # All scores bad — CrossEncoder model issue; fall back to RRF rank order
                    logger.warning("  CrossEncoder  : all scores are nan — model issue, "
                                   "falling back to RRF rank-based scores")
                    normalized_scores = [
                        max(0.9 - i * 0.05, 0.1) for i in range(len(diversified_docs))
                    ]
                else:
                    # Partial nan: replace bad individual scores with a neutral low value
                    normalized_scores = [
                        1 / (1 + math.exp(-(v if v is not None else -10.0)))
                        for v in cleaned
                    ]
                    logger.info(f"  CrossEncoder  : done  ({rerank_ms:.1f} ms)")

            except Exception as e:
                logger.error(f"  CrossEncoder FAILED: {e} — falling back to RRF rank order")
                # Fallback: assign descending scores based on RRF rank position
                normalized_scores = [
                    max(0.9 - i * 0.05, 0.1) for i in range(len(diversified_docs))
                ]

            docs_with_scores = sorted(
                zip(diversified_docs, normalized_scores),
                key=lambda x: x[1],
                reverse=True
            )

            # ── Per-doc score table ───────────────────────────────────────────
            GRN = "\033[0;32m"
            RED = "\033[0;31m"
            RST = "\033[0m"

            logger.info(f"\n"
                        f"  {'#':<4}  {'Score':>7}  {'Gate':^8}  "
                        f"{'Source':<28}  Snippet (first 60 chars)\n"
                        f"  {'─'*4}  {'─'*7}  {'─'*8}  {'─'*28}  {'─'*60}")

            for i, (doc, score) in enumerate(docs_with_scores):
                passes  = score >= self.distance_threshold
                gate    = f"{GRN}✓ PASS{RST}" if passes else f"{RED}✗ DROP{RST}"
                sc_str  = f"{GRN}{score:.4f}{RST}" if passes else f"{RED}{score:.4f}{RST}"
                src     = doc.metadata.get("source", "unknown")[:28]
                snippet = doc.page_content[:60].replace("\n", " ")
                logger.info(f"  {i+1:<4}  {sc_str:>7}  {gate:^8}  {src:<28}  \"{snippet}…\"")

            logger.info("")   # blank line after table

        # ── Context quality gate ──────────────────────────────────────────────
        logger.info(f"  Context gate   : distance_threshold={self.distance_threshold}")

        truncated_docs  = []
        current_tokens  = 0
        seen_parents    = set()

        for doc, score in docs_with_scores:
            if score < self.distance_threshold:
                continue

            content = doc.metadata.get("parent_context", doc.page_content)
            if content in seen_parents:
                continue

            tokens = len(self.enc.encode(content))
            if tokens > self.max_context_tokens:
                content = doc.page_content
                tokens  = len(self.enc.encode(content))

            if current_tokens + tokens <= self.max_context_tokens:
                doc.page_content = content
                truncated_docs.append((doc, score))
                current_tokens += tokens
                seen_parents.add(content)

        logger.info(f"  Context docs   : {len(truncated_docs)} selected  "
                    f"(~{current_tokens} / {self.max_context_tokens} tokens used)\n")

        return truncated_docs, docs_with_scores, vector_docs

    # =========================================================================
    # STEP 5 — Generation
    # =========================================================================

    @smart_trace(span_type="llm", name="{provider}_chat_completion")
    def generate_response(self, query, context, intent, routing_decision):
        logger.info(_section("STEP 5  ·  Response Generation"))
        logger.info(f"  Intent           : {intent}")
        logger.info(f"  Routing decision : {routing_decision}")
        logger.info(f"  Context length   : {len(context)} chars  ({len(context.split()) if context else 0} words)")

        if intent == "greeting":
            prompt   = f"User: {query}\n\nRespond with a short friendly greeting.\n\nResponse:"
            t0       = time.perf_counter()
            response = self.llm.invoke(prompt)
            elapsed  = (time.perf_counter() - t0) * 1000
            logger.info(f"  Path: GREETING   LLM latency={elapsed:.1f} ms")
            logger.info(f"  Answer: \"{response.content.strip()[:120]}\"\n")
            return response.content, prompt, getattr(response, "response_metadata", {})

        if routing_decision == "out_of_scope":
            prompt = f"""SYSTEM INSTRUCTION: The query is OUT OF SCOPE.

User Query: {query}

Respond with a short refusal (max 1 sentence).
"""
            t0       = time.perf_counter()
            response = self.llm.invoke(prompt)
            elapsed  = (time.perf_counter() - t0) * 1000
            logger.warning(f"  Path: OUT_OF_SCOPE   LLM latency={elapsed:.1f} ms")
            logger.warning(f"  Refusal: \"{response.content.strip()[:120]}\"\n")
            return response.content, prompt, getattr(response, "response_metadata", {})

        prompt = f"""SYSTEM RULES:
1. You are a helpful assistant answering questions based ONLY on the provided document excerpts below.
2. The documents are official company policy documents. You ARE authorized to share this information.
3. Answer directly and factually using the context. Do NOT refuse or say it is out of scope.
4. If the exact answer is not found in the context, say "The provided documents don't contain a specific answer to this question."
5. Keep the answer under 3 sentences.

Context (from company documents):
{context}

Question:
{query}

Answer:
"""
        t0       = time.perf_counter()
        response = self.llm.invoke(prompt)
        elapsed  = (time.perf_counter() - t0) * 1000
        logger.info(f"  Path: RAG   LLM latency={elapsed:.1f} ms")
        logger.info(f"  Answer: \"{response.content.strip()[:120]}…\"\n")

        return response.content, prompt, getattr(response, "response_metadata", {})

    # =========================================================================
    # Main Pipeline
    # =========================================================================

    @smart_trace(span_type="generic", name="rag-pipeline", include_io=True)
    def run(self, query):
        t_start = time.perf_counter()
        logger.info(_banner(f"RAG PIPELINE START  ▶  \"{query}\""))

        # STEP 1 — Intent
        intent, _ = self.classify_intent(query)

        routing_decision  = None
        trace_name        = "simple-qa"
        safe_docs         = []
        all_reranked_docs = []      # always initialised — avoids UnboundLocalError
        vector_docs       = []      # raw vector results needed for routing override

        if intent == "search_query":
            # STEP 2 — Query rewrite
            rewritten    = self.rewrite_query(query)
            search_query = rewritten if rewritten else query

            # STEP 3 — Retrieval
            safe_docs, all_reranked_docs, vector_docs = self.retrieve_documents(search_query)

            # STEP 4 — Routing
            best_score = max(
                (float(s) for _, s in all_reranked_docs if not math.isnan(float(s))),
                default=None
            )

            logger.info(_section("STEP 4  ·  Routing Decision"))
            if best_score is not None:
                logger.info(f"  Best reranker score : {best_score:.4f}")
            else:
                logger.info("  Best reranker score : None  (no documents retrieved)")
            logger.info(f"  Routing threshold   : {self.routing_threshold}")

            if best_score is not None and best_score >= self.routing_threshold:
                routing_decision = "rag"
                trace_name       = "rag-qa"
                logger.info(f"  \033[0;32mROUTE → RAG  ✓  "
                            f"({best_score:.4f} ≥ {self.routing_threshold})\033[0m\n")
            else:
                routing_decision = "out_of_scope"
                trace_name       = "out-of-scope-qa"
                logger.warning(f"  ROUTE → OUT_OF_SCOPE  ✗  "
                               f"(best={(best_score if best_score is not None else 0.0):.4f}, "
                               f"threshold={self.routing_threshold})\n")

            # ── Secondary override: catch CrossEncoder fallback routing to RAG ──
            # If reranker score looks like an RRF fallback (≥0.45, e.g. sigmoid(0)=0.5
            # or rank-based 0.9/0.85/...), cross-check raw vector confidence.
            # If the vector store itself isn't confident, override to out_of_scope.
            if routing_decision == "rag" and best_score is not None and best_score > 0.45:
                top_vector_score = float(vector_docs[0][1]) if vector_docs else 0.0
                if top_vector_score < 0.45:
                    routing_decision = "out_of_scope"
                    trace_name       = "out-of-scope-qa"
                    logger.warning(f"  ROUTE OVERRIDE → OUT_OF_SCOPE  "
                                   f"(reranker unreliable, vector confidence too low: "
                                   f"{top_vector_score:.4f})\n")

        # Build context string
        context = ""
        if routing_decision == "rag":
            docs_for_context = safe_docs if safe_docs else all_reranked_docs[:3]
            context = "\n\n".join(
                f"SOURCE {i+1}:\n{doc.page_content}"
                for i, (doc, _) in enumerate(docs_for_context)
            )
            logger.info(f"  Context assembled from {len(docs_for_context)} source(s)  "
                        f"—  {len(context)} chars")

        # STEP 5 — Generate
        output, _, _ = self.generate_response(query, context, intent, routing_decision)

        total_ms = (time.perf_counter() - t_start) * 1000
        logger.info(_banner(
            f"RAG PIPELINE COMPLETE  ◀  {total_ms:.0f} ms  "
            f"|  route={trace_name}  |  intent={intent}"
        ))

        return {
            "trace_name"      : trace_name,
            "output"          : output,
            "safe_docs"       : safe_docs,
            "intent"          : intent,
            "routing_decision": routing_decision,
        }