# RAG Pipeline v7 — Model Upgrade Summary
**Date:** June 11, 2026  
**Context:** Replacing Titan Embeddings v1 + Nova Micro with better quality/cost models

---

## Decision: New Model Stack

| Component | Old | New | Reason |
|---|---|---|---|
| **Embeddings** | Titan Embeddings v1 (1536-dim) | **Cohere Embed v4** (`cohere.embed-v4:0`) | 128K context, better benchmarks, unified Cohere stack with existing reranker |
| **LLM (HyDE/HyQ + Final Answer)** | Nova Micro | **Claude Haiku 4.5** (`us.anthropic.claude-haiku-4-5-20251001-v1:0`) | Better instruction following, precise legal language, fixes answer_correctness (0.50) and answer_relevance (0.62) |
| **Reranker** | Cohere Rerank 3.5 | **Cohere Rerank 3.5** (unchanged) | Already best-in-class |

---

## Verified Model IDs (tested, working)

```python
EMBED_MODEL_ID   = "cohere.embed-v4:0"
LLM_MODEL_ID     = "us.anthropic.claude-haiku-4-5-20251001-v1:0"   # needs us. prefix — inference profile required
EMBED_DIMENSIONS = 1536    # Cohere v4 defaults to 1536, matches existing OpenSearch index
```

**Important:** Claude Haiku 4.5 requires the `us.` cross-region inference profile prefix. The bare model ID `anthropic.claude-haiku-4-5-20251001-v1:0` throws `ValidationException`.

Both models verified via `verify_bedrock_models.py` against account `<ACCOUNT_ID>`, region `us-east-1`, profile `rag-dev`.

---

## Key Technical Notes

**Cohere Embed v4 request format** is different from Titan — requires `input_type` field:
```python
# Ingestion
{"texts": [...], "input_type": "search_document", "embedding_types": ["float"]}

# Query time
{"texts": [...], "input_type": "search_query", "embedding_types": ["float"]}
```

**Dimensions:** Cohere v4 returned 1536 by default (same as Titan v1). `create_index.py` dimension value **does not need to change**. This eliminates the need to change the OpenSearch index mapping.

**Bedrock Model Access page has been retired** — models are now auto-enabled on first invocation. No manual access request needed.

**Why Bedrock over direct API:** IAM-based auth (no API keys in Lambda), single AWS bill, stays within AWS network, X-Ray + CloudWatch integration works automatically.

---

## Files That Need Code Changes

| File | Change |
|---|---|
| `llm_clients.py` | New model IDs; custom Cohere embed functions (`embed_documents`, `embed_query`) replacing LangChain `BedrockEmbeddings` wrapper |
| `chunker.py` | Update import — call `embed_query()` directly instead of via LangChain wrapper |
| `query_graph.py` | Update HyDE embed call to use new `embed_query()` |
| `ingestion_graph.py` | LLM calls auto-inherit Haiku 4.5 via `get_llm()` — no change needed if already using `get_llm()` |
| `create_index.py` | **No change needed** — dimension stays 1536 |

---

## Implementation Checklist (resume here)

- [ ] Turn on OpenSearch (15 min to start)
- [ ] Update `llm_clients.py` — new model IDs + Cohere embed functions
- [ ] Update `chunker.py` — embed call pattern
- [ ] Update `query_graph.py` — HyDE embed call
- [ ] Run `create_index.py` (dimension unchanged)
- [ ] Run `setup_hybrid_search.py`
- [ ] Map IAM backend role in OpenSearch Dashboards (`arn:aws:iam::<ACCOUNT_ID>:role/service-role/rag-document-processor-role-ucg8h778`)
- [ ] Deploy both Lambdas (ingestion + query)
- [ ] Re-ingest Johnson + Raceway Ford PDFs (137 chunks total)
- [ ] Run test query — verify LangSmith traces show Cohere + Haiku
- [ ] Compare LangSmith scores vs baseline (context_precision 0.45, answer_correctness 0.50, answer_relevance 0.62)

---

## Current Infrastructure State

- OpenSearch: **deactivated** (no re-ingestion done, safe to leave off)
- Existing index has Titan v1 embeddings — must re-ingest after model swap
- All other infrastructure unchanged from v7
