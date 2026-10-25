# Architecture Decision Records — Financial RAG Pipeline
### (`rag-pipeline-financial`, supporting the Financial Research Agent POC)

**Project:** Financial Research Agent POC — cross-company financial question answering grounded in SEC 10-K filings (Apple, Microsoft, Amazon)
**Scope of this document:** the RAG (retrieval) layer only — ingestion, chunking, embedding, hybrid search, and query pipeline. The agent orchestration layer (`financial-research-agent`, LangGraph multi-node agent) consumes this pipeline as a tool and is out of scope here.
**AWS account:** <ACCOUNT_ID>, region `us-east-1`, profile `rag-dev`

---

## ADR-001: Fork the existing Legal RAG pipeline, with complete isolation

**Status:** Accepted

**Context:** A working Legal RAG pipeline already existed with a proven ingestion → embed → index → hybrid-search → answer architecture. Building the Financial pipeline from scratch would duplicate proven design work; extending the Legal pipeline directly would create shared-ownership risk between two demo projects with different stakeholders and different data.

**Decision:** Fork the Legal pipeline's codebase as the starting point for `rag-pipeline-financial`, but enforce complete infrastructure isolation from the Legal pipeline — separate S3 buckets, separate OpenSearch index, separate DynamoDB table, separate Lambda functions, separate IAM role (cloned, not shared), separate API Gateway. No feature flags, no shared code paths, no shared runtime resources.

**Consequences:**
- Positive: proven patterns (LangGraph ingestion orchestration, OpenSearch hybrid search, Bedrock embeddings) reused with confidence; zero risk of a Financial-side change breaking the independently owned Legal POC.
- Positive: any dead-code or dead-infrastructure investigation on the Financial side can proceed without needing Legal-side sign-off, since nothing is shared except (currently) the OpenSearch domain host and the LangSmith API key.
- Negative: some duplication of code and infrastructure that a shared-library approach would have avoided; accepted as the right tradeoff for two independently evolving demo projects with different stakeholders.
- Ongoing discipline required: any resource touched during Financial-side work must first be confirmed as Financial-owned (naming convention: `-financial` suffix on Lambda functions and resources) before modification.

---

## ADR-002: Hybrid search via OpenSearch + Cohere Embed v4 + Cohere Rerank v3.5

**Status:** Accepted

**Context:** Financial 10-K questions frequently require both semantic understanding (e.g., "revenue trend" implying multiple related figures) and exact lexical matching (specific line-item labels like "Total net sales" vs. "Total revenue" vs. "Net sales"). Pure vector search alone risks surfacing semantically-similar-but-wrong figures (a known failure mode with financial tables, where many numbers cluster closely in embedding space).

**Decision:** Use OpenSearch's hybrid search pipeline (`hybrid-search-pipeline`) combining a lexical `multi_match` query (weighted toward `hyq_questions` at 2.0x over raw `text` at 1.0x) with a k-NN vector query, then apply Cohere Rerank v3.5 as a second-stage reranker (`top_n=8`, increased from an initial 5) before passing chunks to answer generation.

**Consequences:**
- Positive: combines the recall benefits of semantic search with the precision benefits of lexical/HyQ matching, which is well suited to structured financial data where exact terminology matters.
- Positive: the two-stage retrieve-then-rerank pattern lets a wide net (`top_k=15` candidates) be cast cheaply, with the more expensive rerank step applied only to the reduced candidate set.
- Negative/risk (materialized and fixed): rerank output must be trusted downstream without further filtering that could silently discard correctly-ranked chunks (see ADR-004).
- Cohere Embed v4 defaults to 1536 output dimensions when unspecified — this happens to match the existing index mapping, so no explicit dimension configuration was required, but this is an implicit dependency worth documenting rather than relying on coincidence long-term.

---

## ADR-003: Table-aware chunking strategy for financial documents

**Status:** Accepted

**Context:** Financial 10-K filings are dominated by dense tabular data (income statements, segment breakdowns, geographic revenue tables). A generic semantic chunking strategy (`chunk_semantic()`) has no `block_type` awareness and was found to split table rows mid-figure, corrupting the numeric data before it ever reached the index. Additionally, short table row labels (e.g., segment identifiers, single-word line items) were being silently dropped by a minimum-token filter designed for prose content.

**Decision:** Use a `table_aware` chunking strategy for the PDF/HTML parsing pipeline (`chunker.py`), and explicitly exempt `TABLE`-type blocks from the `MIN_TOKENS` filter that would otherwise discard short but meaningful table rows.

**Consequences:**
- Positive: table structure (and the row/column relationships critical to correctly reading a financial statement) is preserved intact through chunking, rather than being fragmented across chunk boundaries.
- Positive: short but load-bearing labels (e.g., a segment name that's only 1–2 tokens) survive into the index instead of being silently dropped.
- Accepted known limitation: a `section='Contingencies'` mislabel persists on certain page-65 blocks because a segment-data sub-heading doesn't extract as a distinct text line in the source HTML — confirmed via raw page inspection as a genuine source-formatting quirk, not a parser defect, and not worth further engineering effort for the POC's purposes.

---

## ADR-004: Chunk deduplication keyed on `source` + `chunk_index`, not `source` + `page`

**Status:** Accepted (revised from an earlier, defective implementation — see history below)

**Context:** `run_dedup` in `query_graph.py` is a pipeline stage between rerank and answer generation, intended to remove genuinely duplicate chunks (e.g., re-ingestion artifacts) before context is assembled for the LLM. The original implementation keyed on `f"{c['source']}_{c['page']}"`. Because the `page` field is not meaningfully populated at ingestion (confirmed `None` across every chunk in the index, for every source document), this key collapsed to effectively `f"{source}_None"` — meaning dedup was silently keeping only **one chunk per source document**, discarding every other chunk that survived rerank regardless of its actual relevance.

This defect was masked for Apple (whose "Total net sales" figure is redundantly repeated across nearly every Apple chunk, so whichever single chunk survived still happened to carry the right number) but caused incorrect or incomplete answers for Microsoft and Amazon, where the correct "Total revenue"/"Total net sales" figure lived in a chunk distinct from other highly-ranked chunks (e.g., Microsoft Cloud revenue).

**Decision:** Key deduplication on `f"{c['source']}_{c['chunk_index']}"`, using the chunk-level `chunk_index` field that is already populated at ingestion time and unique per chunk within a source document. This required a companion fix in `run_search`: the chunk dict it builds copies OpenSearch's `source_key` field but renames it to `'source'`, and was found to never copy `chunk_index` into the chunk dict at all despite fetching it in the `_source` projection — that field simply never survived from OpenSearch's response into the in-memory chunk object.

**Consequences:**
- Positive: dedup now correctly treats each chunk as distinct unless it's a genuine duplicate, verified end-to-end via live API tests for all three demo companies (Apple, Microsoft, Amazon) producing correct multi-year revenue figures with correct labeling and arithmetic.
- Tradeoff: dedup by `chunk_index` alone is effectively a no-op against any case where the *same fact* legitimately appears in two *different* real chunks (which does happen in 10-Ks, e.g., a segment table and a summary table both stating the same total). This is an accepted tradeoff — correctness of "don't wrongly discard a needed chunk" was prioritized over "collapse literal content duplicates," which was never dedup's primary purpose here.
- Process learning: this defect was only fully caught by chaining the real `run_search → run_rerank → run_dedup` functions together in a local test using live retrieved data — an earlier unit test using hand-built stand-in dictionaries passed cleanly because it encoded the same (incorrect) field-naming assumption as the fix, rather than the pipeline's actual output shape.
- Open item: the underlying reason `page` is never populated at ingestion has not been investigated. It doesn't currently block anything (dedup no longer depends on it), but would matter if page-level citation to the end user is ever required.

---

## ADR-005: Serverless compute (AWS Lambda) for ingestion and query, not persistent EC2/containers

**Status:** Accepted

**Context:** The RAG pipeline's ingestion and query operations are triggered on-demand (document upload, user question) rather than running continuously. Both operations have well-defined start/end points and don't require long-lived state between invocations.

**Decision:** Deploy `rag-ingest-financial` and `rag-query-financial` as separate AWS Lambda functions (Python 3.11, 900-second timeout, 3008 MB memory), fronted by API Gateway (`rag-query-financial-api`) for the query path.

**Consequences:**
- Positive: cost-conscious for a POC with intermittent, demo-driven usage patterns — no idle compute cost between invocations, aligning with the project's cost-conscious AWS architecture preference.
- Positive: clean separation of ingestion and query concerns as independently deployable, independently scalable units.
- Negative/operational cost: Windows-built Lambda packages require forcing Linux-compatible `manylinux2014_x86_64` wheels via pip when Docker isn't available, adding a packaging step; packages exceeding the direct-upload size limit (~51.6 MB observed ceiling) must be staged via S3 rather than uploaded directly — both are now standing operational knowledge, not open risks.
- Negative/operational risk (materialized): the staging discipline required to keep `package/` (the actual zip source) in sync with the edited source file is manual and has caused at least one deploy where the staged code was stale relative to the source — mitigated by an explicit "always verify the live deployed code via download-and-grep, never trust `LastModified` alone" practice, now standing discipline for this project.

---

## ADR-006: OpenSearch fine-grained access control (FGAC) with explicit role mapping

**Status:** Accepted

**Context:** The Financial pipeline's Lambda functions run under their own IAM role (`rag-financial-lambda-role`, cloned from the Legal pipeline's policy set) but share the same underlying OpenSearch domain (`rag-vectors`) as the Legal pipeline. OpenSearch's FGAC model does not automatically grant permissions to a new IAM identity just because it has AWS-level IAM permissions — internal OpenSearch role mapping is a separate, required step.

**Decision:** Explicitly map `rag-financial-lambda-role`'s ARN to an internal OpenSearch role (`all_access`) via the OpenSearch security API, using the domain's basic-auth admin credentials, alongside the existing Legal pipeline's role mapping (not replacing it).

**Consequences:**
- Positive: correct, minimal-friction access without needing to fork or duplicate the OpenSearch domain itself, keeping infrastructure cost down (a second full domain would materially increase cost for a POC).
- Positive: role mappings for Legal and Financial coexist independently — rotating or investigating one does not require touching the other (confirmed: `rag-query`, the Legal function, uses IAM SigV4/AWS4Auth rather than the rotated username/password secret, so credential rotation on the Financial side posed no cross-project risk).
- Negative/shared-resource risk: the domain itself (`rag-vectors`, `t3.small.search`, single-node) remains a single point of shared infrastructure between two independently owned projects — an outage or misconfiguration at the domain level would affect both, even though indices, roles, and Lambda functions are otherwise isolated.

---

## ADR-007: API Gateway key enforcement on the query endpoint

**Status:** Accepted (verified live)

**Context:** The query endpoint (`<API_ID>/prod`) is publicly routable and needed some form of access control appropriate to a POC (not requiring full Cognito/OAuth investment, but also not left fully open).

**Decision:** Enable API Gateway's built-in `apiKeyRequired=true` enforcement on the query resource, with the actual key value held in SSM Parameter Store rather than committed to source or `.env` files directly.

**Consequences:**
- Positive: lightweight, AWS-native access control appropriate to POC scope, with no additional compute or service cost.
- Positive: verified live end-to-end (a request without a valid key is correctly rejected), closing an earlier open question about whether enforcement was genuinely active or merely configured.
- Accepted scope limitation: this protects the RAG query Lambda specifically; it does not protect the separate agent-facing ALB endpoint (`financial-research-agent-alb`), which currently has no auth gate at all — a disclosed, known gap at the agent layer, tracked separately from this RAG-layer decision.

---

## ADR-008: Single-string company extraction and filtering (accepted POC-scope limitation)

**Status:** Accepted for POC scope; flagged as a known limitation, not scheduled for immediate rework

**Context:** The query pipeline supports scoping retrieval to a specific company via a `post_filter` (originally a single `wildcard` match on `source_key`, later widened to a `bool`/`should` combining a ticker-based wildcard and a raw-company-name wildcard for more forgiving matching). The company value itself is derived upstream by a single-string extraction (`_extract_company()`), which can only return one company per query.

**Decision:** Accept single-company filtering as sufficient for the POC's demo scope. Cross-company comparison questions (e.g., comparing Apple vs. Microsoft revenue) are handled at the agent orchestration layer (multiple RAG calls, one per company) rather than by making the RAG layer itself multi-company-aware in a single call.

**Consequences:**
- Positive: keeps the RAG layer's filtering logic simple and predictable for the POC's demonstrated scenarios.
- Negative/known gap: if a single natural-language question mentions multiple companies, only one company's filter is applied at the RAG layer on the first retrieval pass — this is explicitly disclosed as a known limitation in demo framing rather than hidden, consistent with the project's "known gaps as evidence of deep understanding" demo strategy.
- The `post_filter` widening (ticker OR raw company name) was bundled into the same commit as an unrelated `top_n` change and was not independently unit-tested in isolation before being relied upon — it has since been exercised successfully in live tests (correct company-scoped results for all three companies), but is flagged for a dedicated verification pass for commit-hygiene reasons, not because of an observed failure.

---

## Summary table

| ADR | Decision | Status |
|---|---|---|
| 001 | Fork Legal RAG pipeline, enforce full infrastructure isolation | Accepted |
| 002 | Hybrid search: OpenSearch + Cohere Embed v4 + Cohere Rerank v3.5 | Accepted |
| 003 | Table-aware chunking, TABLE blocks exempt from MIN_TOKENS filter | Accepted |
| 004 | Dedup key: `source` + `chunk_index`, not `source` + `page` | Accepted (revised after production defect) |
| 005 | Serverless (Lambda) for ingestion and query, not persistent compute | Accepted |
| 006 | OpenSearch FGAC with explicit per-role mapping, shared domain | Accepted |
| 007 | API Gateway key enforcement on query endpoint | Accepted |
| 008 | Single-company extraction/filtering at RAG layer | Accepted for POC scope |

---

*As of September 12, 2026. Reflects decisions and fixes confirmed through live verification.*
