# query_graph.py
import json
import boto3
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from llm_clients import embed_query, get_llm
from opensearch_client import get_opensearch_client
from config import get_logger
from datetime import datetime, timezone
logger = get_logger("query")

# ── CORS Headers ──────────────────────────────────────────────────────────────
CORS_HEADERS = {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type,x-api-key,Authorization',
    'Access-Control-Allow-Methods': 'POST,OPTIONS'
}

# ── Bedrock client (reranking only — Cohere not in LangChain-AWS yet) ─────────
bedrock_client = boto3.client('bedrock-runtime',
    region_name=os.environ.get('AWS_REGION', 'us-east-1'))


# ── Step 1: HyDE ──────────────────────────────────────────────────────────────

hyde_prompt = PromptTemplate.from_template("""Write a short factual passage that would directly answer this question.
Write as if it's extracted from a company's SEC 10-K filing. 2-3 sentences only.

Question: {question}

Passage:""")

hyde_chain = hyde_prompt | get_llm() | StrOutputParser()

def run_hyde(state: dict) -> dict:
    try:
        hyde_text = hyde_chain.invoke({"question": state["question"]})
        logger.info("hyde complete", extra={"step": "run_hyde", "hyde_preview": hyde_text[:100]})
        state["hyde_text"] = hyde_text
    except Exception as e:
        logger.warning("hyde failed", extra={"step": "run_hyde", "error": str(e)})
        state["hyde_text"] = state["question"]
    return state


# ── Step 2: Embed + Hybrid Search ─────────────────────────────────────────────

def run_search(state: dict) -> dict:
    question_vector = embed_query(state["hyde_text"])
    question = state["question"]
    company = state.get("company", "")
    top_k = 15  # A1: increased from 10 — larger candidate pool for reranker

    query = {
        "size": top_k,
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "multi_match": {
                            "query": question,
                            "fields": ["text^1.0", "hyq_questions^2.0"]
                        }
                    },
                    {
                        "knn": {
                            "embedding": {
                                "vector": question_vector,
                                "k": top_k
                            }
                        }
                    }
                ]
            }
        },
        "_source": ["text", "source_key", "page", "chunk_index", "hyq_questions", "section"]
    }
    
    COMPANY_TICKER_MAP = {
        "apple": "aapl",
        "microsoft": "msft",
        "amazon": "amzn",
    }

    if company:
        ticker = COMPANY_TICKER_MAP.get(company.strip().lower(), company.lower())
        query["post_filter"] = {
            "bool": {
                "should": [
                    {"wildcard": {"source_key": f"*{ticker}*"}},
                    {"wildcard": {"source_key": f"*{company.strip().lower()}*"}}
                ]
            }
        }

    client = get_opensearch_client()
    response = client.search(
        index='rag-chunks-financial',
        body=query,
        params={"search_pipeline": "hybrid-search-pipeline"}
    )

    hits = response['hits']['hits']
    logger.info("hybrid search complete", extra={"step": "run_search", "candidates": len(hits)})
    state["chunks"] = [
        {
            'text': h['_source']['text'],
            'source': h['_source']['source_key'],
            'page': h['_source'].get('page'),
            'chunk_index': h['_source'].get('chunk_index'),
            'section': h['_source'].get('section', ''),
            'score': h['_score']
        }
        for h in hits
    ]
    return state


# ── Step 3: Rerank (Cohere — direct Bedrock, no LangChain wrapper) ────────────

def run_rerank(state: dict) -> dict:
    chunks = state["chunks"]
    question = state["question"]
    top_n = 8  # A2: increased from 5 — pass more chunks to answer generation

    if not chunks:
        return state

    try:
        response = bedrock_client.invoke_model(
            modelId='cohere.rerank-v3-5:0',
            body=json.dumps({
                "query": question,
                "documents": [c['text'] for c in chunks],
                "top_n": top_n,
                "api_version": 2
            })
        )
        results = json.loads(response['body'].read()).get('results', [])
        reranked = []
        for item in results:
            c = chunks[item['index']].copy()
            c['rerank_score'] = item['relevance_score']
            c['original_score'] = c['score']
            c['score'] = item['relevance_score']
            reranked.append(c)
        logger.info("rerank complete", extra={"step": "run_rerank", "chunks_in": len(chunks), "chunks_out": len(reranked)})
        state["chunks"] = reranked
    except Exception as e:
        logger.warning("rerank failed", extra={"step": "run_rerank", "error": str(e)})
        state["chunks"] = chunks[:top_n]

    return state


# ── Step 4: Deduplicate by page ───────────────────────────────────────────────

def run_dedup(state: dict) -> dict:
    seen, unique = set(), []
    for c in state["chunks"]:
        key = f"{c['source']}_{c['chunk_index']}"
        if key not in seen:
            seen.add(key)
            unique.append(c)
    logger.info("dedup complete", extra={"step": "run_dedup", "unique_chunks": len(unique)})
    state["chunks"] = unique
    return state


# ── Step 5: Answer Generation ─────────────────────────────────────────────────

answer_prompt = PromptTemplate.from_template("""You are a helpful assistant. Answer the question based only on the provided context.

When the context contains financial figures, identify the EXACT line item label (e.g. "Total net sales", "Net sales by segment", "Products", "Services") and the EXACT reporting period (e.g. fiscal year, quarter) associated with each number before using it. If a chunk contains multiple similar figures and it is not clear which one matches the question, say so explicitly rather than guessing.

When comparing a figure across two periods (e.g. "did revenue increase or decrease"), you MUST follow this exact procedure before writing your answer:
1. State the earlier period's figure.
2. State the later period's figure.
3. Compute and state the numeric difference (later minus earlier).
4. Only after stating the difference, choose "increased" if the difference is positive or "declined"/"decreased" if the difference is negative. Never state "increased" or "declined" without first showing the two numbers and the difference that justifies it.

Context:
{context}

Question: {question}

If the exact requested time period or granularity is not in the context (e.g. the question asks for a specific quarter but the context only contains annual/fiscal-year figures), do NOT refuse. Instead, state the closest available figure explicitly, and clearly note the granularity mismatch (e.g. "This filing reports annual figures, not quarterly. The most recent fiscal year total was...").

Only say "I cannot find this information in the provided documents." if no related figure exists in the context at all, regardless of period or granularity. Do not say this and then also provide a fallback figure in the same answer — if you have a fallback figure to offer, lead directly with that figure and the granularity caveat, and omit the refusal sentence entirely.
Answer:""")

answer_chain = answer_prompt | get_llm() | StrOutputParser()

# Token budget constant — cap context passed to LLM
CONTEXT_TOKEN_BUDGET = 3000  # A3: increased from 2000 — Haiku 4.5 handles larger context
CHARS_PER_TOKEN = 4  # rough approximation: 1 token ≈ 4 characters

def run_answer(state: dict) -> dict:
    # ── Token budget enforcement ──────────────────────────────────────────────
    # Build context chunks one by one, stopping when budget is exceeded.
    # Prevents runaway prompt sizes and controls costs.
    budget_chars = CONTEXT_TOKEN_BUDGET * CHARS_PER_TOKEN
    context_parts = []
    used_chars = 0

    for c in state["chunks"]:
        part = f"Source: {c['source']} (Page {c['page']}, Section: {c['section']})\n{c['text']}"
        if used_chars + len(part) > budget_chars:
            logger.warning("token budget reached", extra={
                "step": "run_answer",
                "chunks_used": len(context_parts),
                "chunks_total": len(state["chunks"]),
                "budget_chars": budget_chars,
                "used_chars": used_chars
            })
            break
        context_parts.append(part)
        used_chars += len(part)

    context = "\n\n".join(context_parts)
    logger.info("context built", extra={
        "step": "run_answer",
        "chunks_used": len(context_parts),
        "chunks_total": len(state["chunks"]),
        "estimated_tokens": used_chars // CHARS_PER_TOKEN
    })

    state["answer"] = answer_chain.invoke({
        "context": context,
        "question": state["question"]
    })
    logger.info("answer generated", extra={"step": "run_answer", "question_preview": state['question'][:80]})
    return state


# ── Full pipeline (LCEL chain of RunnableLambdas) ─────────────────────────────

query_pipeline = (
    RunnableLambda(run_hyde)
    | RunnableLambda(run_search)
    | RunnableLambda(run_rerank)
    | RunnableLambda(run_dedup)
    | RunnableLambda(run_answer)
)


def health_check() -> dict:
    checks = {}
    overall = "healthy"

    # Check OpenSearch
    try:
        client = get_opensearch_client()
        client.cluster.health()
        checks["opensearch"] = "connected"
    except Exception as e:
        checks["opensearch"] = "unreachable"
        overall = "degraded"
        logger.warning("health check opensearch failed", extra={"error": str(e)})

    return {
        "status": overall,
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ── Lambda handler ────────────────────────────────────────────────────────────

def lambda_handler(event, context):

    logger.info("incoming event", extra={
        "httpMethod": event.get('httpMethod'),
        "path": event.get('path'),
        "resource": event.get('resource')
    })
        
        
    # Health check — no auth required
    if event.get('httpMethod') == 'GET' and event.get('path') == '/health':
        result = health_check()
        status_code = 200 if result["status"] == "healthy" else 503
        return {
            'statusCode': status_code,
            'headers': CORS_HEADERS,
            'body': json.dumps(result)
        }
        
    if event.get('httpMethod') == 'OPTIONS':
        return {'statusCode': 200, 'headers': CORS_HEADERS, 'body': ''}

    if isinstance(event.get('body'), str):
        body = json.loads(event.get('body', '{}'))
    elif isinstance(event.get('body'), dict):
        body = event.get('body', {})
    else:
        body = event

    question = body.get('question', '')

    # ── Input Validation ──────────────────────────────────────────────────────────
    if not question or not isinstance(question, str):
        logger.warning("validation failed", extra={"reason": "missing or non-string question"})
        return {
            'statusCode': 400,
            'headers': CORS_HEADERS,
            'body': json.dumps({'error': 'question is required and must be a string'})
        }

    question = question.strip()
    company = body.get('company', '').strip().lower()

    if len(question) == 0:
        logger.warning("validation failed", extra={"reason": "blank question after strip"})
        return {
            'statusCode': 400,
            'headers': CORS_HEADERS,
            'body': json.dumps({'error': 'question cannot be blank'})
        }

    if len(question) < 3:
        logger.warning("validation failed", extra={"reason": "question too short", "length": len(question)})
        return {
            'statusCode': 400,
            'headers': CORS_HEADERS,
            'body': json.dumps({'error': 'question too short (minimum 3 characters)'})
        }

    if len(question) > 1000:
        logger.warning("validation failed", extra={"reason": "question too long", "length": len(question)})
        return {
            'statusCode': 400,
            'headers': CORS_HEADERS,
            'body': json.dumps({'error': 'question too long (maximum 1000 characters)'})
        }
    logger.info("query received", extra={"question_preview": question[:80]})

    state = query_pipeline.invoke({"question": question, "company": company})

    return {
        'statusCode': 200,
        'headers': CORS_HEADERS,
        'body': json.dumps({
            'question': state["question"],
            'answer': state["answer"],
            'context': [c['text'] for c in state["chunks"]],
            'sources': [
                {
                    'source': c['source'],
                    'page': c['page'],
                    'section': c.get('section', ''),
                    'score': c['score']
                }
                for c in state["chunks"]
            ]
        })
    }
