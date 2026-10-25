# ingestion_graph.py
import json
import boto3
import os
import tempfile
import hashlib
import time
from datetime import datetime, timezone
from typing import TypedDict, Optional
from urllib.parse import unquote_plus

from langgraph.graph import StateGraph, END

from detector import detect
from parsers import parse
from chunker import chunk
from opensearch_client import get_opensearch_client
from llm_clients import embed_query, get_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import get_opensearch_endpoint, get_logger
logger = get_logger("ingestion")

s3_client = boto3.client('s3')
bedrock_client = boto3.client('bedrock-runtime',
    region_name=os.environ.get('AWS_REGION', 'us-east-1'))
dynamodb = boto3.resource('dynamodb', region_name=os.environ.get('AWS_REGION', 'us-east-1'))

METADATA_TABLE   = os.environ.get('METADATA_TABLE', 'rag-metadata-financial')
PROCESSED_BUCKET = os.environ.get('PROCESSED_BUCKET', 'rag-processed-<ACCOUNT_ID>')
HASH_INDEX_NAME  = 'content-hash-index'


# ── State ─────────────────────────────────────────────────────────────────────
# LangGraph passes this dict between every node.
# Each node reads what it needs and writes its outputs back in.

class IngestionState(TypedDict):
    # inputs
    bucket_name:  str
    object_key:   str
    # computed along the pipeline
    file_bytes:   Optional[bytes]
    content_hash: Optional[str]
    file_type:    Optional[str]
    blocks:       Optional[list]
    chunks:       Optional[list]
    embeddings:   Optional[list]
    # control flow
    status:       Optional[str]   # completed | duplicate | rejected | failed
    skip_reason:  Optional[str]
    error:        Optional[str]


# ── Node 1: Download ──────────────────────────────────────────────────────────

def node_download(state: IngestionState) -> IngestionState:
    logger.info("node_download started", extra={"node": "node_download", "object_key": state['object_key'], "bucket": state['bucket_name']})
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
        s3_client.download_file(state['bucket_name'], state['object_key'], tmp_path)
    with open(tmp_path, 'rb') as f:
        file_bytes = f.read()
    os.unlink(tmp_path)
    logger.info("node_download complete", extra={"node": "node_download", "bytes": len(file_bytes)})
    return {**state, "file_bytes": file_bytes}


# ── Node 2: Duplicate Check ───────────────────────────────────────────────────

def node_check_duplicate(state: IngestionState) -> IngestionState:
    content_hash = hashlib.sha256(state['file_bytes']).hexdigest()
    logger.info("node_check_duplicate", extra={"node": "node_check_duplicate", "hash_prefix": content_hash[:20]})

    table = dynamodb.Table(METADATA_TABLE)
    response = table.query(
        IndexName=HASH_INDEX_NAME,
        KeyConditionExpression=boto3.dynamodb.conditions.Key('content_hash').eq(content_hash)
    )
    items = response.get('Items', [])

    if items:
        existing = items[0]
        logger.warning("duplicate detected", extra={"node": "node_check_duplicate", "existing_document_id": existing['document_id'], "status": "duplicate"})
        return {**state, "content_hash": content_hash, "status": "duplicate"}

    return {**state, "content_hash": content_hash}


# ── Node 3: Detect File Type ──────────────────────────────────────────────────

def node_detect(state: IngestionState) -> IngestionState:
    filename = state['object_key'].split('/')[-1]
    result = detect(state['file_bytes'], filename)
    logger.info("node_detect complete", extra={"node": "node_detect", "file_type": result.parser_key, "confidence": result.confidence})

    if result.parser_key == 'unknown':
        logger.warning("unknown file type", extra={"node": "node_detect", "object_key": state['object_key'], "status": "rejected"})
        _save_metadata(state['object_key'], 'unknown', 0, 0, 'rejected',
                       state['content_hash'],
                       error=f"unknown_file_type:{result.confidence}")
        return {**state, "file_type": "unknown", "status": "rejected"}

    return {**state, "file_type": result.parser_key}


# ── Node 4: Parse ─────────────────────────────────────────────────────────────

def node_parse(state: IngestionState) -> IngestionState:
    blocks = parse(state['file_bytes'], state['file_type'])
    logger.info("node_parse complete", extra={"node": "node_parse", "blocks": len(blocks)})


    if not blocks:
        logger.error("no blocks extracted", extra={"node": "node_parse", "object_key": state['object_key'], "status": "failed"})

        _save_metadata(state['object_key'], state['file_type'], 0, 0, 'failed',
                       state['content_hash'], error='no_blocks_extracted')
        return {**state, "blocks": [], "status": "failed", "skip_reason": "no_blocks_extracted"}

    return {**state, "blocks": blocks}


# ── Node 5: Chunk ─────────────────────────────────────────────────────────────

def node_chunk(state: IngestionState) -> IngestionState:
    chunks = chunk(state['blocks'], state['file_type'])
    for c in chunks:
        c.source_key = state['object_key']
    logger.info("node_chunk complete", extra={"node": "node_chunk", "blocks_in": len(state['blocks']), "chunks_out": len(chunks)})
    return {**state, "chunks": chunks}


# ── Node 6: Embed + HyQ ───────────────────────────────────────────────────────

hyq_prompt = PromptTemplate.from_template("""Given the following text passage, generate exactly 3 questions that this passage directly answers.
Return ONLY the questions, one per line, with no numbering, bullets, or extra text.

Passage:
{chunk_text}

Questions:""")

hyq_chain = hyq_prompt | get_llm() | StrOutputParser()

def _embed_with_retry(text: str, max_retries: int = 3) -> list:
    for attempt in range(max_retries):
        try:
            return embed_query(text)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning("bedrock embed retry", extra={"node": "node_embed", "attempt": attempt+1, "wait_seconds": wait, "error": str(e)})
            time.sleep(wait)

def _generate_hyq(chunk_text: str) -> list[str]:
    try:
        raw = hyq_chain.invoke({"chunk_text": chunk_text})
        questions = [q.strip() for q in raw.strip().split('\n') if q.strip()]
        return [q for q in questions if q.endswith('?')][:3]
    except Exception as e:
        logger.warning("hyq failed", extra={"node": "node_embed", "error": str(e)})
        return []

def node_embed(state: IngestionState) -> IngestionState:
    chunks = state['chunks']
    logger.info("node_embed started", extra={"node": "node_embed", "chunks": len(chunks)})
    embeddings = []
    for i, c in enumerate(chunks):

        embedding  = _embed_with_retry(c.text)
        hyq        = _generate_hyq(c.text)
        logger.info("chunk embedded", extra={"node": "node_embed", "chunk": i+1, "tokens": c.token_count, "page": c.page, "hyq_count": len(hyq)})

        embeddings.append({
            'chunk_index':   c.chunk_index,
            'text':          c.text,
            'embedding':     embedding,
            'token_count':   c.token_count,
            'page':          c.page,
            'source_key':    c.source_key,
            'hyq_questions': ' | '.join(hyq),
            'section':       getattr(c, 'section', '') or '',
        })
    total_hyq = sum(1 for e in embeddings if e['hyq_questions']) * 3
    logger.info("node_embed complete", extra={"node": "node_embed", "embeddings": len(embeddings), "total_hyq": total_hyq})
    return {**state, "embeddings": embeddings}


# ── Node 7: Store ─────────────────────────────────────────────────────────────

def node_store(state: IngestionState) -> IngestionState:
    client = get_opensearch_client()
    object_key = state['object_key']

    # Delete stale chunks
    try:
        response = client.delete_by_query(
            index='rag-chunks-financial',
            body={"query": {"term": {"source_key": object_key}}}
        )
        deleted = response.get('deleted', 0)
        if deleted:
            logger.info("stale chunks deleted", extra={"node": "node_store", "deleted": deleted, "object_key": object_key})
        else:
            logger.info("no stale chunks", extra={"node": "node_store", "object_key": object_key})
    except Exception as e:
        logger.warning("delete stale chunks failed", extra={"node": "node_store", "error": str(e)})

    # Index new chunks
    for item in state['embeddings']:
        client.index(index='rag-chunks-financial', body={
            'text':          item['text'],
            'embedding':     item['embedding'],
            'source_key':    item['source_key'],
            'page':          item['page'],
            'chunk_index':   item['chunk_index'],
            'hyq_questions': item['hyq_questions'],
            'section':       item['section'],
        })
    logger.info("node_store complete", extra={"node": "node_store", "chunks_stored": len(state['embeddings']), "object_key": object_key})


    # Save metadata
    _save_metadata(object_key, state['file_type'],
                   len(state['blocks']), len(state['chunks']),
                   'completed', state['content_hash'])

    # Backup chunks JSON to S3
    output_key  = f"chunks/{object_key}.json"
    chunks_data = [c.__dict__ for c in state['chunks']]
    s3_client.put_object(
        Bucket=PROCESSED_BUCKET,
        Key=output_key,
        Body=json.dumps(chunks_data, indent=2),
        ContentType='application/json'
    )

    # Summary
    logger.info("pipeline complete", extra={
    "node":        "node_store",
    "object_key":  object_key,
    "file_type":   state['file_type'],
    "blocks":      len(state['blocks']),
    "chunks":      len(state['chunks']),
    "hash_prefix": state['content_hash'][:20],
    "s3_backup":   f"s3://{PROCESSED_BUCKET}/{output_key}",
    "status":      "completed"
})

    return {**state, "status": "completed"}


# ── Routing ───────────────────────────────────────────────────────────────────
# This is what makes it LangGraph instead of a plain chain.
# After duplicate check and detect, we branch instead of continuing blindly.

def route_after_duplicate(state: IngestionState) -> str:
    return END if state.get("status") == "duplicate" else "node_detect"

def route_after_detect(state: IngestionState) -> str:
    return END if state.get("status") == "rejected" else "node_parse"

def route_after_parse(state: IngestionState) -> str:
    return END if state.get("status") == "failed" else "node_chunk"


# ── Build Graph ───────────────────────────────────────────────────────────────

def build_ingestion_graph():
    graph = StateGraph(IngestionState)

    graph.add_node("node_download",         node_download)
    graph.add_node("node_check_duplicate",  node_check_duplicate)
    graph.add_node("node_detect",           node_detect)
    graph.add_node("node_parse",            node_parse)
    graph.add_node("node_chunk",            node_chunk)
    graph.add_node("node_embed",            node_embed)
    graph.add_node("node_store",            node_store)

    graph.set_entry_point("node_download")

    graph.add_edge("node_download",        "node_check_duplicate")
    graph.add_conditional_edges("node_check_duplicate", route_after_duplicate,
                                 {"node_detect": "node_detect", END: END})
    graph.add_conditional_edges("node_detect", route_after_detect,
                                 {"node_parse": "node_parse", END: END})
    graph.add_conditional_edges("node_parse", route_after_parse,
                                 {"node_chunk": "node_chunk", END: END})
    graph.add_edge("node_chunk",  "node_embed")
    graph.add_edge("node_embed",  "node_store")
    graph.add_edge("node_store",  END)

    return graph.compile()

ingestion_graph = build_ingestion_graph()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_metadata(object_key, file_type, blocks, chunks, status, content_hash, error=None):
    dynamodb.Table(METADATA_TABLE).put_item(Item={
        'document_key':  object_key,
        'document_id':   object_key,
        'status':        status,
        'file_type':     file_type,
        'blocks':        blocks,
        'chunks':        chunks,
        'timestamp':     datetime.now(timezone.utc).isoformat(),
        'content_hash':  content_hash,
        'error':         error or 'none'
    })
    logger.info("dynamodb saved", extra={"node": "node_store", "object_key": object_key, "status": status})


# ── Lambda Handler ────────────────────────────────────────────────────────────

def lambda_handler(event, context):

    logger.info("lambda triggered", extra={"records": len(event['Records'])})

    processed, failed, skipped = [], [], []

    for record in event['Records']:
        try:
            body = json.loads(record['body'])

            if 'Event' in body and body['Event'] == 's3:TestEvent':
                logger.info("skipping s3 test event")
                continue

            s3_event    = body['Records'][0]
            bucket_name = s3_event['s3']['bucket']['name']
            object_key  = unquote_plus(s3_event['s3']['object']['key'])
            file_size   = s3_event['s3']['object'].get('size', 0)

            logger.info("processing record", extra={"bucket": bucket_name, "object_key": object_key, "file_size": file_size})

            final_state = ingestion_graph.invoke({
                "bucket_name":  bucket_name,
                "object_key":   object_key,
                "file_bytes":   None,
                "content_hash": None,
                "file_type":    None,
                "blocks":       None,
                "chunks":       None,
                "embeddings":   None,
                "status":       None,
                "skip_reason":  None,
                "error":        None,
            })

            status = final_state.get("status")
            if status in ("duplicate", "rejected", "failed"):
                skipped.append(object_key)
            else:
                processed.append(object_key)

        except Exception as e:
            logger.error("pipeline failed", extra={"object_key": object_key, "error": str(e)})
            failed.append(str(e))
            raise

    logger.info("batch complete", extra={"processed": len(processed), "skipped": len(skipped), "failed": len(failed)})

    return {
        'statusCode': 200,
        'processed':  len(processed),
        'skipped':    len(skipped),
        'failed':     len(failed)
    }
