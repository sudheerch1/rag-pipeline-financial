import json
import boto3
import os
import tempfile
import hashlib
import time
from detector import detect
from parsers import parse
from chunker import chunk
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from config import get_opensearch_endpoint
from urllib.parse import unquote_plus
from detector import detect

s3_client = boto3.client('s3')
bedrock_client = boto3.client('bedrock-runtime', region_name=os.environ.get('AWS_REGION', 'us-east-1'))

dynamodb = boto3.resource('dynamodb')
METADATA_TABLE = os.environ.get('METADATA_TABLE', 'rag-metadata-financial')
PROCESSED_BUCKET = os.environ.get('PROCESSED_BUCKET', 'rag-processed-911414006786')
HASH_INDEX_NAME = 'content-hash-index'

OPENSEARCH_ENDPOINT = get_opensearch_endpoint()


def get_opensearch_client():
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        'us-east-1',
        'es',
        session_token=credentials.token
    )
    return OpenSearch(
        hosts=[{'host': OPENSEARCH_ENDPOINT, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )


# ── Duplicate Detection ───────────────────────────────────────────────────────

def compute_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def check_duplicate(content_hash: str) -> dict | None:
    table = dynamodb.Table(METADATA_TABLE)
    response = table.query(
        IndexName=HASH_INDEX_NAME,
        KeyConditionExpression=boto3.dynamodb.conditions.Key('content_hash').eq(content_hash)
    )
    items = response.get('Items', [])
    return items[0] if items else None


# ── HyQ: Hypothetical Question Generation ────────────────────────────────────

def generate_hyq_questions(chunk_text: str) -> list[str]:
    """
    Generate 3 hypothetical questions that this chunk answers.
    Uses Amazon Nova Micro for fast, cost-effective generation.

    HyQ improves retrieval by matching user questions against
    pre-generated questions rather than raw text — much better semantic match.

    Returns list of questions, empty list on failure (non-blocking).
    """
    prompt = f"""Given the following text passage, generate exactly 3 questions that this passage directly answers.
Return ONLY the questions, one per line, with no numbering, bullets, or extra text.

Passage:
{chunk_text}

Questions:"""

    try:
        response = bedrock_client.invoke_model(
            modelId='amazon.nova-micro-v1:0',
            body=json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ]
            })
        )
        result = json.loads(response['body'].read())
        raw = result['output']['message']['content'][0]['text']

        # Parse questions — one per line, filter empty lines
        questions = [q.strip() for q in raw.strip().split('\n') if q.strip()]
        questions = [q for q in questions if q.endswith('?')][:3]
        return questions

    except Exception as e:
        print(f"[HyQ] Generation failed for chunk (non-blocking): {str(e)}")
        return []


def embed_with_retry(text: str, max_retries: int = 3) -> list:
    for attempt in range(max_retries):
        try:
            response = bedrock_client.invoke_model(
                modelId='amazon.titan-embed-text-v1',
                body=json.dumps({'inputText': text})
            )
            return json.loads(response['body'].read())['embedding']
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"[Bedrock] Attempt {attempt + 1} failed, retrying in {wait}s: {e}")
            time.sleep(wait)


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_chunks(chunks):
    """
    Convert chunks to vectors using Bedrock Titan Embeddings.
    Also generates HyQ questions for each chunk.
    """
    embeddings = []
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        print(f"  [Embedding] Chunk {i + 1}/{total} | tokens: {chunk.token_count} | page: {chunk.page}")

        # Generate embedding (with retry + exponential backoff)
        embedding = embed_with_retry(chunk.text)

        # Generate HyQ questions
        hyq_questions = generate_hyq_questions(chunk.text)
        if hyq_questions:
            print(f"  [HyQ] Chunk {i + 1}: generated {len(hyq_questions)} questions")

        embeddings.append({
            'chunk_index': chunk.chunk_index,
            'text': chunk.text,
            'embedding': embedding,
            'token_count': chunk.token_count,
            'page': chunk.page,
            'source_key': chunk.source_key,
            'hyq_questions': ' | '.join(hyq_questions),  # stored as searchable text
            'section': getattr(chunk, 'section', '') or '',
        })
    return embeddings


# ── OpenSearch ────────────────────────────────────────────────────────────────

def store_in_opensearch(embeddings):
    client = get_opensearch_client()
    for item in embeddings:
        client.index(
            index='rag-chunks',
            body={
                'text': item['text'],
                'embedding': item['embedding'],
                'source_key': item['source_key'],
                'page': item['page'],
                'chunk_index': item['chunk_index'],
                'hyq_questions': item['hyq_questions'],
                'section': item['section'],
            }
        )
    print(f"[OpenSearch] Stored {len(embeddings)} chunks in index: rag-chunks")


# ── Delete Stale Chunks ───────────────────────────────────────────────────────

def delete_old_chunks(source_key):
    client = get_opensearch_client()
    delete_query = {
        "query": {
            "term": {
                "source_key": source_key
            }
        }
    }
    try:
        response = client.delete_by_query(
            index='rag-chunks',
            body=delete_query
        )
        deleted_count = response.get('deleted', 0)
        if deleted_count > 0:
            print(f"[OpenSearch] Deleted {deleted_count} stale chunks for: {source_key}")
        else:
            print(f"[OpenSearch] No stale chunks found for: {source_key} (first time ingestion)")
    except Exception as e:
        print(f"[OpenSearch] Warning: Could not delete old chunks for {source_key}: {e}")


# ── DynamoDB Metadata ─────────────────────────────────────────────────────────

def save_metadata(object_key, file_type, blocks, chunks, status, content_hash, error=None):
    table = dynamodb.Table(METADATA_TABLE)
    from datetime import datetime, timezone
    table.put_item(
        Item={
            'document_id': object_key,
            'status': status,
            'file_type': file_type,
            'blocks': blocks,
            'chunks': chunks,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'content_hash': content_hash,
            'error': error or 'none'
        }
    )
    print(f"[DynamoDB] Metadata saved | file: {object_key} | status: {status}")


# ── Lambda Handler ────────────────────────────────────────────────────────────

def lambda_handler(event, context):
    # ── LOG: Pipeline entry point — shows SQS triggered Lambda ──
    print("=" * 60)
    print("RAG Ingestion Pipeline — triggered")
    print(f"Records in batch: {len(event['Records'])}")
    print("=" * 60)

    processed = []
    failed = []
    skipped = []

    for record in event['Records']:
        try:
            body = json.loads(record['body'])

            if 'Event' in body and body['Event'] == 's3:TestEvent':
                print("[SQS] Skipping S3 test event")
                continue

            s3_event = body['Records'][0]
            bucket_name = s3_event['s3']['bucket']['name']
            object_key = unquote_plus(s3_event['s3']['object']['key'])
            file_size = s3_event['s3']['object'].get('size', 0)
            event_time = s3_event.get('eventTime', 'unknown')

            # ── LOG: Shows the SQS message content — what S3 event arrived ──
            print(f"[SQS] Message received")
            print(f"  Bucket     : {bucket_name}")
            print(f"  File       : {object_key}")
            print(f"  Size       : {file_size:,} bytes ({round(file_size / 1024, 1)} KB)")
            print(f"  Event time : {event_time}")

            result = process_file(bucket_name, object_key)

            if result.get('skipped'):
                skipped.append(result)
            else:
                processed.append(result)

        except Exception as e:
            print(f"[ERROR] Failed to process record: {str(e)}")
            failed.append(str(e))
            raise

    # ── LOG: Final batch summary ──
    print("=" * 60)
    print(f"BATCH COMPLETE | processed: {len(processed)} | skipped: {len(skipped)} | failed: {len(failed)}")
    print("=" * 60)

    return {
        'statusCode': 200,
        'processed': len(processed),
        'skipped': len(skipped),
        'failed': len(failed)
    }


# ── File Processing ───────────────────────────────────────────────────────────

def process_file(bucket_name, object_key):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = tmp_file.name
        s3_client.download_file(bucket_name, object_key, tmp_path)

    try:
        with open(tmp_path, 'rb') as f:
            file_bytes = f.read()

        # ── LOG: Step 1 — File downloaded ──
        print(f"[1/7] File downloaded to /tmp/ ({len(file_bytes):,} bytes)")

        # Duplicate detection
        content_hash = compute_hash(file_bytes)

        # ── LOG: Step 2 — Hash computed ──
        print(f"[2/7] SHA-256 hash computed: {content_hash[:20]}...")

        existing = check_duplicate(content_hash)
        if existing:
            # ── LOG: Duplicate detected ──
            print(f"[SKIP] DUPLICATE DETECTED — same content already ingested")
            print(f"  Existing file : {existing['document_id']}")
            print(f"  Ingested at   : {existing['timestamp']}")
            print(f"  Skipping      : {object_key}")
            return {
                'skipped': True,
                'source_key': object_key,
                'duplicate_of': existing['document_id'],
                'content_hash': content_hash
            }

        # Detect file type
        result = detect(file_bytes, object_key.split('/')[-1])
        file_type = result.parser_key

        # ── LOG: Step 3 — File type detected ──
        print(f"[3/7] File type detected: {file_type} (confidence: {result.confidence})")

        # Handle unrecognised or corrupted files cleanly
        if file_type == 'unknown':
            print(f"[SKIP] Unrecognised file type for: {object_key}")
            save_metadata(
                object_key=object_key,
                file_type='unknown',
                blocks=0,
                chunks=0,
                status='rejected',
                content_hash=content_hash,
                error=f"unknown_file_type:{result.confidence}"
            )
            return {
                'skipped': True,
                'source_key': object_key,
                'reason': 'unknown_file_type',
                'confidence': result.confidence
            }

        # Parse file into blocks
        blocks = parse(file_bytes, file_type)

        # ── LOG: Step 4 — Parsed into blocks ──
        print(f"[4/7] Parsed into {len(blocks)} paragraph-level blocks")

        # Guard: no blocks means parser got nothing useful from the file
        if not blocks:
            print(f"[ERROR] No blocks extracted from {object_key} — marking as failed")
            save_metadata(
                object_key=object_key,
                file_type=file_type,
                blocks=0,
                chunks=0,
                status='failed',
                content_hash=content_hash,
                error='no_blocks_extracted'
            )
            return {
                'skipped': True,
                'source_key': object_key,
                'reason': 'no_blocks_extracted'
            }

        # Chunk blocks
        chunks = chunk(blocks, file_type)
        for c in chunks:
            c.source_key = object_key

        # ── LOG: Step 5 — Chunks created ──
        print(f"[5/7] Semantic chunking complete → {len(chunks)} chunks from {len(blocks)} blocks")

        # Generate embeddings + HyQ questions
        print(f"[6/7] Generating Titan embeddings + HyQ questions for {len(chunks)} chunks...")
        embeddings = embed_chunks(chunks)
        total_hyq = sum(1 for e in embeddings if e['hyq_questions']) * 3

        # ── LOG: Step 6 — Embeddings done ──
        print(f"[6/7] Done — {len(embeddings)} embeddings | ~{total_hyq} HyQ questions generated")

        # Delete old chunks for this file before indexing new ones
        delete_old_chunks(object_key)

        # Store in OpenSearch
        store_in_opensearch(embeddings)

        # ── LOG: Step 7 — OpenSearch indexed ──
        print(f"[7/7] Chunks indexed in OpenSearch — pipeline complete!")

        # Save metadata
        save_metadata(
            object_key=object_key,
            file_type=file_type,
            blocks=len(blocks),
            chunks=len(chunks),
            status='completed',
            content_hash=content_hash
        )

        # Save chunks to processed bucket
        output_key = f"chunks/{object_key}.json"
        chunks_data = [chunk.__dict__ for chunk in chunks]
        s3_client.put_object(
            Bucket=PROCESSED_BUCKET,
            Key=output_key,
            Body=json.dumps(chunks_data, indent=2),
            ContentType='application/json'
        )
        print(f"[S3] Chunks JSON backed up to: s3://{PROCESSED_BUCKET}/{output_key}")

        # ── LOG: Final pipeline summary ──
        print("-" * 60)
        print(f"PIPELINE SUMMARY")
        print(f"  File     : {object_key}")
        print(f"  Type     : {file_type}")
        print(f"  Blocks   : {len(blocks)}")
        print(f"  Chunks   : {len(chunks)}")
        print(f"  HyQ Qs   : ~{total_hyq} (3 per chunk)")
        print(f"  Hash     : {content_hash[:20]}...")
        print(f"  Status   : completed ✅")
        print("-" * 60)

        return {
            'source_key': object_key,
            'file_type': file_type,
            'blocks': len(blocks),
            'chunks': len(chunks),
            'output_key': output_key,
            'content_hash': content_hash
        }

    except Exception as e:
        print(f"[ERROR] Pipeline failed for {object_key}: {str(e)}")
        save_metadata(
            object_key=object_key,
            file_type='unknown',
            blocks=0,
            chunks=0,
            status='failed',
            content_hash=locals().get('content_hash', 'unknown'),
            error=str(e)
        )
        raise

    finally:
        os.unlink(tmp_path)