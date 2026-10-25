# restore_from_s3.py
"""
Restore OpenSearch embeddings (Financial RAG) from S3 backup.
Run this locally after recreating OpenSearch manually AND after:
  1. Role mapping completed in OpenSearch Admin portal
  2. python create_index.py     (recreates rag-chunks-financial with correct mapping)
  3. python setup_hybrid_search.py   (recreates the hybrid search pipeline)

FIX APPLIED:
  - INDEX_NAME corrected from 'rag-chunks' to 'rag-chunks-financial'
  - CACHE_PREFIX corrected from 'latest' to 'latest-financial', matching
    the corrected delete_opensearch.py backup location.
  Previously this script restored into the WRONG (unrelated/empty) index,
  meaning "restore" silently did nothing useful for the financial pipeline.

What it does:
  1. Reads all chunk JSON files from S3 backup bucket (latest-financial/)
  2. Bulk indexes them into rag-chunks-financial in OpenSearch
  3. Zero embedding calls, zero HyQ/LLM calls, zero cost — this is a pure
     restore of previously-computed data, IF the backup actually ran
     correctly beforehand.

Usage:
  set OPENSEARCH_ENDPOINT=search-rag-vectors-xxxx.us-east-1.es.amazonaws.com
  set OPENSEARCH_PASSWORD=YourPassword#123
  set AWS_ACCOUNT_ID=123456789012
  python restore_from_s3.py

Requirements:
  pip install opensearch-py boto3
"""

import os
import sys
import json
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk

# ── CONFIG ─────────────────────────────────────────────────────────────────────
OPENSEARCH_ENDPOINT = os.environ.get('OPENSEARCH_ENDPOINT', '')
OPENSEARCH_USER     = os.environ.get('OPENSEARCH_USER', 'ragadmin')
OPENSEARCH_PASSWORD = os.environ.get('OPENSEARCH_PASSWORD', '')
AWS_ACCOUNT_ID      = os.environ.get('AWS_ACCOUNT_ID', '<ACCOUNT_ID>')
BACKUP_BUCKET       = f'rag-opensearch-backup-{AWS_ACCOUNT_ID}'
CACHE_PREFIX        = 'latest-financial'   # FIX: was 'latest'
INDEX_NAME          = os.environ.get('TARGET_INDEX_NAME', 'rag-chunks-financial')
# ^ Defaults to the live index for a real restore. For a test-restore
#   verification pass, set TARGET_INDEX_NAME to a scratch index
#   (e.g. rag-chunks-financial-test-restore) BEFORE running this script,
#   so you never write test data into the live index by accident.
AWS_PROFILE         = 'rag-dev'
AWS_REGION          = 'us-east-1'
BATCH_SIZE          = 100
# ──────────────────────────────────────────────────────────────────────────────


def get_opensearch_client():
    return OpenSearch(
        hosts=[{'host': OPENSEARCH_ENDPOINT, 'port': 443}],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60,
    )


def get_s3_client():
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    return session.client('s3')


def list_backup_chunks(s3) -> list:
    """List all chunk JSON files in S3 latest-financial/ prefix."""
    paginator = s3.get_paginator('list_objects_v2')
    pages     = paginator.paginate(Bucket=BACKUP_BUCKET, Prefix=CACHE_PREFIX + '/')

    keys = [
        obj['Key']
        for page in pages
        for obj in page.get('Contents', [])
        if obj['Key'].endswith('.json')
    ]
    return keys


def load_chunks_from_s3(s3, keys: list) -> list:
    """Read all chunk JSON files from S3."""
    chunks = []
    errors = 0

    for i, key in enumerate(keys):
        try:
            obj   = s3.get_object(Bucket=BACKUP_BUCKET, Key=key)
            chunk = json.loads(obj['Body'].read())
            chunks.append(chunk)
        except Exception as e:
            print(f"  ✗ Error reading {key}: {e}")
            errors += 1

        if (i + 1) % 100 == 0:
            print(f"  Loaded {i+1}/{len(keys)} chunks...")

    return chunks, errors


def bulk_index(os_client, chunks: list) -> dict:
    """Bulk index all chunks into OpenSearch rag-chunks-financial index."""
    actions = []

    for chunk in chunks:
        actions.append({
            "_index":  INDEX_NAME,
            "_id":     f"{chunk['source_key']}_{chunk['chunk_index']}",
            "_source": {
                "text":          chunk['text'],
                "embedding":     chunk['embedding'],
                "source_key":    chunk['source_key'],
                "page":          chunk['page'],
                "chunk_index":   chunk['chunk_index'],
                "hyq_questions": chunk['hyq_questions'],
                "section":       chunk.get('section', ''),
            }
        })

    indexed = 0
    errors  = 0

    for i in range(0, len(actions), BATCH_SIZE):
        batch = actions[i:i + BATCH_SIZE]
        try:
            success, failed = bulk(os_client, batch, raise_on_error=False)
            indexed += success
            errors  += len(failed) if failed else 0
            print(f"  Batch {i//BATCH_SIZE + 1}: {success} indexed ✓")
        except Exception as e:
            print(f"  Batch {i//BATCH_SIZE + 1} failed: {e}")
            errors += len(batch)

    return {"indexed": indexed, "errors": errors}


def verify_restore(os_client, expected: int):
    """Count docs in OpenSearch and compare with S3 backup count."""
    count = os_client.count(index=INDEX_NAME)['count']
    print(f"\n  S3 chunks    : {expected}")
    print(f"  OpenSearch   : {count}")

    if count >= expected:
        print(f"  ✅ Restore verified — all {count} chunks in OpenSearch")
    else:
        print(f"  ⚠️  Only {count}/{expected} chunks indexed — re-run script")


if __name__ == '__main__':

    missing = [k for k, v in {
        'OPENSEARCH_ENDPOINT': OPENSEARCH_ENDPOINT,
        'OPENSEARCH_PASSWORD': OPENSEARCH_PASSWORD,
    }.items() if not v]

    if missing:
        print(f"\n❌ Missing config: {', '.join(missing)}")
        print("\nSet them before running:")
        for m in missing:
            print(f"  set {m}=<value>")
        sys.exit(1)

    is_live_index = (INDEX_NAME == 'rag-chunks-financial')
    target_label  = "⚠️  LIVE INDEX" if is_live_index else "TEST/SCRATCH INDEX"

    print(f"""
============================================================
  RAG Pipeline (Financial) — Restore from S3 Backup
============================================================
  Source : s3://{BACKUP_BUCKET}/{CACHE_PREFIX}/
  Target : {OPENSEARCH_ENDPOINT}/{INDEX_NAME}   [{target_label}]
  Cost   : Zero Titan/Cohere calls. Zero HyQ/LLM calls.

  PREREQUISITE — run these first if the domain was just recreated:
    1. Role mapping in OpenSearch Admin portal completed
    2. python create_index.py
    3. python setup_hybrid_search.py

  Set TARGET_INDEX_NAME env var to restore into a scratch index
  instead of live (recommended for verification runs).
============================================================
""")

    if is_live_index:
        confirm = input("No TARGET_INDEX_NAME set — this will restore into the LIVE index. Type 'yes' to continue: ")
        if confirm.lower() != 'yes':
            print("Cancelled. Set TARGET_INDEX_NAME to a scratch index name and re-run for a test restore.")
            sys.exit(0)

    print(f"Connecting to OpenSearch...")
    os_client = get_opensearch_client()
    try:
        info = os_client.info()
        print(f"Connected ✅  cluster: {info['cluster_name']}\n")
    except Exception as e:
        print(f"❌ Cannot connect: {e}")
        print("Check OPENSEARCH_ENDPOINT and OPENSEARCH_PASSWORD.")
        sys.exit(1)

    s3 = get_s3_client()

    print(f"Listing chunks in s3://{BACKUP_BUCKET}/{CACHE_PREFIX}/...")
    keys = list_backup_chunks(s3)

    if not keys:
        print("❌ No chunks found in S3 backup.")
        print(f"   Run delete_opensearch.py first to create the backup,")
        print(f"   or re-trigger real ingestion (S3 upload) if no backup exists yet.")
        sys.exit(1)

    print(f"Found {len(keys)} chunks in S3 backup\n")

    print("Loading chunks from S3...")
    chunks, load_errors = load_chunks_from_s3(s3, keys)
    print(f"Loaded {len(chunks)} chunks ({load_errors} errors)\n")

    print(f"Indexing into OpenSearch ({INDEX_NAME})...")
    result = bulk_index(os_client, chunks)

    print(f"""
  Indexed : {result['indexed']}
  Errors  : {result['errors']}
""")

    verify_restore(os_client, len(keys))

    if is_live_index:
        print(f"""
============================================================
  ✅ RESTORE COMPLETE — LIVE INDEX ({INDEX_NAME})
  Your Financial RAG pipeline is ready to query.
============================================================
""")
    else:
        print(f"""
============================================================
  ✅ TEST RESTORE COMPLETE — SCRATCH INDEX ({INDEX_NAME})
  This did NOT touch the live index. Compare the chunk count
  above against the expected 441 (Apple 73, Microsoft 265,
  Amazon 103), then delete this scratch index when done:
    DELETE {OPENSEARCH_ENDPOINT}/{INDEX_NAME}
============================================================
""")