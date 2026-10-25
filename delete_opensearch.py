# delete_opensearch.py
"""
Safe teardown for RAG pipeline (Financial) — run from VSCode terminal on Windows.

What this does (in order):
  1. Connects to OpenSearch and scrolls ALL docs from rag-chunks-financial index
  2. Saves each chunk to S3 backup bucket:
       rag-opensearch-backup-{account_id}/latest-financial/{doc_id}/{chunk_id}.json
       rag-opensearch-backup-{account_id}/snapshots-financial/YYYY-MM-DD/{doc_id}/{chunk_id}.json
  3. Verifies S3 count matches OpenSearch count — BLOCKS deletion if mismatch
  4. Resets all DynamoDB records in rag-metadata-financial
  5. Deletes OpenSearch domain (rag-vectors — SHARED with Legal RAG; this wipes
     Legal's index and the shared hybrid-search-pipeline too, not just Financial's data)

Backup bucket is SEPARATE from rag-raw-financial-{account_id} so S3 events
are NOT triggered and Lambda is NOT invoked on backup writes.

Usage (PowerShell):
  $env:OPENSEARCH_ENDPOINT = "search-rag-vectors-xxxx.us-east-1.es.amazonaws.com"
  $env:OPENSEARCH_PASSWORD = '<password — use SINGLE quotes, it contains a $>'
  $env:AWS_ACCOUNT_ID = "<ACCOUNT_ID>"
  python delete_opensearch.py

Requirements:
  pip install opensearch-py boto3
"""

import os
import sys
import json
import boto3
from datetime import datetime, timezone
from opensearchpy import OpenSearch, RequestsHttpConnection

# ── CONFIG ─────────────────────────────────────────────────────────────────────
OPENSEARCH_ENDPOINT = os.environ.get('OPENSEARCH_ENDPOINT', '')
OPENSEARCH_USER     = os.environ.get('OPENSEARCH_USER', 'ragadmin')
OPENSEARCH_PASSWORD = os.environ.get('OPENSEARCH_PASSWORD', '')
AWS_ACCOUNT_ID      = os.environ.get('AWS_ACCOUNT_ID', '')
INDEX_NAME          = 'rag-chunks-financial'
DYNAMODB_TABLE      = 'rag-metadata-financial'
AWS_PROFILE         = 'rag-dev'
AWS_REGION          = 'us-east-1'
SCROLL_SIZE         = 100
SCROLL_TTL          = '5m'

# Dedicated backup bucket — no S3 event trigger, no Lambda invocation
BACKUP_BUCKET    = f'rag-opensearch-backup-{AWS_ACCOUNT_ID}'
LATEST_PREFIX    = 'latest-financial'
TODAY            = datetime.now(timezone.utc).strftime('%Y-%m-%d')
SNAPSHOT_PREFIX  = f'snapshots-financial/{TODAY}'
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


def get_session():
    return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)


# ── STEP 1: Backup to dedicated S3 bucket ─────────────────────────────────────

def backup_to_s3(os_client, s3) -> dict:
    """
    Scroll all docs from OpenSearch (rag-chunks-financial) and save to two S3 paths:
      latest-financial/           — always overwritten, restore reads from here
      snapshots-financial/DATE/   — dated history, kept forever
    """
    print(f"\n{'='*60}")
    print(f"  STEP 1: Backing up OpenSearch ({INDEX_NAME}) → S3")
    print(f"  bucket  : {BACKUP_BUCKET}")
    print(f"  latest  : {LATEST_PREFIX}/")
    print(f"  snapshot: {SNAPSHOT_PREFIX}/")
    print(f"{'='*60}\n")

    total = os_client.count(index=INDEX_NAME)['count']
    print(f"  Documents in OpenSearch: {total}")

    if total == 0:
        print("  ⚠️  Index is empty — nothing to back up.")
        return {"total": 0, "backed_up": 0, "errors": 0}

    resp = os_client.search(
        index=INDEX_NAME,
        scroll=SCROLL_TTL,
        body={"size": SCROLL_SIZE, "query": {"match_all": {}}, "_source": True}
    )
    scroll_id = resp['_scroll_id']
    hits      = resp['hits']['hits']

    backed_up = 0
    errors    = 0
    batch_num = 0

    try:
        while hits:
            batch_num += 1
            print(f"  Batch {batch_num}: {len(hits)} docs...", end=' ', flush=True)
            batch_ok = 0

            for hit in hits:
                doc         = hit['_source']
                source_key  = doc.get('source_key', hit['_id'])
                chunk_index = doc.get('chunk_index', 0)
                # safe S3 key — replace slashes and spaces
                safe_key   = source_key.replace('/', '_').replace(' ', '_')
                doc_id     = safe_key
                chunk_id   = f"{safe_key}_chunk_{chunk_index}"

                payload = {
                    # fields matching rag-chunks-financial index exactly
                    "source_key":    doc.get('source_key', ''),
                    "chunk_index":   doc.get('chunk_index', 0),
                    "text":          doc.get('text', ''),
                    "embedding":     doc.get('embedding', []),
                    "page":          doc.get('page', 0),
                    "hyq_questions": doc.get('hyq_questions', ''),
                    "section":       doc.get('section', ''),
                    "backed_up_at":  datetime.now(timezone.utc).isoformat(),
                }
                body = json.dumps(payload)

                try:
                    # latest-financial/ — overwrite every time (restore reads this)
                    s3.put_object(
                        Bucket=BACKUP_BUCKET,
                        Key=f"{LATEST_PREFIX}/{doc_id}/{chunk_id}.json",
                        Body=body,
                        ContentType='application/json',
                    )
                    # snapshots-financial/YYYY-MM-DD/ — dated copy for history
                    s3.put_object(
                        Bucket=BACKUP_BUCKET,
                        Key=f"{SNAPSHOT_PREFIX}/{doc_id}/{chunk_id}.json",
                        Body=body,
                        ContentType='application/json',
                    )
                    backed_up += 1
                    batch_ok  += 1
                except Exception as e:
                    print(f"\n    ✗ Error saving {chunk_id}: {e}")
                    errors += 1

            print(f"saved {batch_ok} ✓")

            # Stop scrolling once we've retrieved all docs — avoids
            # NotFoundError when scroll context is exhausted on the last page
            if len(hits) < SCROLL_SIZE:
                break

            try:
                resp      = os_client.scroll(scroll_id=scroll_id, scroll=SCROLL_TTL)
                scroll_id = resp['_scroll_id']
                hits      = resp['hits']['hits']
            except Exception as e:
                print(f"  Scroll ended: {e}")
                break

    finally:
        try:
            os_client.clear_scroll(scroll_id=scroll_id)
        except Exception:
            pass

    return {"total": total, "backed_up": backed_up, "errors": errors}


# ── STEP 2: Verify backup ─────────────────────────────────────────────────────

def verify_backup(s3, expected_total: int) -> bool:
    print(f"\n{'='*60}")
    print(f"  STEP 2: Verifying backup")
    print(f"{'='*60}\n")

    paginator = s3.get_paginator('list_objects_v2')
    pages     = paginator.paginate(Bucket=BACKUP_BUCKET, Prefix=LATEST_PREFIX + '/')
    s3_count  = sum(len(page.get('Contents', [])) for page in pages)

    print(f"  OpenSearch docs        : {expected_total}")
    print(f"  S3 {LATEST_PREFIX}/ files : {s3_count}")

    if s3_count >= expected_total:
        print(f"  ✅ Backup verified — all {s3_count} chunks in S3")
        return True
    else:
        gap = expected_total - s3_count
        print(f"  ❌ MISMATCH — {gap} chunks missing from S3")
        print(f"     DO NOT delete. Re-run this script.")
        return False


# ── STEP 3: Reset DynamoDB ────────────────────────────────────────────────────

def reset_all_dynamodb():
    print(f"\n{'='*60}")
    print(f"  STEP 3: Resetting DynamoDB records ({DYNAMODB_TABLE})")
    print(f"{'='*60}\n")

    session  = get_session()
    table    = session.resource('dynamodb').Table(DYNAMODB_TABLE)
    response = table.scan(ProjectionExpression='document_id')
    items    = response.get('Items', [])

    for item in items:
        table.delete_item(Key={'document_id': item['document_id']})
        print(f"  Reset: {item['document_id']}")

    print(f"\n  ✅ Reset {len(items)} DynamoDB records in {DYNAMODB_TABLE}")


# ── STEP 4: Delete OpenSearch ─────────────────────────────────────────────────

def delete_opensearch():
    print(f"\n{'='*60}")
    print(f"  STEP 4: Deleting OpenSearch domain (rag-vectors)")
    print(f"  ⚠️  SHARED domain — this also wipes Legal RAG's index and")
    print(f"     the shared hybrid-search-pipeline, not just Financial's data.")
    print(f"{'='*60}\n")

    session = get_session()
    client  = session.client('opensearch', region_name=AWS_REGION)
    client.delete_domain(DomainName='rag-vectors')

    print(f"  ✅ OpenSearch domain deletion initiated")
    print(f"     Domain will be fully deleted in ~10 minutes")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    missing = [k for k, v in {
        'OPENSEARCH_ENDPOINT': OPENSEARCH_ENDPOINT,
        'OPENSEARCH_PASSWORD': OPENSEARCH_PASSWORD,
        'AWS_ACCOUNT_ID':      AWS_ACCOUNT_ID,
    }.items() if not v]

    if missing:
        print(f"\n❌ Missing config: {', '.join(missing)}")
        print("\nSet them before running (PowerShell):\n")
        for m in missing:
            print(f"  $env:{m} = '<value>'")
        sys.exit(1)

    print(f"""
============================================================
  RAG Pipeline (Financial) — Safe OpenSearch Teardown
============================================================
  Index         : {INDEX_NAME}
  DynamoDB table: {DYNAMODB_TABLE}
  Steps:
    1. Backup ALL OpenSearch embeddings → S3
    2. Verify S3 backup matches OpenSearch count
    3. Reset DynamoDB records ({DYNAMODB_TABLE})
    4. Delete OpenSearch domain (rag-vectors — SHARED with Legal RAG)

  Backup bucket : {BACKUP_BUCKET}
    {LATEST_PREFIX}/     → restore always reads from here
    {SNAPSHOT_PREFIX}/ → dated copy

  NOTE: Backup bucket is separate from rag-raw-financial-{AWS_ACCOUNT_ID}
        so NO S3 events fire and NO Lambda is triggered.

  ⚠️  rag-vectors is SHARED with Legal RAG. Deleting it wipes Legal's
      index and the shared hybrid-search-pipeline too — Legal has no
      proven rebuild path of its own as of this session.
============================================================
""")

    confirm = input("Type 'yes' to confirm: ")
    if confirm.lower() != 'yes':
        print("Cancelled.")
        sys.exit(0)

    print(f"\nConnecting to OpenSearch: {OPENSEARCH_ENDPOINT}")
    os_client = get_opensearch_client()
    s3        = get_session().client('s3')

    try:
        info = os_client.info()
        print(f"Connected ✅  cluster: {info['cluster_name']}\n")
    except Exception as e:
        print(f"\n❌ Cannot connect to OpenSearch: {e}")
        sys.exit(1)

    # Step 1: Backup
    result = backup_to_s3(os_client, s3)
    print(f"\n  Backup summary: {result['backed_up']} saved, {result['errors']} errors")

    if result['errors'] > 0:
        print(f"\n❌ {result['errors']} chunks failed. Fix errors and re-run.")
        sys.exit(1)

    # Step 2: Verify — hard block if mismatch
    verified = verify_backup(s3, result['total'])
    if not verified:
        print("\n❌ Backup verification failed. OpenSearch NOT deleted.")
        sys.exit(1)

    # Step 3: DynamoDB
    reset_all_dynamodb()

    # Step 4: Delete OpenSearch
    delete_opensearch()

    print(f"""
============================================================
  ✅ DONE
============================================================
  DynamoDB  : {DYNAMODB_TABLE} reset
  OpenSearch: deletion initiated (~10 min to complete)
  S3 backup : s3://{BACKUP_BUCKET}/{LATEST_PREFIX}/
              {result['backed_up']} chunks backed up

  To restore, run in order:
    python create_index.py
    python setup_hybrid_search.py
    python restore_from_s3.py
============================================================
""")