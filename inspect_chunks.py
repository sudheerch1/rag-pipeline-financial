"""
inspect_chunks.py — View raw chunks stored in OpenSearch for a given file.
Usage:
    python inspect_chunks.py                          # shows all files in index
    python inspect_chunks.py --file ABCWordDoc.docx   # shows chunks for a specific file
    python inspect_chunks.py --file ABCWordDoc.docx --limit 5  # limit to 5 chunks
"""

import argparse
import json
import sys
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
from config import get_opensearch_endpoint, get_opensearch_credentials

PROFILE = "rag-dev"

def get_opensearch_client():
    endpoint = get_opensearch_endpoint(profile_name=PROFILE)
    username, password = get_opensearch_credentials(profile_name=PROFILE)
    host = endpoint.replace("https://", "")
    client = OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=(username, password),
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )
    return client

def list_files(client):
    """Show all unique source files in the index."""
    resp = client.search(
        index="rag-chunks-financial",
        body={
            "size": 0,
            "aggs": {
                "files": {
                    "terms": {"field": "source_key", "size": 100}
                }
            }
        }
    )
    buckets = resp["aggregations"]["files"]["buckets"]
    if not buckets:
        print("No documents found in OpenSearch index.")
        return
    print(f"\n{'='*60}")
    print(f"Files in OpenSearch ({len(buckets)} total):")
    print(f"{'='*60}")
    for b in buckets:
        print(f"  {b['key']}  ({b['doc_count']} chunks)")
    print()

def inspect_chunks(client, source_key, limit):
    """Fetch and display raw chunks for a specific file."""
    resp = client.search(
        index="rag-chunks",
        body={
            "size": limit,
            "query": {
                "term": {"source_key": source_key}
            },
            "sort": [{"chunk_index": {"order": "asc"}}]
        }
    )

    hits = resp["hits"]["hits"]
    total = resp["hits"]["total"]["value"]

    if not hits:
        print(f"\nNo chunks found for: {source_key}")
        print("Check the filename matches exactly (case-sensitive).")
        return

    print(f"\n{'='*60}")
    print(f"File: {source_key}")
    print(f"Total chunks in index: {total} | Showing: {len(hits)}")
    print(f"{'='*60}")

    for i, hit in enumerate(hits):
        src = hit["_source"]
        embedding = src.get("embedding", [])
        hyq = src.get("hyq_questions", "")
        hyq_list = [q.strip() for q in hyq.split("\n") if q.strip()] if hyq else []

        print(f"\n--- Chunk {src.get('chunk_index', '?')} (doc_id: {hit['_id']}) ---")
        print(f"  source_key  : {src.get('source_key', 'MISSING')}")
        print(f"  page        : {src.get('page', 'MISSING')}")
        print(f"  section     : {src.get('section', 'MISSING')}")
        print(f"  paragraph   : {src.get('row_index', 'MISSING')}")
        print(f"  chunk_index : {src.get('chunk_index', 'MISSING')}")
        print(f"  embedding   : {len(embedding)} dimensions {'✅' if len(embedding) == 1536 else '❌ WRONG SIZE'}")
        print(f"  hyq_questions ({len(hyq_list)} questions):")
        for q in hyq_list:
            print(f"    - {q}")
        print(f"  text preview:")
        text = src.get("text", "")
        preview = text[:300].replace("\n", " ")
        print(f"    {preview}{'...' if len(text) > 300 else ''}")

    print(f"\n{'='*60}")
    print(f"Summary for {source_key}:")
    print(f"  Total chunks : {total}")

    # Check for missing fields across all returned chunks
    missing_page     = sum(1 for h in hits if h["_source"].get("page") is None)
    missing_section  = sum(1 for h in hits if not h["_source"].get("section"))
    missing_para = sum(1 for h in hits if h["_source"].get("row_index") is None)
    missing_hyq      = sum(1 for h in hits if not h["_source"].get("hyq_questions"))
    missing_embed    = sum(1 for h in hits if len(h["_source"].get("embedding", [])) != 1536)

    print(f"  page missing       : {missing_page}/{len(hits)} chunks")
    print(f"  section missing    : {missing_section}/{len(hits)} chunks")
    print(f"  paragraph missing  : {missing_para}/{len(hits)} chunks")
    print(f"  hyq missing        : {missing_hyq}/{len(hits)} chunks")
    print(f"  embedding wrong    : {missing_embed}/{len(hits)} chunks")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect raw OpenSearch chunks")
    parser.add_argument("--file", type=str, help="Source file name (e.g. ABCWordDoc.docx)")
    parser.add_argument("--limit", type=int, default=3, help="Number of chunks to display (default: 3)")
    args = parser.parse_args()

    client = get_opensearch_client()

    if not args.file:
        list_files(client)
        print("Tip: run with --file <filename> to inspect chunks for a specific file.")
    else:
        inspect_chunks(client, args.file, args.limit)


if __name__ == "__main__":
    main()