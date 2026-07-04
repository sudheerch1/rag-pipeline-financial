"""
delete_source_chunks.py

Deletes all chunks for a given source_key from rag-chunks-financial.
Use before forcing a re-ingestion of a document whose content hash
hasn't changed but whose processing code (parser/chunker) has.

Usage:
    python delete_source_chunks.py apple-10k-fy2024.html
"""

import sys
from opensearch_client import get_opensearch_client

if len(sys.argv) != 2:
    print("Usage: python delete_source_chunks.py <source_key>")
    sys.exit(1)

source_key = sys.argv[1]
client = get_opensearch_client()

response = client.delete_by_query(
    index='rag-chunks-financial',
    body={"query": {"term": {"source_key": source_key}}}
)

deleted = response.get('deleted', 0)
print(f"Deleted {deleted} chunk(s) for source_key='{source_key}' from rag-chunks-financial.")