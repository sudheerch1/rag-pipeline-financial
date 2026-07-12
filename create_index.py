"""
create_index.py (Financial RAG)

FIX APPLIED: index name corrected from 'rag-chunks' to 'rag-chunks-financial'.
The previous version created the wrong index — meaning the explicit knn_vector
mapping (correct dimension, HNSW/cosine config) was NEVER applied to the
index actually used by ingestion_graph.py / query_graph.py. That index has
been relying on OpenSearch's auto-created default mapping instead, which may
not have the intended vector search configuration.

Run this once after creating a new OpenSearch domain, before ingesting or
restoring any data.
"""
import requests
import os
from requests.auth import HTTPBasicAuth
from config import get_opensearch_endpoint, get_opensearch_credentials


INDEX_NAME = os.environ.get('TARGET_INDEX_NAME', 'rag-chunks-financial')

# Get everything from AWS
endpoint = f"https://{get_opensearch_endpoint(profile_name='rag-dev')}"
username, password = get_opensearch_credentials(profile_name='rag-dev')
auth = HTTPBasicAuth(username, password)

is_live_index = (INDEX_NAME == 'rag-chunks-financial')
target_label = "⚠️  LIVE INDEX" if is_live_index else "TEST/SCRATCH INDEX"
print(f"Index: {INDEX_NAME}   [{target_label}]")

if is_live_index:
    confirm = input("No TARGET_INDEX_NAME set — this will create/modify the LIVE index. Type 'yes' to continue: ")
    if confirm.lower() != 'yes':
        print("Cancelled. Set TARGET_INDEX_NAME to a scratch index name and re-run for a test create.")
        exit(0)

print(f"Connecting to: {endpoint}")
print(f"Username: {username}")

# Index mapping with cosine similarity
index_body = {
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 1536,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",   # ← cosine similarity
                    "engine": "faiss",
                    "parameters": {
                        "ef_construction": 128,
                        "m": 24
                    }
                }
            },
            "source_key": {"type": "keyword"},
            "page": {"type": "integer"},
            "chunk_index": {"type": "integer"},
            "hyq_questions": {"type": "text"},     # ← for HyQ
            "section": {"type": "keyword"},        # ← for section attribution
            "paragraph": {"type": "integer"}       # ← for paragraph attribution
        }
    },
    "settings": {
        "index.knn": True
    }
}

# Create index
url = f"{endpoint}/{INDEX_NAME}"
response = requests.put(
    url,
    auth=auth,
    json=index_body,
    headers={"Content-Type": "application/json"}
)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")

if response.status_code in (200, 201):
    print(f"\n✅ Index '{INDEX_NAME}' created successfully with explicit mapping.")
elif response.status_code == 400 and "resource_already_exists_exception" in response.text:
    print(f"\n⚠️  Index '{INDEX_NAME}' already exists — no changes made.")
    print("   If you need to reset the mapping, delete the index first (data loss).")
else:
    print("\n❌ Unexpected response. Check the error above.")
