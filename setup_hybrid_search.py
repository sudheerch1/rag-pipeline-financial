"""
setup_hybrid_search.py
----------------------
Creates the OpenSearch hybrid search pipeline that combines:
  - KNN vector search (semantic similarity)
  - BM25 text search (keyword matching)

This script is idempotent - safe to run multiple times.
Run this once after creating a new OpenSearch index.

Usage:
    python setup_hybrid_search.py
"""

import requests
from requests.auth import HTTPBasicAuth
from config import get_opensearch_endpoint, get_opensearch_credentials

# ── Connect to OpenSearch ─────────────────────────────────────────────────────

endpoint = f"https://{get_opensearch_endpoint(profile_name='rag-dev')}"
username, password = get_opensearch_credentials(profile_name='rag-dev')
auth = HTTPBasicAuth(username, password)

print(f"Connecting to: {endpoint}")

# ── Create Hybrid Search Pipeline ────────────────────────────────────────────
#
# This pipeline tells OpenSearch how to combine vector and BM25 scores:
#   1. normalization: min_max → normalize both scores to 0-1 range
#   2. combination: arithmetic_mean → average the two scores
#   3. weights: [0.5, 0.5] → equal weight to vector and BM25
#
# Weight tuning guide:
#   [0.7, 0.3] → favor semantic search (better for conceptual questions)
#   [0.3, 0.7] → favor keyword search (better for exact term matching)
#   [0.5, 0.5] → balanced (good default for legal/technical documents)

pipeline_body = {
    "description": "Hybrid search pipeline combining KNN vector search and BM25",
    "phase_results_processors": [
        {
            "normalization-processor": {
                "normalization": {
                    "technique": "min_max"
                },
                "combination": {
                    "technique": "arithmetic_mean",
                    "parameters": {
                        "weights": [0.5, 0.5]
                    }
                }
            }
        }
    ]
}

pipeline_url = f"{endpoint}/_search/pipeline/hybrid-search-pipeline"

response = requests.put(
    pipeline_url,
    auth=auth,
    json=pipeline_body,
    headers={"Content-Type": "application/json"}
)

print(f"Status: {response.status_code}")
print(f"Response: {response.text}")

if response.status_code == 200:
    print("\n✅ Hybrid search pipeline created successfully!")
    print("   Pipeline name: hybrid-search-pipeline")
    print("   Vector weight: 0.5")
    print("   BM25 weight:   0.5")
    print("\nNext step: update query_handler.py to use hybrid search")
else:
    print("\n❌ Failed to create pipeline. Check the error above.")