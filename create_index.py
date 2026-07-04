import requests
from requests.auth import HTTPBasicAuth
from config import get_opensearch_endpoint, get_opensearch_credentials

# Get everything from AWS
endpoint = f"https://{get_opensearch_endpoint(profile_name='rag-dev')}"
username, password = get_opensearch_credentials(profile_name='rag-dev')
auth = HTTPBasicAuth(username, password)

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
            "hyq_questions": {"type": "text"},     # ← for HyQ (next step)
            "section": {"type": "keyword"},        # ← for section attribution
            "paragraph": {"type": "integer"}       # ← for paragraph attribution
        }
    },
    "settings": {
        "index.knn": True
    }
}

# Create index
url = f"{endpoint}/rag-chunks-financial"
response = requests.put(
    url,
    auth=auth,
    json=index_body,
    headers={"Content-Type": "application/json"}
)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
