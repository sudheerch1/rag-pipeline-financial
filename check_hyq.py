"""
check_hyq.py

Standalone diagnostic script — queries OpenSearch directly for the
"Total net sales" chunk(s) and prints their hyq_questions field.

Run from the rag-pipeline-financial repo root (same directory as
opensearch_client.py), inside the .venv, so the import resolves:

    python check_hyq.py

Purpose: confirm whether HyQ-generated questions for table-row chunks
are weak/generic (root cause) vs. well-formed (rules out HyQ as the issue).
"""

from opensearch_client import get_opensearch_client

client = get_opensearch_client()

response = client.search(
    index='rag-chunks-financial',
    body={
        "query": {
            "match_phrase": {
                "text": "Total net sales $ 391,035"
            }
        },
        "_source": ["text", "hyq_questions", "source_key", "page", "section"]
    }
)

hits = response['hits']['hits']
print(f"\n{len(hits)} matching chunk(s) found.\n")

for hit in hits:
    src = hit['_source']
    print("=" * 80)
    print(f"source_key: {src.get('source_key')}")
    print(f"page:       {src.get('page')}")
    print(f"section:    {src.get('section')}")
    print(f"\nTEXT:\n{src.get('text')}")
    print(f"\nHYQ_QUESTIONS:\n{src.get('hyq_questions')}")
    print()