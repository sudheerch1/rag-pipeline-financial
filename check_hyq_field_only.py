from opensearch_client import get_opensearch_client

client = get_opensearch_client()

response = client.search(
    index="rag-chunks-financial",
    body={
        "query": {
            "bool": {
                "must": [{"match_phrase": {"text": "Total revenue"}}],
                "filter": [{"match_phrase": {"text": "245,122"}}]
            }
        },
        "_source": ["text", "hyq_questions", "source_key"],
        "size": 5
    }
)

for i, hit in enumerate(response["hits"]["hits"]):
    src = hit["_source"]
    print(f"\n--- Chunk {i+1} ---")
    print("source_key:", src.get("source_key"))
    print("hyq_questions is:", repr(src.get("hyq_questions")))
    print("hyq_questions type:", type(src.get("hyq_questions")))
