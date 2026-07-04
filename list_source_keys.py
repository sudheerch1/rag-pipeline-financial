"""
list_source_keys.py

Lists all distinct source_key values currently indexed in rag-chunks-financial,
along with a chunk count for each. Use this to see what's actually been
ingested vs. what the DynamoDB metadata table claims.

Usage:
    python list_source_keys.py
"""

from opensearch_client import get_opensearch_client

client = get_opensearch_client()


def try_agg(field_name):
    return client.search(
        index='rag-chunks-financial',
        body={
            "size": 0,
            "aggs": {
                "sources": {
                    "terms": {
                        "field": field_name,
                        "size": 100
                    }
                }
            }
        }
    )


# rag-chunks-financial was auto-created (dynamic mapping), so source_key
# may be mapped as "text" (not directly aggregatable) with an automatic
# "source_key.keyword" sub-field instead. Try both.
try:
    response = try_agg("source_key")
except Exception:
    response = try_agg("source_key.keyword")

buckets = response['aggregations']['sources']['buckets']

print(f"\n{len(buckets)} distinct source_key value(s) found in rag-chunks-financial:\n")
for b in buckets:
    print(f"  {b['key']:<40} {b['doc_count']} chunk(s)")
print()
