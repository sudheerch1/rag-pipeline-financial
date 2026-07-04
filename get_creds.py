from config import get_opensearch_credentials
u, p = get_opensearch_credentials(profile_name="rag-dev")
print(u)
print(p)
