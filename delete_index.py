import requests
from requests.auth import HTTPBasicAuth
from config import get_opensearch_endpoint, get_opensearch_credentials

# Get everything from AWS
endpoint = f"https://{get_opensearch_endpoint(profile_name='rag-dev')}"
username, password = get_opensearch_credentials(profile_name='rag-dev')
auth = HTTPBasicAuth(username, password)

print(f"Connecting to: {endpoint}")

# Delete index
response = requests.delete(
    f"{endpoint}/rag-chunks-financial",
    auth=auth
)

print(f"Status: {response.status_code}")
print(f"Response: {response.text}")