import boto3
import requests
from requests_aws4auth import AWS4Auth

region = 'us-east-1'
service = 'es'
host = 'https://search-rag-vectors-xzipwfal5kzob35v4x7uzpnz7i.us-east-1.es.amazonaws.com'

credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    service,
    session_token=credentials.token
)

url = f"{host}/_cat/indices?v&h=index,docs.count,store.size,pri.store.size"
response = requests.get(url, auth=awsauth)
print(response.status_code)
print(response.text)