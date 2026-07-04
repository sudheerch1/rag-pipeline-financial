"""
test_query_lambda.py
--------------------
Invokes the rag-query Lambda directly (bypassing API Gateway)
to see raw errors and responses.

Usage:
    python test_query_lambda.py
"""

import boto3
import json

PROFILE_NAME   = 'rag-dev'
FUNCTION_NAME  = 'rag-query-financial'
TEST_QUESTION  = 'What was Apple total net sales for fiscal year 2023?'

session  = boto3.Session(profile_name=PROFILE_NAME)
client   = session.client('lambda', region_name='us-east-1')

payload = {
    "httpMethod": "POST",
    "body": json.dumps({"question": TEST_QUESTION, "company": "apple"}),
    "headers": {"x-api-key": "gPEWpbTVLq9NPD0YJYD4pCszNr7UcF4ln1etxZd0"}
}

print(f"Invoking {FUNCTION_NAME}...")
print(f"Question: {TEST_QUESTION}\n")

response = client.invoke(
    FunctionName=FUNCTION_NAME,
    InvocationType='RequestResponse',
    Payload=json.dumps(payload)
)

# Check for function errors
if response.get('FunctionError'):
    print(f"❌ Function error: {response['FunctionError']}")

# Parse response
result = json.loads(response['Payload'].read())
print(f"Status Code: {result.get('statusCode')}")

if 'errorMessage' in result:
    print(f"❌ Error: {result['errorMessage']}")
    print(f"Stack trace:\n{chr(10).join(result.get('stackTrace', []))}")
else:
    body = json.loads(result.get('body', '{}'))
    print(f"✅ Answer: {body.get('answer', 'No answer')[:200]}...")
    print(f"\nSources:")
    for s in body.get('sources', []):
        print(f"  - {s.get('source')} p.{s.get('page')} | section: '{s.get('section', 'N/A')}' | score: {s.get('score', 0):.3f}")