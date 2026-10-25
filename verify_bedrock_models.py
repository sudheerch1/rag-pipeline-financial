"""
verify_bedrock_models.py
------------------------
Verifies that your AWS account can successfully invoke:
  1. Cohere Embed v4   (cohere.embed-v4:0)
  2. Claude Haiku 4.5  (anthropic.claude-haiku-4-5-20251001-v1:0)

Usage:
    python verify_bedrock_models.py

Requirements:
    pip install boto3
    AWS profile 'rag-dev' must be configured
"""

import boto3
import json


AWS_REGION  = "us-east-1"
AWS_PROFILE = "rag-dev"

COHERE_MODEL_ID = "cohere.embed-v4:0"
CLAUDE_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def get_client():
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    return session.client("bedrock-runtime")


def test_cohere_embed(client):
    print("\n" + "="*50)
    print("TEST 1: Cohere Embed v4")
    print(f"Model ID: {COHERE_MODEL_ID}")
    print("="*50)

    request_body = {
        "texts": ["The plaintiff alleged breach of contract in Superior Court."],
        "input_type": "search_document",
        "embedding_types": ["float"]
    }

    print("Sending a sample legal sentence for embedding...")

    try:
        response = client.invoke_model(
            modelId=COHERE_MODEL_ID,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )

        result = json.loads(response["body"].read())
        embeddings = result["embeddings"]["float"]
        vector = embeddings[0]

        print(f"  Status      : SUCCESS ✅")
        print(f"  Dimensions  : {len(vector)}")
        print(f"  First 5 vals: {[round(v, 4) for v in vector[:5]]}")
        return True

    except client.exceptions.AccessDeniedException:
        print("  Status : FAILED ❌  — AccessDeniedException")
        print("  Action : Go to Bedrock console → Model catalog → find Cohere Embed v4 → Enable")
        return False
    except Exception as e:
        print(f"  Status : FAILED ❌")
        print(f"  Error  : {e}")
        return False


def test_claude_haiku(client):
    print("\n" + "="*50)
    print("TEST 2: Claude Haiku 4.5")
    print(f"Model ID: {CLAUDE_MODEL_ID}")
    print("="*50)

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 50,
        "messages": [
            {
                "role": "user",
                "content": "In one sentence, what is a legal holding?"
            }
        ]
    }

    print("Sending a simple legal question...")

    try:
        response = client.invoke_model(
            modelId=CLAUDE_MODEL_ID,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )

        result = json.loads(response["body"].read())
        answer = result["content"][0]["text"]
        input_tokens  = result["usage"]["input_tokens"]
        output_tokens = result["usage"]["output_tokens"]

        print(f"  Status        : SUCCESS ✅")
        print(f"  Response      : {answer}")
        print(f"  Tokens used   : {input_tokens} input / {output_tokens} output")
        return True

    except client.exceptions.AccessDeniedException:
        print("  Status : FAILED ❌  — AccessDeniedException")
        print("  Action : Go to Bedrock console → Model catalog → find Claude Haiku 4.5 → Enable")
        return False
    except Exception as e:
        print(f"  Status : FAILED ❌")
        print(f"  Error  : {e}")

        # Suggest inference profile fallback if ValidationException
        if "ValidationException" in str(type(e)):
            print(f"\n  Hint: Try the cross-region inference profile ID instead:")
            print(f"        us.anthropic.claude-haiku-4-5-20251001-v1:0")
        return False


def main():
    print("\nBedrock Model Access Verification")
    print(f"Account : <ACCOUNT_ID>")
    print(f"Region  : {AWS_REGION}")
    print(f"Profile : {AWS_PROFILE}")

    try:
        client = get_client()
    except Exception as e:
        print(f"\nFailed to create Bedrock client: {e}")
        print("Make sure your 'rag-dev' AWS profile is configured correctly.")
        return

    cohere_ok = test_cohere_embed(client)
    claude_ok  = test_claude_haiku(client)

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"  Cohere Embed v4  : {'✅ Ready' if cohere_ok else '❌ Not accessible'}")
    print(f"  Claude Haiku 4.5 : {'✅ Ready' if claude_ok else '❌ Not accessible'}")

    if cohere_ok and claude_ok:
        print("\n  Both models verified. You can proceed with code changes.")
    else:
        print("\n  Fix the failing models before proceeding.")


if __name__ == "__main__":
    main()