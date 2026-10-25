"""
setup_cloudfront_s3_policy.py
-----------------------------
Sets up S3 bucket for CloudFront static website hosting.
Makes bucket publicly readable via CloudFront website endpoint.

Usage:
    python setup_cloudfront_s3_policy.py
"""

import boto3
import json

PROFILE_NAME = 'rag-dev'
BUCKET_NAME  = 'rag-frontend-<ACCOUNT_ID>'

session   = boto3.Session(profile_name=PROFILE_NAME)
s3_client = session.client('s3', region_name='us-east-1')

# Step 1 - Disable Block Public Access
print(f"Step 1: Disabling Block Public Access for {BUCKET_NAME}...")
s3_client.put_public_access_block(
    Bucket=BUCKET_NAME,
    PublicAccessBlockConfiguration={
        'BlockPublicAcls':        False,
        'IgnorePublicAcls':       False,
        'BlockPublicPolicy':      False,
        'RestrictPublicBuckets':  False
    }
)
print("✅ Block Public Access disabled")

# Step 2 - Set public read bucket policy
print(f"\nStep 2: Setting public read policy for {BUCKET_NAME}...")
policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid":       "PublicRead",
            "Effect":    "Allow",
            "Principal": "*",
            "Action":    "s3:GetObject",
            "Resource":  f"arn:aws:s3:::{BUCKET_NAME}/*"
        }
    ]
}
s3_client.put_bucket_policy(
    Bucket=BUCKET_NAME,
    Policy=json.dumps(policy)
)
print("✅ Public read policy set")

print(f"""
Setup complete!
→ S3 bucket is publicly readable
→ CloudFront uses S3 website endpoint
→ HTTPS served by CloudFront

Try opening: https://<DISTRIBUTION_ID>.cloudfront.net
""")