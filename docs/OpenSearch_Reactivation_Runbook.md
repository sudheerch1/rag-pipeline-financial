# OpenSearch Reactivation Runbook
## RAG Pipeline — Standalone Checklist

**Use this when:** OpenSearch domain was deleted to save costs and needs to be recreated.

---

## ⚠️ Order Is Strict — Do Not Skip Steps

| Step | Action | Time |
|---|---|---|
| 1 | Create OpenSearch cluster in AWS console | ~15 min wait |
| 2 | Update endpoint in Parameter Store | 1 min |
| 3 | `python create_index.py` | 30 sec |
| 4 | `python setup_hybrid_search.py` | 30 sec |
| 5 | Map IAM backend role in Dashboards UI | 2 min |
| 6 | Reset document hashes in DynamoDB | 1 min |
| 7 | Upload PDFs to S3 (one at a time) | 5–10 min each |
| 8 | Verify chunk counts in DynamoDB | 1 min |
| 9 | Run test query + confirm LangSmith trace | 2 min |

**Total time: ~40–50 minutes**

---

## Step 1 — Create OpenSearch Cluster

AWS Console → Amazon OpenSearch Service → Create domain

| Setting | Value |
|---|---|
| Domain name | `rag-vectors` |
| Deployment type | Development and testing |
| Engine version | OpenSearch 2.11 |
| Instance type | `t3.small.search` |
| Number of nodes | 1 |
| Storage | EBS, gp3, 20 GB |
| Fine-grained access control | Enabled |
| Master user | `admin` / (your password) |
| Domain access policy | Only use fine-grained access control |
| Encryption at rest | Enabled |
| Node-to-node encryption | Enabled |
| Require HTTPS | Yes |

Wait for status to show **Active** (~15 minutes).

---

## Step 2 — Update Endpoint in Parameter Store

Copy the domain endpoint from the OpenSearch console (format: `search-rag-vectors-xxxx.us-east-1.es.amazonaws.com`).

```powershell
aws ssm put-parameter `
  --name "/rag/opensearch/endpoint" `
  --value "search-rag-vectors-XXXX.us-east-1.es.amazonaws.com" `
  --type String --overwrite `
  --profile rag-dev --region us-east-1
```

---

## Step 3 — Create Index

```powershell
python create_index.py
```

**What this does:** Creates the `rag-chunks` index with `knn_vector` mapping (1536-dim, HNSW/cosine), BM25 text fields, and `hyq_questions` field.

**Must run BEFORE uploading any PDF.** If you upload first, you'll get a `knn_vector` mapping conflict error.

---

## Step 4 — Create Hybrid Search Pipeline

```powershell
python setup_hybrid_search.py
```

**What this does:** Creates the `hybrid-search-pipeline` with BM25 + KNN combination, 0.5/0.5 weights, min-max score normalization.

---

## Step 5 — Map IAM Backend Role in Dashboards

Open: `https://<your-opensearch-endpoint>/_dashboards`

Log in with: `admin` / (your password)

Navigate:
```
Security (left nav) → Roles → all_access → Mapped users tab → Backend roles → Add
```

Add this ARN:
```
arn:aws:iam::<ACCOUNT_ID>:role/service-role/rag-document-processor-role-ucg8h778
```

> ⚠️ This is the **IAM Role ARN**, NOT the Lambda function ARN. Both Lambdas share this role.

Click **Map** and save.

---

## Step 6 — Reset Document Hashes

DynamoDB still has the content hashes from the previous ingestion. Without resetting, the ingestion Lambda will detect them as duplicates and skip.

```powershell
python reset_document.py --file johnson_v._superior_court_1.pdf
python reset_document.py --file raceway_ford_cases.pdf
# If amiodarone case was also indexed:
# python reset_document.py --file <amiodarone_filename>.pdf
```

To list all documents currently in DynamoDB:
```powershell
python reset_document.py --list
```

---

## Step 7 — Upload PDFs to S3

Upload **one at a time**. Wait for each ingestion to complete before uploading the next (avoids Bedrock Titan throttling).

```powershell
aws s3 cp "C:\Tech\PROD-Grade-POC\Dataset\johnson_v._superior_court_1.pdf" `
  s3://rag-raw-<ACCOUNT_ID>/ --profile rag-dev

# Watch logs in a separate terminal:
aws logs tail /aws/lambda/rag-document-processor --follow --profile rag-dev
# Wait for: "Status: completed" before uploading next file

aws s3 cp "C:\Tech\PROD-Grade-POC\Dataset\raceway_ford_cases.pdf" `
  s3://rag-raw-<ACCOUNT_ID>/ --profile rag-dev
```

---

## Step 8 — Verify Chunk Counts in DynamoDB

```powershell
aws dynamodb scan `
  --table-name rag-metadata `
  --profile rag-dev --region us-east-1 `
  --query "Items[*].{id:document_id.S, chunks:chunks.N, status:status.S}"
```

Expected:
| Document | Chunks |
|---|---|
| `johnson_v._superior_court_1.pdf` | 77 |
| `raceway_ford_cases.pdf` | 60 |

---

## Step 9 — Smoke Test

```powershell
# Health check (should return "healthy" + "opensearch": "reachable")
(Invoke-WebRequest `
  -Uri "https://<API_ID>.execute-api.us-east-1.amazonaws.com/prod/health" `
  -UseBasicParsing).Content

# Query test
Invoke-WebRequest `
  -Uri "https://<API_ID>.execute-api.us-east-1.amazonaws.com/prod/query" `
  -Method POST `
  -Headers @{
    "x-api-key"="gPEWpbTVLq9NPD0YJYD4pCszNr7UcF4ln1etxZd0"
    "Content-Type"="application/json"
  } `
  -Body '{"question":"What was Gregory Johnson convicted of?"}' `
  -UseBasicParsing
```

Then open LangSmith → rag-pipeline project → confirm trace appeared with all 5 steps (run_hyde, run_search, run_rerank, run_dedup, run_answer).

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `knn_vector mapping conflict` | PDF uploaded before `create_index.py` | Delete index, re-run `create_index.py`, re-upload PDFs |
| `403 Forbidden` from OpenSearch | IAM role not mapped in Dashboards | Re-do Step 5 |
| `Duplicate detected` — ingestion skips file | DynamoDB hash not reset | Re-do Step 6 |
| `Service Unavailable` on health check | OpenSearch still initializing | Wait 2–3 more minutes |
| Lambda times out on large PDF | Bedrock Titan throttling | Upload one PDF at a time, wait between |
| `OTEL_TRACES_EXPORTER` warning in logs | Expected warning, not an error | Traces still work; ignore |
