# delete_opensearch.py
import boto3

def reset_all_dynamodb():
    session = boto3.Session(profile_name='rag-dev')
    table = session.resource('dynamodb').Table('rag-metadata-financial')
    response = table.scan(ProjectionExpression='document_id')
    items = response.get('Items', [])
    for item in items:
        table.delete_item(Key={'document_id': item['document_id']})
        print(f"  Reset: {item['document_id']}")
    print(f"✅ Reset {len(items)} DynamoDB records")

def delete_opensearch():
    session = boto3.Session(profile_name='rag-dev')
    client = session.client('opensearch', region_name='us-east-1')
    client.delete_domain(DomainName='rag-vectors')
    print("✅ OpenSearch domain deletion initiated")
    print("   Domain will be fully deleted in ~10 minutes")

if __name__ == '__main__':
    confirm = input("This will delete OpenSearch AND reset all DynamoDB records. Type 'yes' to confirm: ")
    if confirm.lower() != 'yes':
        print("Cancelled.")
        exit()
    
    print("\nResetting DynamoDB records...")
    reset_all_dynamodb()
    
    print("\nDeleting OpenSearch domain...")
    delete_opensearch()
    
    print("\n✅ Done. DynamoDB and OpenSearch are now in sync.")
    print("   Next time you recreate OpenSearch, all files will reprocess cleanly.")