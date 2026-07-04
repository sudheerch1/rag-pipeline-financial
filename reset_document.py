"""
reset_document.py
-----------------
Deletes a document's metadata from DynamoDB so it can be reprocessed.

Use this when:
  - OpenSearch index was recreated (old embeddings gone, need fresh ones)
  - Document content was updated and you want to force reprocessing
  - Testing ingestion pipeline changes

Usage:
    python reset_document.py                                    # interactive prompt to pick document
    python reset_document.py --file my_document.pdf            # resets specific doc
    python reset_document.py --list                            # lists all documents
    python reset_document.py --all                             # resets ALL documents

Examples:
    python reset_document.py --file johnson_v._superior_court_1.pdf
    python reset_document.py --file raceway_ford_cases.pdf
    python reset_document.py --list
    python reset_document.py --all
"""

import sys
import boto3

PROFILE_NAME   = 'rag-dev'
METADATA_TABLE = 'rag-metadata-financial'


def get_table():
    return boto3.Session(profile_name=PROFILE_NAME).resource('dynamodb').Table(METADATA_TABLE)


def list_documents():
    """List all documents currently in DynamoDB metadata table."""
    table = get_table()
    response = table.scan(ProjectionExpression='document_id, #s, #ts',
                          ExpressionAttributeNames={'#s': 'status', '#ts': 'timestamp'})
    items = response.get('Items', [])
    if not items:
        print("No documents found in metadata table.")
        return []
    print(f"\nDocuments in {METADATA_TABLE}:")
    print("-" * 60)
    for item in items:
        print(f"  {item['document_id']}")
        print(f"    status:    {item.get('status', 'unknown')}")
        print(f"    timestamp: {item.get('timestamp', 'unknown')}")
    print(f"\nTotal: {len(items)} document(s)")
    return items


def reset_document(document_id: str):
    """Delete document metadata so it can be reprocessed."""
    table = get_table()
    table.delete_item(Key={'document_id': document_id})
    print(f"✅ Reset: '{document_id}' deleted from DynamoDB")
    print(f"   Next upload of this file will trigger full reprocessing")


def reset_all_documents():
    """Delete ALL document metadata — use with caution!"""
    table = get_table()
    response = table.scan(ProjectionExpression='document_id')
    items = response.get('Items', [])

    if not items:
        print("No documents to reset.")
        return

    confirm = input(f"⚠️  This will reset {len(items)} documents. Type 'yes' to confirm: ")
    if confirm.lower() != 'yes':
        print("Cancelled.")
        return

    for item in items:
        table.delete_item(Key={'document_id': item['document_id']})
        print(f"  Deleted: {item['document_id']}")

    print(f"\n✅ Reset {len(items)} documents from DynamoDB")


def interactive_reset():
    """Show list of documents and let user pick which one to reset."""
    table = get_table()
    response = table.scan(ProjectionExpression='document_id, #s, #ts',
                          ExpressionAttributeNames={'#s': 'status', '#ts': 'timestamp'})
    items = response.get('Items', [])

    if not items:
        print("No documents found in metadata table.")
        return

    print(f"\nDocuments in {METADATA_TABLE}:")
    print("-" * 60)
    for i, item in enumerate(items, 1):
        print(f"  [{i}] {item['document_id']}")
        print(f"      status:    {item.get('status', 'unknown')}")
        print(f"      timestamp: {item.get('timestamp', 'unknown')}")

    print(f"\nTotal: {len(items)} document(s)")
    print("\nOptions:")
    print("  Enter a number to reset that document")
    print("  Enter 'all' to reset all documents")
    print("  Enter 'q' to quit")

    choice = input("\nYour choice: ").strip().lower()

    if choice == 'q':
        print("Cancelled.")
        return

    if choice == 'all':
        reset_all_documents()
        return

    try:
        index = int(choice) - 1
        if 0 <= index < len(items):
            document_id = items[index]['document_id']
            confirm = input(f"Reset '{document_id}'? (yes/no): ").strip().lower()
            if confirm == 'yes':
                reset_document(document_id)
            else:
                print("Cancelled.")
        else:
            print(f"Invalid choice. Enter a number between 1 and {len(items)}.")
    except ValueError:
        print("Invalid input. Enter a number, 'all', or 'q'.")


if __name__ == '__main__':
    args = sys.argv[1:]

    if '--list' in args:
        list_documents()
    elif '--all' in args:
        reset_all_documents()
    elif '--file' in args:
        file_index = args.index('--file')
        if file_index + 1 < len(args):
            reset_document(args[file_index + 1])
        else:
            print("❌ Error: --file requires a filename argument")
            print("   Example: python reset_document.py --file raceway_ford_cases.pdf")
    elif args:
        # backward compatible — still works if someone passes filename directly
        reset_document(args[0])
    else:
        # no args — show interactive prompt instead of hardcoded default
        interactive_reset()