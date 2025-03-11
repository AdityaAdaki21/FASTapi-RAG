import chromadb

client = chromadb.Client()
collection = client.create_collection("test_collection")
print("Successfully created collection")