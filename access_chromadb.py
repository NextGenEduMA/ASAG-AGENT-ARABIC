import chromadb
from chromadb.utils import embedding_functions
import pandas as pd

def connect_to_chromadb():
    """Connect to the ChromaDB instance"""
    try:
        client = chromadb.Client()
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-mpnet-base-v2"
        )
        
        # Try to get the collection, create it if it doesn't exist
        try:
            collection = client.get_collection(
                name="arabic_educational_content",
                embedding_function=embedding_function
            )
        except Exception:
            print("Collection doesn't exist. Creating a new one...")
            collection = client.create_collection(
                name="arabic_educational_content",
                embedding_function=embedding_function
            )
            
            # Add sample data
            sample_text = "هذا نص تجريبي للتأكد من عمل قاعدة البيانات. يمكن استخدام هذا النص لاختبار وظائف البحث والاسترجاع."
            collection.add(
                documents=[sample_text],
                metadatas=[{"content_id": "sample", "chunk_id": 0}],
                ids=["sample_0"]
            )
            print("Added sample text to the collection")
            
        print("Successfully connected to ChromaDB collection")
        return collection
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        return None

def list_all_documents(collection):
    """List all documents in the collection"""
    if not collection:
        return
    
    try:
        # Get all documents (limited to 1000)
        result = collection.get(limit=1000)
        
        # Create a DataFrame for better display
        df = pd.DataFrame({
            'id': result['ids'],
            'document': result['documents'],
            'metadata': result['metadatas']
        })
        
        print(f"Total documents: {len(df)}")
        return df
    except Exception as e:
        print(f"Error listing documents: {e}")
        return None

def search_documents(collection, query, n_results=5):
    """Search for documents similar to the query"""
    if not collection:
        return
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        print(f"Found {len(results['documents'][0])} relevant documents")
        
        # Create a DataFrame for better display
        data = []
        for i in range(len(results['documents'][0])):
            data.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"Error searching documents: {e}")
        return None

def add_document(collection, text, content_id="manual_entry"):
    """Add a document to the collection"""
    if not collection:
        return
    
    try:
        # Generate a unique ID
        import uuid
        doc_id = f"{content_id}_{str(uuid.uuid4())[:8]}"
        
        # Add the document
        collection.add(
            documents=[text],
            metadatas=[{"content_id": content_id, "chunk_id": 0}],
            ids=[doc_id]
        )
        print(f"Document added successfully with ID: {doc_id}")
        return True
    except Exception as e:
        print(f"Error adding document: {e}")
        return False

def delete_document(collection, document_id):
    """Delete a document by ID"""
    if not collection:
        return
    
    try:
        collection.delete(ids=[document_id])
        print(f"Document {document_id} deleted successfully")
        return True
    except Exception as e:
        print(f"Error deleting document: {e}")
        return False

if __name__ == "__main__":
    collection = connect_to_chromadb()
    
    if not collection:
        print("Failed to connect to ChromaDB. Make sure it's properly initialized.")
        exit(1)
    
    while True:
        print("\n=== ChromaDB Access Tool ===")
        print("1. List all documents")
        print("2. Search documents")
        print("3. Add a document")
        print("4. Delete a document")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == "1":
            df = list_all_documents(collection)
            if df is not None and not df.empty:
                print("\nDocument Preview (first 5):")
                print(df.head())
                
                # Ask if user wants to see all
                if len(df) > 5 and input("Show all documents? (y/n): ").lower() == 'y':
                    pd.set_option('display.max_colwidth', 50)
                    print(df)
        
        elif choice == "2":
            query = input("Enter your search query: ")
            n_results = int(input("Number of results to return: ") or "5")
            results = search_documents(collection, query, n_results)
            
            if results is not None and not results.empty:
                pd.set_option('display.max_colwidth', 100)
                print(results)
        
        elif choice == "3":
            text = input("Enter the text to add: ")
            content_id = input("Enter content ID (or press Enter for default): ") or "manual_entry"
            if text:
                add_document(collection, text, content_id)
        
        elif choice == "4":
            doc_id = input("Enter document ID to delete: ")
            if doc_id:
                delete_document(collection, doc_id)
        
        elif choice == "5":
            print("Exiting ChromaDB Access Tool")
            break
        
        else:
            print("Invalid choice. Please try again.")