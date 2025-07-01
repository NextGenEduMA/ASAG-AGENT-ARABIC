import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv

load_dotenv()

print("--- ChromaDB Setup ---")
print("This script will download the default embedding model if it doesn't exist.")
print("This may take a few minutes depending on your internet connection...")

# This path is where ChromaDB will store the downloaded model.
cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'chroma')
print(f"Checking for model in: {cache_dir}")

try:
    # This line automatically triggers the download if the model is not found in the cache.
    # By running it here, we ensure the download happens outside of a live web request.
    ef = embedding_functions.DefaultEmbeddingFunction()
    
    # We can also initialize a client to make sure all directories are created.
    client = chromadb.Client()

    print("\n✅ Setup complete. The embedding model is now cached locally.")
    print("You can now run your Flask app ('python app.py') without download timeouts.")

except Exception as e:
    print(f"\n❌ An error occurred during setup: {e}")
    print("Please check your internet connection and try running 'python setup.py' again.")