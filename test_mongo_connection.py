from pymongo import MongoClient
from dotenv import load_dotenv
import os
import ssl
import certifi

# Charger les variables d'environnement
load_dotenv()

# Récupérer l'URI MongoDB
mongo_uri = os.getenv("MONGO_URI")

# Créer un contexte SSL personnalisé
ssl_context = ssl.create_default_context(cafile=certifi.where())

try:
    # Connexion avec contexte SSL personnalisé
    client = MongoClient(
        mongo_uri,
        tls=True,
        tlsCAFile=certifi.where()
    )
    
    # Vérifier la connexion avec une commande ping
    client.admin.command('ping')
    
    print("Connexion à MongoDB réussie!")
    
    # Afficher les bases de données disponibles
    print("Bases de données disponibles:")
    for db in client.list_database_names():
        print(f" - {db}")
        
except Exception as e:
    print(f"Erreur de connexion: {e}")