from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

def get_db(name, embed_dim=None, metric=None):
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Only list indexes and create if we are in setup mode (dimensions provided)
    if (embed_dim is not None) and (metric is not None):
        existing_indexes = pc.list_indexes().names()
        if name not in existing_indexes:
            spec = ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            )
            pc.create_index(
                name=name,
                dimension=embed_dim,
                metric=metric,
                spec=spec,
            )
            print(f"Created Pinecone index: {name}")
        else:
            print(f"Pinecone index ready: {name}")

    # Return the index object
    index = pc.Index(name=name)
    return index