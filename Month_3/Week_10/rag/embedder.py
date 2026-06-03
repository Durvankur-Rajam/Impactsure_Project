import chromadb
from sentence_transformers import SentenceTransformer


EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "hallucicheck"


def get_embedder():
    print(f"Loading embedder: {EMBED_MODEL}")
    return SentenceTransformer(EMBED_MODEL)


def get_collection(client=None):
    if client is None:
        client = chromadb.Client()
    try:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"Loaded existing collection: {COLLECTION_NAME}")
    except Exception:
        collection = client.create_collection(COLLECTION_NAME)
        print(f"Created new collection: {COLLECTION_NAME}")
    return collection, client


def embed_and_store(chunks, source_name, embedder=None, collection=None, client=None):
    if embedder is None:
        embedder = get_embedder()
    if collection is None:
        collection, client = get_collection(client)

    batch_size = 100
    total_stored = 0

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_ids    = [f"{source_name}_{i + j}" for j in range(len(batch_chunks))]
        batch_metas  = [{"source": source_name, "chunk_index": i + j} for j in range(len(batch_chunks))]
        embeddings   = embedder.encode(batch_chunks).tolist()

        collection.add(
            documents  = batch_chunks,
            embeddings = embeddings,
            ids        = batch_ids,
            metadatas  = batch_metas
        )
        total_stored += len(batch_chunks)

    print(f"Stored {total_stored} chunks from '{source_name}' in ChromaDB")
    return collection, client