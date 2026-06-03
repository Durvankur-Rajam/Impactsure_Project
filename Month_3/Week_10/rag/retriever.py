def retrieve(query, collection, embedder, top_k=3):
    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings = query_embedding,
        n_results        = top_k
    )

    docs    = results['documents'][0]
    metas   = results['metadatas'][0]
    sources = [m['source'] for m in metas]

    return docs, sources


def build_context(docs, max_chars=1500):
    context = "\n\n".join(docs)
    return context[:max_chars]