import json
import numpy as np
from sentence_transformers import SentenceTransformer
import time

print('Loading embedding model all-MiniLM-L6-v2...')
model = SentenceTransformer('all-MiniLM-L6-v2')

# 1. Embed Corpus Chunks
print('Reading data/chunk_texts.json...')
with open('data/chunk_texts.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

texts = [c['text'] for c in chunks]
print(f'Encoding {len(texts)} corpus chunks (batch_size=64)...')
t0 = time.time()
corpus_embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
t1 = time.time()
print(f'Corpus embeddings complete in {t1 - t0:.2f}s. Shape: {corpus_embeddings.shape}')
np.save('data/corpus_embeddings.npy', corpus_embeddings.astype(np.float32))

# 2. Embed Evaluation Queries
print('Reading data/test_queries.json...')
with open('data/test_queries.json', 'r', encoding='utf-8') as f:
    queries = json.load(f)

query_texts = [q['query_text'] for q in queries]
print(f'Encoding {len(query_texts)} evaluation queries...')
query_embeddings = model.encode(query_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
print(f'Query embeddings complete. Shape: {query_embeddings.shape}')
np.save('data/query_embeddings.npy', query_embeddings.astype(np.float32))

print('All embeddings generated and saved to data/!')
