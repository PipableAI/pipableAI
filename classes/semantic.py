import jax.numpy as jnp
from IPython.display import HTML
from sentence_transformers import SentenceTransformer


class _semantic_search():
  def __init__(self,embedder = None):
    super().__init__()
    self.embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") if embedder == None else embedder

  def vectorize(self,data_list):
    return self.embedder.encode(data_list)

  def create_key_vectors(self,key_list):
    self.key_vectors = self.vectorize(key_list)
    return self

  def find_similar_score(self,query_list):
    key = jnp.asarray(self.key_vectors)
    query = self.vectorize(query_list)
    query = jnp.asarray(query)
    sim_score = jnp.dot(key,query.T)
    return sim_score

# returning 0 means success, 1 means error
# tuple error mechanism not implemented yet