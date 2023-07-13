import jax.numpy as jnp
from IPython.display import HTML
from sentence_transformers import SentenceTransformer


class _semantic_search():
  def __init__(self,embedder = None,key_vectors = None):
    super().__init__()
    self.embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") if embedder == None else embedder
    self.key_vectors=key_vectors

  def vectorize(self,data_list):
    return self.embedder.encode(data_list)

  def create_key_vectors(self,key_list):
    self.key_vectors = self.vectorize(key_list)
    return self

  def find_similar_score(self,query_list):
    key_vectors = self.key_vectors
    if key_vectors.any() == False:
      print("Key vectors not found.")
    else:
      key = jnp.asarray(key_vectors)
      query = self.vectorize(query_list)
      query = jnp.asarray(query)
      sim_score = jnp.dot(key,query.T)
    return sim_score