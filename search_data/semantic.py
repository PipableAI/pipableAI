from sentence_transformers import SentenceTransformer
from pprint import pprint
from IPython.display import HTML
import jax.numpy as jnp

class semantic_search():
  def __init__(self,embedder = None,key_vectors="None"):
    super().__init__()
    self.embedder = embedder
    self.key_vectors=key_vectors

  def initialize(self):
    emb_m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    self.embedder = emb_m
    return self

  def vectorize(self,data_list):
    emb = self.embedder
    #pprint(display(HTML('<span style="color:#DB4437">pipable : encoding data into a vector space that I can understand </span>')))
    embeddings = emb.encode(data_list)
    #pprint(display(HTML('<span style="color:#0F9D58">pipable: Vectors created </span>')))
    return embeddings

  def create_key_vectors(self,key_list):
    self.key_vectors = self.vectorize(key_list)
    return self

  def find_similar_score(self,query_list):
    key_vectors = self.key_vectors
    if key_vectors.any() == False:
      pprint(display(HTML('<span style="color:#0F9D58">pipable: Please update the state with by a list of documents to search against by pipable.table = table method </span>')))
    else:
      key = jnp.asarray(key_vectors)
      query = self.vectorize(query_list)
      #query = jnp.asarray(self.vectorize(query_list))
      query = jnp.asarray(query)
      sim_score = jnp.dot(key,query.T)
      #pprint(display(HTML('<span style="color:#DB4437">pipable: Here are the similarity scores in scale of 0-1: </span>')))
    return sim_score