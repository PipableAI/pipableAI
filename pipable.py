import copy
import os
import logging

import jax.numpy as jnp
import pandas as pd
import yaml

from classes.google import _google_search
from classes.llm import _llm
from classes.pandas import _pandas_search
from classes.postgres import _postgres_search
from classes.reader import _data_reader
from classes.semantic import _semantic_search
from pipable_utils import PIPABLE_LOGGER_CREATE
    
PIPABLE_LOG = PIPABLE_LOGGER_CREATE("PIP_MAIN")

class Pipable():
  def __init__(self, path = ""):
    super().__init__()
    
    with open(path) as f:
      config = yaml.safe_load(f)

    self.reader = _data_reader()
    self.llm_ = _llm(openaiAPIKEY=config["keys"]["openAI"])
    self.sem_s = _semantic_search()
    self.askgoogle = _google_search().initialise(
      google_api_key=config["keys"]["google"],
      search_engine_key=config["keys"]["search_engine"]
    )

    self.action_desc = config["action_desc"]
    #WARNING: use action_sem_search for all semantic search functions related to action_desc ONLY
    self.action_sem_search = copy.deepcopy(self.sem_s)
    self.action_sem_search.create_key_vectors(list(self.action_desc.values()))

    self.context = config["context"]
    #WARNING: use context_sem_search for all semantic search functions related to context ONLY
    self.context_sem_search = copy.deepcopy(self.sem_s)
    self.context_sem_search.create_key_vectors(list(self.context.values()))

    dataType = config["dataType"]
    if dataType == "csv" or dataType == "parquet" or dataType == "pdf":
      output = None
      if dataType == "csv":
        flag, output = self.reader.read_csv(config["pathToData"])
        if flag == 1:
          print(output)
          return
      elif dataType == "parquet":
        flag, output = self.reader.read_parquet(config["pathToData"])
        if flag == 1:
          print(output)
          return
      elif dataType == "pdf":
        flag, output = self.reader.read_pdf(config["pathToData"])
        if flag == 1:
          print(output)
          return
      self.datasearch = _pandas_search(openai_key=config["keys"]["openAI"],df=output, pathlog=config["pathToData"], datatype=dataType).initialize()
    elif dataType == "postgres":
      self.datasearch = _postgres_search(openai_key=config["keys"]["openAI"],file_path=config["pathToData"]).initialize()
    else:
      print("ERROR: no valid data type specified. Valid data types are csv, parquet, PDF, and postgres.")
      return
        
    self.key2method = {
      "llm":self.llm_.ask_llm,
      "llm_google":self.llm_.ask_llm,
      "find_similar_score":self.sem_s.find_similar_score,
      "create_key_vectors":self.sem_s.create_key_vectors,
      "vectorize":self.sem_s.vectorize,
      "google_search":self.askgoogle.ask_google,
      "data_search":self.datasearch.search_data,
      "read_csv":self.reader.read_csv,
      "read_parquet":self.reader.read_parquet,
      "read_pdf":self.reader.read_pdf
    }
    #WARNING: exposed read functions do not override all configurations related to initialization
    if not os.path.exists("logs.parquet"):
      temp = pd.DataFrame({
        "timestamp": [],
        "query": [],
        "datatype": [],
        "database": [],
        "pipableAI": [],
        "openAI": [],
        "success": [],
        "error": []
      })
      temp.to_parquet("logs.parquet", engine = 'pyarrow')
    PIPABLE_LOG.info(f"Pipable class instantiated from {path}")
  
  def ask(self,query,model=""):
    # model auto selection based on query
    if model == "":
      score = int(jnp.argmax(self.action_sem_search.find_similar_score(query_list=query)))
      model = list(self.action_desc.keys())[score]
      PIPABLE_LOG.info(f"Sem search resolved query to expert model {model} (score of {score})")
    # specified or auto selected model is valid
    if model in self.action_desc:
      if model == "data_search":
        contextscore = int(jnp.argmax(self.context_sem_search.find_similar_score(query_list=query)))
        smartcontext = list(self.context.values())[contextscore]
        PIPABLE_LOG.debug(f"data_search: q:{query}, smartcontext:{smartcontext[:20]}....")
        flag, result = self.key2method[model](query, smartcontext)
      else:
        flag, result = self.key2method[model](query)

      datatype = ""
      try:
        datatype = result.dtype
      except:
        datatype = type(result)
      if flag == 1:
        datatype = "ERROR"

      if model == "llm_google":
        googflag, googres = self.askgoogle.ask_google(result)
        googdtype = type(googres)
        if googflag == 1:
          googdtype = "ERROR"

        return ({
          "isError": bool(flag),
          "output": result,
          "model_id": "llm",
          "dtype": datatype
        }, {
          "isError": bool(googflag),
          "output": googres,
          "model_id": "google_search",
          "dtype": googdtype
        })
      return ({
        "isError": bool(flag),
        "output": result,
        "model_id": model,
        "dtype": datatype
      })
    # specified model is invalid
    else:
      print(model, ": No such model found. Ensure that correct model_id is entered. Refer to .get_help() for model ids.")
      return ({
        "isError": True,
        "output": "No expert found",
        "model_id": "",
        "dtype": "ERROR"
      })

  def get_help(self):
    print("You can ask any question using the ask function. It takes two Parameters, query and model. Different models and their query are mentioned below:")
    print("MODEL\t\tDESCRIPTION")
    for i in self.action_desc:
      print(i,"\t\t",self.action_desc[i])
  
  def find_action(self, query):
    score = int(jnp.argmax(self.action_sem_search.find_similar_score(query_list=query)))
    model = list(self.action_desc.keys())[score]
    return copy.deepcopy(self.key2method[model])
    
  def reset_llm(self):
    self.llm_.reset_thread()