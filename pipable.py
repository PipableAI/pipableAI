import copy
import os

import jax.numpy as jnp
import pandas as pd
import yaml

from classes.google import _google_search
from classes.llm import _llm
from classes.pandas import _pandas_search
from classes.postgres import _postgres_search
from classes.reader import _data_reader
from classes.semantic import _semantic_search


class _proxy_results():
  def __init__(self):
    self.llm_thread = ""
    self.action_list=[]
    self.output_objects = []
    self.error_outputs = []
    self.human_prompt = []
    self.current_output_type = -1

  def update_llm_thread(self,thread):
    self.llm_thread = thread

  def update_action(self,model_id):
    self.action_list.append(model_id)

  def update_outputobjs(self,outputObj=None):
    self.current_output_type = 0
    self.output_objects.append(outputObj)
  
  def update_errorobjs(self,errorObj=None):
    self.current_output_type = 1
    self.error_outputs.append(errorObj)
  
  def reset_outputs(self):
    self.output_objects = []
    self.error_outputs = []
    self.current_output_type = -1
    self.llm_thread = ""
    self.action_list = []
    self.human_prompt = []
  
  def update_human_prompt(self,query):
    self.human_prompt.append(query)

class _output_obj():
  def __init__(self,output="",model_id=""):
    self._output = output
    self._model_id = model_id

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
      self.datasearch = _pandas_search(openai_key=config["keys"]["openAI"],df=output, pathlog=config["pathToData"], datatype=dataType, context=config["context"]).initialize()
    elif dataType == "postgres":
      self.datasearch = _postgres_search(openai_key=config["keys"]["openAI"],file_path=config["pathToData"], context=config["context"]).initialize()
    elif dataType == "mysql":
      print("ERROR: mysql data type not yet implemented. Valid data types are csv, parquet, PDF, and postgres.")
    elif dataType == "json":
      print("ERROR: json data type not yet implemented. Valid data types are csv, parquet, PDF, and postgres.")
    else:
      print("Error: no valid data type specified. Valid data types are csv, parquet, PDF, and postgres.")
      return None
        
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

    self.action_desc = config["action_desc"]

    #WARNING: use internal_sem_s for all semantic search functions related to action_desc
    self.internal_sem_s = copy.deepcopy(self.sem_s)
    self.internal_sem_s.create_key_vectors(list(self.action_desc.values()))
    self.results_proxy = _proxy_results()

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

    return
  
  def ask(self,query,model=""):
    self.results_proxy.update_human_prompt(query)
    # model auto selection based on query
    if model == "":
      score = int(jnp.argmax(self.internal_sem_s.find_similar_score(query_list=query)))
      model = list(self.action_desc.keys())[score]
    # specified or auto selected model is valid
    if model in self.action_desc:
      flag, result = self.key2method[model](query)
      
      if model == "llm_google":
        flag, googres = self.askgoogle.ask_google(result)
        # failure to search not implemented yet
        result += str(googres)
        self.results_proxy.update_llm_thread(result)

      self.results_proxy.update_action(model)
      if flag == 1:
        self.results_proxy.update_errorobjs(_output_obj(output=result,model_id=model))
      else:
        self.results_proxy.update_outputobjs(_output_obj(output=result,model_id=model))
    # specified model is invalid
    else:
      print(model, ": No such model found. Ensure that correct model_id is entered. Refer to .get_help() for model ids.")
    return

  def get_help(self):
    print("You can ask any question using the ask function. It takes two Parameters, query and model. Different models and their query are mentioned below:")
    print("MODEL\t\tDESCRIPTION")
    for i in self.action_desc:
      print(i,"\t\t",self.action_desc[i])
  
  def get_latest_output(self):
    if self.results_proxy.current_output_type == -1:
      print("No outputs found")
      return
    else:
      if self.results_proxy.current_output_type == 0:
        return self.results_proxy.output_objects[-1]._output
      else:
        return self.results_proxy.error_outputs[-1]._output

  def get_all_outputs(self):
    return {
      "outputs":[x._output for x in self.results_proxy.output_objects],
      "errors":[x._output for x in self.results_proxy.error_outputs]
    }
  
  def reset_chain(self):
    self.results_proxy.reset_outputs()
  
  def find_action(self, query):
    score = int(jnp.argmax(self.internal_sem_s.find_similar_score(query_list=query)))
    model = list(self.action_desc.keys())[score]
    return copy.deepcopy(self.key2method[model])

# returning 0 means success, 1 means error