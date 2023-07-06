from search_data.aggregated_stats import _aggregated_stats
from search_data.csv import _csv_search
from search_data.postgres import _postgres_search
from search_data.semantic import _semantic_search
from search_engine.ada import _ada
from search_engine.google import _google_search

import jax.numpy as jnp
import json
import pandas as pd

class _proxy_results():
  def __init__(self):
    self.ada_thread = ""
    self.action_list=[]
    self.output_objects = []
    self.error_outputs = []
    self.current_output_type = -1

  def update_ada_thread(self,thread):
    self.ada_thread = thread

  def update_action(self,model_id):
    self.action_list.append(model_id)

  def update_outputobjs(self,outputObj=None):
    self.current_output_type = 0
    self.output_objects.append(outputObj)
  
  def update_error_outputobjs(self,outputObj=None):
    self.current_output_type = 1
    self.error_outputs.append(outputObj)

class _output_obj():
  def __init__(self,output="",model_id=""):
    self._output = output
    self._model_id = model_id

class Pipable():
  def __init__(self, dataType = "", PGname = "", PGhost = "", PGuser = "", PGpass = "", PGport = 5432 , pathToData = "", pathToADD = "", openaiKEY="", googleCustomKEY="", googleProgrammableKEY=""):
    super().__init__()
    openai_APIKEY = openaiKEY
    google_custom_api_key = googleCustomKEY
    programmable_search_engine_api_key = googleProgrammableKEY

    _tempfile = open(pathToADD)
    self.action_desc = json.load(_tempfile)
    _schema=self.action_desc.pop("schema")
    _tempfile.close()

    self.ada_ = _ada(openaiAPIKEY=openai_APIKEY).initialize()
    self.sem_s = _semantic_search().initialize()
    self.askgoogle = _google_search().initialise(google_api_key=google_custom_api_key,search_engine_key=programmable_search_engine_api_key)

    if dataType == "csv":
      self.datasearch = _csv_search(openai_key=openai_APIKEY,path_csv_file=pathToData).initialize(schema = _schema)
      self.agg_stat = _aggregated_stats().initialize(pathToData)
    elif dataType == "postgres":
      self.datasearch = _postgres_search(openai_key=openai_APIKEY,PGname=PGname,PGhost=PGhost,PGuser=PGuser,PGpass=PGpass,PGport=PGport).initialize()
      # URGENT -> need to implement aggregated stats for postgres
    elif dataType == "mysql":
      print("ERROR: mysql data type not yet implemented. Valid data types are csv and postgres.")
    elif dataType == "json":
      print("ERROR: json data type not yet implemented. Valid data types are csv and postgres.")
    else:
      print("Error: no valid data type specified. Valid data types are csv and postgres.")
      return None
    
    self.key2method = {
      "ada":self.ada_.ask_ada,
      "semantic_search":self.sem_s.find_similar_score,
      "agg_stats":self.agg_stat.get_stats,
      "agg_corr":self.agg_stat.get_corr,
      "google_search":self.askgoogle.ask_google,
      "data_search":self.datasearch.search_data_natural
    }

    self.sem_s.create_key_vectors(list(self.action_desc.values()))
    self.results_proxy = _proxy_results()
  
  def ask(self,query,model=""):
    # model auto selection based on query
    if model == "":
      score = int(jnp.argmax(self.sem_s.find_similar_score(query_list=query)))
      model = list(self.action_desc.keys())[score]
    # specified or auto selected model is valid
    if model in self.action_desc:
      result = self.key2method[model](query)
      
      if model == "ada":
        google_search_result = self.askgoogle.ask_google(result)
        self.results_proxy.update_ada_thread(result)
        self.results_proxy.update_action([model,"google_search"])
        self.results_proxy.update_outputobjs(_output_obj(output=[result,google_search_result],model_id=[model,"google_search"]))
    
      elif model == "data_search":
        self.results_proxy.update_action(model)
        if result[1] == "normal":
          self.results_proxy.update_outputobjs(_output_obj(output=result[0],model_id=model))
        else:
          self.results_proxy.update_error_outputobjs(_output_obj(output=result[0],model_id=model))
      else:
        self.results_proxy.update_action(model)
        self.results_proxy.update_outputobjs(_output_obj(output=result,model_id=model))
    # specified model is invalid
    else:
      print(model, "- No such model found. Ensure that correct model_id is entered. Refer to .get_help() for model ids.")
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
    return {"outputs":[x._output for x in self.results_proxy.output_objects],"errors":[x._output for x in self.results_proxy.error_outputs]}