from search_data.aggregated_stats import _aggregated_stats
from search_data.csv import _data_search
from search_data.semantic import _semantic_search
from search_engine.ada import _ada
from search_engine.google import _google_search

import jax.numpy as jnp
import json

class _proxy_results():
  def __init__(self):
    self.ada_thread = ""
    self.action_list=[]
    self.output_objects = []
    # self.model_list = ["_ada","sem-search","google-search","_aggregated_stats"]

  def update_ada_thread(self,thread):
    self.ada_thread = thread

  def update_action(self,model_id):
    self.action_list.append(model_id)

  def update_outputobjs(self,outputObj=None):
    self.output_objects.append(outputObj)

class _output_obj():
  def __init__(self,output="",model_id=""):
    self._output = output
    self._model_id = model_id

class Pipable():
  def __init__(self,pathToCSV = "",pathToADD = "",actions=[],func_desc=[], openaiKEY="", googleCustomKEY="", googleProgrammableKEY=""):
    super().__init__()
    openai_APIKEY = openaiKEY
    google_custom_api_key = googleCustomKEY
    programmable_search_engine_api_key = googleProgrammableKEY

    self.ada_ = _ada(openaiAPIKEY=openai_APIKEY).initialize()
    self.sem_s = _semantic_search().initialize()
    self.askgoogle = _google_search().initialise(google_api_key=google_custom_api_key,search_engine_key=programmable_search_engine_api_key)
    self.agg_stat = _aggregated_stats().initialize(pathToCSV)
    self.datasearch = _data_search(openai_key=openai_APIKEY,path_csv_file=pathToCSV).initialize()

    _tempfile = open(pathToADD)
    self.action_desc = json.load(_tempfile)
    _tempfile.close()
    
    self.actions = [self.ada_.ask_ada,self.sem_s.find_similar_score,self.agg_stat.get_stats,self.agg_stat.get_corr,self.askgoogle.ask_google,self.datasearch.search_csv_natural]
    self.sem_s.create_key_vectors(list(self.action_desc.values()))
    self.results_proxy = _proxy_results()

  def temp_ask(self,query):
    #action_descriptions = self.action_descriptions
    score = self.actions[1](query_list=query)
    # action = self.actions[int(jnp.argmax(score))]
    return int(jnp.argmax(score))
  
  def ask(self,query,model=""):
    if model == "":
      model = list(self.action_desc.keys())[(self.temp_ask(query))]
    if model in self.action_desc:
      result = self.actions[list(self.action_desc).index(model)](query)
      if model == "_ada":
        self.askgoogle.ask_google(result)
        self.results_proxy.update_ada_thread(result)
      self.results_proxy.update_action(model)
      self.results_proxy.update_outputobjs(_output_obj(output=result,model_id=model))
    else:
      print(model, "- No such model found. Ensure that correct model_id is entered. Refer to .get_help() for model ids.")
    return

  def get_help(self):
    print("You can ask any question using the ask function. It takes two Parameters, query and model. Different models and their query are mentioned below:")
    print("MODEL\t\tDESCRIPTION")
    for i in self.action_desc:
      print(i,"\t\t",self.action_desc[i])
  
  def get_latest_output(self):
    if len(self.results_proxy.output_objects) == 0:
      print("No outputs found")
      return
    return self.results_proxy.output_objects[-1]._output

  def get_all_outputs(self):
    return [x._output for x in self.results_proxy.output_objects]

# # sample usage
# a = Pipable(
#   pathToCSV="sample_data/medSample.csv",
#   pathToADD="sample_data/medSampleADD.json",
#   openaiKEY="OPENAI_API_KEY",
#   googleCustomKEY="GOOGLE_CUSTOM_SEARCH_API_KEY",
#   googleProgrammableKEY="GOOGLE_PROGRAMMABLE_SEARCH_ENGINE_API_KEY"
# )
#a.ask("Get all patient ids and vital in the form of table that have vitals as Heart Rate and value between 100 to 150 between march to april 2023")
#a.ask("What all risks are associated with the increase in the heart rate")