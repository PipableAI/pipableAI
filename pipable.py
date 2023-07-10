import jax.numpy as jnp
import yaml

from search_data.csv import _csv_search
from search_data.postgres import _postgres_search
from search_data.semantic import _semantic_search
from search_engine.ada import _ada
from search_engine.google import _google_search
import pandas as pd

class _proxy_results():
  def __init__(self):
    self.ada_thread = ""
    self.action_list=[]
    self.output_objects = []
    self.error_outputs = []
    self.human_prompt = []
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
  
  def reset_outputs(self):
    self.output_objects = []
    self.error_outputs = []
    self.current_output_type = -1
    self.ada_thread = ""
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

    openai_APIKEY = config["keys"]["openAI"]
    google_custom_api_key = config["keys"]["google"]
    programmable_search_engine_api_key = config["keys"]["search_engine"]

    self.ada_ = _ada(openaiAPIKEY=openai_APIKEY).initialize()
    self.sem_s = _semantic_search().initialize()
    self.askgoogle = _google_search().initialise(google_api_key=google_custom_api_key,search_engine_key=programmable_search_engine_api_key)

    dataType = config["dataType"]

    if dataType == "csv":
      self.datasearch = _csv_search(openai_key=openai_APIKEY,file_path=config["pathToData"]).initialize(schema = config["schema"])
    elif dataType == "postgres":
      self.datasearch = _postgres_search(openai_key=openai_APIKEY,file_path=config["pathToData"]).initialize(schema = config["schema"])
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
      "google_search":self.askgoogle.ask_google,
      "data_search":self.datasearch.search_data
    }

    self.action_desc = config["action_desc"]

    self.sem_s.create_key_vectors(list(self.action_desc.values()))
    self.results_proxy = _proxy_results()

    import os
    folder_path = './parquet_files'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return
  
  def ask(self,query,model=""):
    self.results_proxy.update_human_prompt(query)
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
        self.results_proxy.update_outputobjs(_output_obj(output={"object_type":"string","summary":result,"sources":google_search_result},model_id=[model,"google_search"]))
    
      elif model == "data_search":
        if result["exec"] == "successful":
          self.results_proxy.update_action(model)
          self.results_proxy.update_outputobjs(_output_obj(output=result,model_id=model))
        else:
          self.results_proxy.update_error_outputobjs(_output_obj(output=result,model_id=model))

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
  
  def reset_chain(self):
    self.results_proxy.reset_outputs()
  
  def parse_parquet_to_DataFrame(self,filename):
    return pd.read_parquet("./parquet_files/{}".format(filename), engine='pyarrow')

    