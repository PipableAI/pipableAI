from search_data.aggregated_stats import _aggregated_stats
from search_data.csv import _data_search
from search_data.semantic import _semantic_search
from search_engine.ada import _ada
from search_engine.google import _google_search
import jax.numpy as jnp

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
  def __init__(self,pathToCSV = "",actions=[],func_desc=[]):
    super().__init__()
    self.ada_ = _ada().initialize()
    self.sem_s = _semantic_search().initialize()
    self.askgoogle = _google_search().initialise()
    self.agg_stat = _aggregated_stats().initialize(pathToCSV)
    self.datasearch = _data_search().initialize()
    self.action_descriptions= ["Ask generic health questions like - why is my blood pressure high ? what causes increase in heart rate ?, what leads to sudden drop in blood pressure ?, why increase in weight is a risk to heart ? why sudden loss of weight can be a risk to heart ? Similar health related queries accompanied with ask Ada. Ask Ada:",
                               "Perform semantic search given a query. Queries can be like find similar items",
                               "find over all statistics of the data given",
                               "find correlation amongst different features",
                               "Query google to find answers to certain questions. Can you use google to get me the results, look at google , get search results , get reference links, do google search.",
                               "Get me the results of all patients. Show me the results of a particular entity. Get me the list of entities where val > X. Analyse the data in a particular time period."]
    self.actions = [self.ada_.ask_ada,self.sem_s.find_similar_score,self.agg_stat.get_stats,self.agg_stat.get_corr,self.askgoogle.ask_google,self.datasearch.search_csv_natural]
    self.sem_s.create_key_vectors(self.action_descriptions)
    self.results_proxy = _proxy_results()
    self.model_ids = ["_ada","_semantic_search","agg_stats","agg_corr","_google_search","datasearch"]

  def temp_ask(self,query):
    action_descriptions = self.action_descriptions
    score = self.actions[1](query_list=query)
    # action = self.actions[int(jnp.argmax(score))]
    return int(jnp.argmax(score))

  def ask(self,query,model=""):
    model_id = self.parse_model_input(model)
    if model_id == -1:
      return
    if (model_id == 0):
      model_id = self.temp_ask(query)
      model_id+=1
    result = self.actions[model_id-1](query)
    if(model_id == 1):
      self.askgoogle.ask_google(result)
      self.results_proxy.update_ada_thread(result)
    self.results_proxy.update_action(model)
    self.results_proxy.update_outputobjs(_output_obj(output=result,model_id=self.model_ids[model_id-1]))
    return

  def parse_model_input(self,model):
    model_found=0
    for i in range(len(self.model_ids)):
      if self.model_ids[i] == model:
        model_found = i+1
        break

    if (model_found == 0 and model != ""):
      print("No such model found. Ensure that correct model_id is entered. For model ids refer to .get_help() for more details.")
      return -1

    return model_found

  def get_help(self):
    print("You can ask any question using the ask function. It takes two Parameters, query and model. Different models and their query are mentioned below :-")
    print("MODEL             DESCRIPTION")
    for i in range(len(self.model_ids)):
      print("{}                   :- {}".format(self.model_ids[i],self.action_descriptions[i]))


# # sample usage
# a = Pipable(pathToCSV="sample_data/alyf.csv")
# a.ask("Get all patient ids and vital in the form of table that have vitals as Heart Rate and value between 100 to 150 between march to april 2023")