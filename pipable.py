from search_data.aggregated_stats import aggregated_stats
from search_data.csv import data_search
from search_data.semantic import semantic_search
from search_engine.ada import ada
from search_engine.google import google_search
from pprint import pprint
import jax.numpy as jnp

class proxyResults():
  def __init__(self):
    self.ada_thread = ""
    self.action_list=[]
    # self.model_list = ["ada","sem-search","google-search","aggregated_stats"]

  def update_ada_thread(self,thread):
    self.ada_thread = thread

  def update_action(self,model_id):
    self.action_list.append(model_id)

class masterClass():
  def __init__(self,pathToCSV = "",actions=[],func_desc=[]):
    super().__init__()
    self.ada_ = ada().initialize()
    self.sem_s = semantic_search().initialize()
    self.askgoogle = google_search().initialise()
    self.agg_stat = aggregated_stats().initialize(pathToCSV)
    self.datasearch = data_search().initialize()
    self.action_descriptions= ["Ask generic health questions like - why is my blood pressure high ? what causes increase in heart rate ?, what leads to sudden drop in blood pressure ?, why increase in weight is a risk to heart ? why sudden loss of weight can be a risk to heart ? Similar health related queries accompanied with ask Ada. Ask Ada:",
                               "Perform semantic search given a query. Queries can be like find similar items",
                               "find over all statistics of the data given",
                               "find correlation amongst different features",
                               "Query google to find answers to certain questions. Can you use google to get me the results, look at google , get search results , get reference links, do google search.",
                               "Get me the results of all patients. Show me the results of a particular entity. Get me the list of entities where val > X. Analyse the data in a particular time period."]
    self.actions = [self.ada_.ask_ada,self.sem_s.find_similar_score,self.agg_stat.get_stats,self.agg_stat.get_corr,self.askgoogle.ask_google,self.datasearch.search_csv_data]
    self.sem_s.create_key_vectors(self.action_descriptions)
    self.results_proxy = proxyResults()

  def temp_ask(self,query):
    action_descriptions = self.action_descriptions
    #print(query)
    score = self.actions[1](query_list=query)
    action = self.actions[int(jnp.argmax(score))]
    return (action,int(jnp.argmax(score)))

  def ask(self,query,model=0):
    if (model == 0):
      action,model_id = self.temp_ask(query)
      model_id+=1
      result=action(query)
      if(model_id == 1):
        self.askgoogle.ask_google(result)
        self.results_proxy.update_ada_thread(result)
      self.results_proxy.update_action(model_id)
    elif (model > 5):
      pprint("No Such Model Found. Please enter correct Model ID")
      return
    else:
      result = self.actions[model-1](query)
      if(model == 1):
        self.askgoogle.ask_google(result)
        self.results_proxy.update_ada_thread(result)
      self.results_proxy.update_action(model)
    return