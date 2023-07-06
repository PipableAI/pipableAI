import pandas as pd
import json
import openai

class _data_search():
  def __init__(self, openai_key = "", df = None,path_csv_file=""):
    super().__init__()
    self.openai_key = openai_key
    self.df = df
    self.path_to_csv = path_csv_file
    self.df_schema = ""
    self._queries=[]

  def initialize(self,schema):
    self.df = pd.read_csv(self.path_to_csv)
    self.df_schema = "df = pd.DataFrame({"
    for i in range(len(schema["keys"])):
      self.df_schema += "{}:{} #{}".format(schema["keys"][i],schema["dataTypes"][i],schema["descriptors"][i])
    self.df_schema+="})" 
    return self

  def search_csv_natural(self, query):
    prompt = (
        '''
        Only return the pandas query.

        Don't return any comments.
        
    '''
        + self.df_schema + "Task :" + query
    )
    openai.api_key =self.openai_key
    completion = openai.ChatCompletion.create(temperature=0.8,model="gpt-3.5-turbo",
      messages=[
        {"role": "user","content":prompt }
      ]
    )
    obj = completion.choices[0].message.content
    try:
      df = self.df
      df4 = eval(obj)
      self._queries.append((obj,"normal"))
      return (df4,"normal")
    except Exception as e:
      print("Some error has occured!")
      self._queries.append((obj,"error"))
      return (obj,"error")

  # CSV data when returned is not as parseable as SQL data is, prefer natural language when using CSV
  def search_csv_data(self, query):
    response = self.agent_data({"input":query})
    return (response["intermediate_steps"][-1],"normal")
  
  def get_queries(self):
    return self._queries