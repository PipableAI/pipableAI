import openai
import pandas as pd


class _csv_search():
  def __init__(self, openai_key = "", df = None,file_path=""):
    super().__init__()
    self.openai_key = openai_key
    self.df = df
    self.path_to_csv = file_path
    self.df_schema = ""
    self._queries=[]

  def initialize(self,schema):
    self.df = pd.read_csv(self.path_to_csv)
    self.df_schema = "df = pd.DataFrame({"
    for i in range(len(schema["keys"])):
      self.df_schema += "{}:{} #{}".format(schema["keys"][i],schema["dataTypes"][i],schema["descriptors"][i])
    self.df_schema+="})" 
    return self

  def search_data(self, query):
    prompt = (
        "Only return the pandas query. Don't return any comments."
        + self.df_schema + "Task :" + query
    )
    openai.api_key =self.openai_key
    completion = openai.ChatCompletion.create(
      temperature=0.8,
      model="gpt-3.5-turbo",
      messages=[{"role": "user","content":prompt }]
    )
    obj = completion.choices[0].message.content
    print(obj)
    try:
      df = eval(obj)
      self._queries.append((obj,"normal"))
      return (df,"normal")
    except Exception as e:
      print("Some error has occured!")
      self._queries.append((obj,"error"))
      return (obj,"error")
  
  def get_queries(self):
    return self._queries