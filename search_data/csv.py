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

  def autoschema(self):
    self.df_schema = "df = pd.DataFrame({\n"
    for i in range(len(self.df.columns)):
      self.df_schema += "\t{}: {},\n".format(self.df.columns[i],self.df.dtypes[i])
    self.df_schema+="})"

  def initialize(self):
    self.df = pd.read_csv(self.path_to_csv)
    self.autoschema()
    return self

  def search_data(self, query):
    prompt = (
        "Only return the pandas query. Don't return any comments."
        + self.df_schema + "Task :" + query
    )
    print(self.df_schema)
    openai.api_key =self.openai_key
    completion = openai.ChatCompletion.create(
      temperature=0.8,
      model="gpt-3.5-turbo",
      messages=[{"role": "user","content":prompt }]
    )
    obj = completion.choices[0].message.content
    try:
      print(obj)
      df = self.df
      df = eval(obj)
      self._queries.append((obj,"normal"))
      df.to_parquet('./parquet_files/{}_output.parquet'.format(len(self._queries)-1))
      return {"object_type":"DataFrame","output_file_name":"{}_output.parquet".format(len(self._queries)-1),"exec":"successful"}
    except Exception as e:
      print("Some error has occured!")
      self._queries.append((obj,"error"))
      return {"object_type":"None","exec":"error"}
  
  def get_queries(self):
    return self._queries
