import openai
import pandas as pd


class _csv_search():
  def __init__(self, openai_key = "", df = None, pathlog = ""):
    super().__init__()
    self.openai_key = openai_key
    self.df = df
    self.df_schema = ""
    self.pathlog = pathlog

  def autoschema(self):
    self.df_schema = "df = pd.DataFrame({\n"
    for i in range(len(self.df.columns)):
      self.df_schema += "\t{}: {},\n".format(self.df.columns[i],self.df.dtypes[i])
    self.df_schema+="})"

  def initialize(self):
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
      temperature=0.5,
      model="gpt-3.5-turbo",
      messages=[{"role": "user","content":prompt }]
    )
    obj = completion.choices[0].message.content

    print(obj)
    df = self.df
    try:
      df = eval(obj)
      print("Query executed successfully.")
      # log success
      current_log = pd.DataFrame({
        "timestamp": [pd.Timestamp.now()],
        "query": [query],
        "datatype": ["csv"],
        "database": [self.pathlog],
        "pipableAI": [""],
        "openAI": [obj],
        "success": [True],
        "error": [""]
      })
      temp = pd.read_parquet("logs.parquet", engine = 'pyarrow')
      pd.concat([temp, current_log], ignore_index = True).to_parquet("logs.parquet", engine = 'pyarrow')
      return df
    except Exception as e:
      print("Generated query failed. Try regenerating.")
      # log error
      current_log = pd.DataFrame({
        "timestamp": [pd.Timestamp.now()],
        "query": [query],
        "datatype": ["csv"],
        "database": [self.pathlog],
        "pipableAI": [""],
        "openAI": [obj],
        "success": [False],
        "error": [str(e)]
      })
      temp = pd.read_parquet("logs.parquet", engine = 'pyarrow')
      pd.concat([temp, current_log], ignore_index = True).to_parquet("logs.parquet", engine = 'pyarrow')
      return None