import openai
import pandas as pd
import psycopg2


class _postgres_search():
  def __init__(self, openai_key = "", openai = None, file_path = {}, cursor = None, conn = None, db = None, agent = None):
    super().__init__()
    self.openai_key = openai_key
    self.PGname = file_path["pgdata"]
    self.PGhost = file_path["pghost"]
    self.PGuser = file_path["pguser"]
    self.PGpass = file_path["pgpass"]
    self.PGport = file_path["pgport"]
    self.PGsche = file_path["pgsche"]
    self.cur = cursor
    self.conn = conn
    self.pg_schema = ""
    self._queries=[]

  def autoschema(self):
    self.cur.execute("SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = '{}' ORDER BY table_name;".format(self.PGsche))
    rows = self.cur.fetchall()

    schema = "Postgres SQL tables formatted as table_name(column_name:data_type,column_name:data_type,...)\n"
    
    schema += "{}(".format(rows[0][0])
    for i in range(1, len(rows)):
      if rows[i][0] != rows[i-1][0]:
        schema += ")\n{}(".format(rows[i][0])
      schema += "{}:{},".format(rows[i][1], rows[i][2])
    schema += ")\n"
    return schema

  def initialize(self):
    self.conn = psycopg2.connect(
      user=self.PGuser,
      password=self.PGpass,
      host=self.PGhost,
      port=self.PGport,
      database=self.PGname,
      options=f"-c search_path={self.PGsche}"
    )
    self.cur = self.conn.cursor()
    self.pg_schema = self.autoschema()
    return self

  def search_data(self, query):
    prompt = (
        "Only return the postgres query. Don't return any comments."
        + self.pg_schema + "Task :" + query
    )
    openai.api_key =self.openai_key
    completion = openai.ChatCompletion.create(
      temperature=0.8,
      model="gpt-3.5-turbo",
      messages=[{"role": "user","content":prompt }]
    )
    obj = completion.choices[0].message.content
    try:
      print(obj)
      self.cur.execute(obj)
      print("Query executed successfully")
      df = pd.DataFrame(self.cur.fetchall())
      self._queries.append((obj,"normal"))
      df.columns = [desc[0] for desc in self.cur.description]
      print(df)
      df.to_parquet('./parquet_files/{}_output.parquet'.format(len(self._queries)-1))
      return {"object_type":"DataFrame","output_file_name":"{}_output.parquet".format(len(self._queries)),"exec":"successful"}
    except Exception as e:
      print("Some error has occured!")
      self._queries.append((obj,"error"))
      return {"object_type":"None","exec":"error"}
  
  def get_queries(self):
    return self._queries