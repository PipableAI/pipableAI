import openai
import pandas as pd
import psycopg2


class _postgres_search():
  def __init__(self, openai_key = "", openai = None, file_path = {}, cursor = None, conn = None, db = None, agent = None, context = ""):
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
    self.context = context

  def autoschema(self):
    self.cur.execute("SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = '{}' ORDER BY table_name;".format(self.PGsche))
    rows = self.cur.fetchall()
    self.pg_schema = '''
    Format:
    [(table_name, column_name, data_type),...]
    '''
    self.pg_schema += str(rows)

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
    self.autoschema()
    return self

  def search_data(self, query):
    prompt = (
        "Only return the postgres query. Don't return any comments."
        + self.pg_schema
        + self.context
        + "Task: " + query
    )
    openai.api_key =self.openai_key
    completion = openai.ChatCompletion.create(
      temperature=0.5,
      model="gpt-3.5-turbo",
      messages=[{"role": "user","content":prompt }]
    )
    obj = completion.choices[0].message.content
    print(obj)

    try:
      self.cur.execute(obj)
      df = pd.DataFrame(self.cur.fetchall())
      df.columns = [desc[0] for desc in self.cur.description]
      print("Query executed successfully.")
      # log success
      current_log = pd.DataFrame({
        "timestamp": [pd.Timestamp.now()],
        "query": [query],
        "datatype": ["postgres"],
        "database": [self.PGsche],
        "pipableAI": [""],
        "openAI": [obj],
        "success": [True],
        "error": [""]
      })
      temp = pd.read_parquet("logs.parquet", engine = 'pyarrow')
      pd.concat([temp, current_log], ignore_index = True).to_parquet("logs.parquet", engine = 'pyarrow')
      return (0, df)
    except Exception as e:
      print("Generated query failed. Try regenerating.")
      # log error
      current_log = pd.DataFrame({
        "timestamp": [pd.Timestamp.now()],
        "query": [query],
        "datatype": ["postgres"],
        "database": [self.PGsche],
        "pipableAI": [""],
        "openAI": [obj],
        "success": [False],
        "error": [str(e)]
      })
      temp = pd.read_parquet("logs.parquet", engine = 'pyarrow')
      pd.concat([temp, current_log], ignore_index = True).to_parquet("logs.parquet", engine = 'pyarrow')
      return (1, str(e))