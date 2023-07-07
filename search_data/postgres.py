import openai
import pandas as pd
import psycopg2


class _postgres_search():
  def __init__(self, openai_key = "", openai = None, PGname = "", PGhost = "", PGuser = "", PGpass = "", PGport = 5432, PGsche = "", cursor = None, conn = None, db = None, agent = None):
    super().__init__()
    self.openai_key = openai_key
    self.PGname = PGname
    self.PGhost = PGhost
    self.PGuser = PGuser
    self.PGpass = PGpass
    self.PGport = PGport
    self.PGsche = PGsche
    self.cur = cursor
    self.conn = conn
    self.pg_schema = ""
    self._queries=[]

  def initialize(self, schema):
    self.conn = psycopg2.connect(
      user=self.PGuser,
      password=self.PGpass,
      host=self.PGhost,
      port=self.PGport,
      database=self.PGname,
      options=f"-c search_path={self.PGsche}"
    )
    self.cur = self.conn.cursor()
    for i in schema:
      self.pg_schema += "{}(".format(i)
      for j in schema[i]['keys']:
        self.pg_schema += "{}, ".format(j)
      self.pg_schema += "\b\b)\n"
    return self

  def search_data_natural(self, query):
    prompt = (
        " Only return the postgres query. Don't return any comments."
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
      self.cur.execute(obj)
      df = pd.DataFrame(self.cur.fetchall())
      self._queries.append((obj,"normal"))
      return (df,"normal")
    except Exception as e:
      print("Some error has occured!")
      self._queries.append((obj,"error"))
      return (obj,"error")
  
  def get_queries(self):
    return self._queries
  





# import os
# import openai

# openai.api_key = os.getenv("OPENAI_API_KEY")

# response = openai.Completion.create(
  # model="text-davinci-003",
  # prompt="### Postgres SQL tables, with their properties:\n#\n# Employee(id, name, department_id)\n# Department(id, name, address)\n# Salary_Payments(id, employee_id, amount, date)\n#\n### A query to list the names of the departments which employed more than 10 employees in the last 3 months\nSELECT",
  # temperature=0,
  # max_tokens=150,
  # top_p=1.0,
  # frequency_penalty=0.0,
  # presence_penalty=0.0,
  # stop=["#", ";"]
# )

# Employee:
#   keys:
#     - id
#     - name
#     - department_id
#   dataTypes:
#     - int
#     - string
#     - int
#   descriptors:
# Department:
#   keys:
#     - id
#     - name
#     - address
#   dataTypes:
#     - int
#     - string
#     - string
#   descriptors:
# Salary_Payments:
#   keys:
#     - id
#     - employee_id
#     - amount
#     - date
#   dataTypes:
#     - int
#     - int
#     - int
#     - date
#   descriptors: