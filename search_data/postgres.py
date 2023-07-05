import psycopg2
from langchain.llms.openai import OpenAI
from langchain.sql_database import SQLDatabase
from langchain import SQLDatabaseChain

class _postgres_search():
  def __init__(self, openai_key = "", openai = None, PGname = "", PGhost = "", PGuser = "", PGpass = "", PGport = "", cursor = None, conn = None, db = None, agent = None):
    super().__init__()
    self.openai_key = openai_key
    self.openai = openai
    self.PGname = PGname
    self.PGhost = PGhost
    self.PGuser = PGuser
    self.PGpass = PGpass
    self.PGport = PGport
    self.cur = cursor
    self.conn = conn
    self.db = db
    self.agent = agent

  def initialize(self):
    print("debug before AI init")
    self.openai = OpenAI(temperature=0, openai_api_key=self.openai_key)

    # print("debug before psycopg2 init")
    # self.conn = psycopg2.connect(
    #   dbname=self.PGname,
    #   user=self.PGuser,
    #   password=self.PGpass,
    #   host=self.PGhost,
    #   port=self.PGport
    # )
    # self.cur = self.conn.cursor()

    print("debug before SQLDatabase init")
    self.db = SQLDatabase.from_uri(
      f"postgresql+psycopg2://{self.PGuser}:{self.PGpass}@{self.PGhost}:{self.PGport}/{self.PGname}",
    )

    print("debug before SQLDatabaseChain init")
    self.agent = SQLDatabaseChain.from_llm(llm = self.openai, db = self.db, verbose=True)

    return self

  def search_data_natural(self, query):
    prompt = (
        """
            Given an input question, first create a syntactically correct postgresql query to run, then look at the results of the query and return the answer.
            Use the following format:

            Question: "Question here"
            SQLQuery: "SQL Query to run"
            SQLResult: "Result of the SQLQuery"
            Answer: "Final answer here"

            {question}

              def search_data(self, query):
                pass
        """
        + query
    )
    obj = self.agent({"query":prompt})
    return obj["answer"]