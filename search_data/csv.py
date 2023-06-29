import pandas as pd
from langchain.llms.openai import OpenAI
from langchain.agents import create_pandas_dataframe_agent
import json
class data_search():
  def __init__(self, tempkey = "", openai = None, df = None, agent = None, agent_data = None):
    super().__init__()
    self.tempkey = tempkey
    self.openai = openai
    self.df = df
    self.agent = agent
    self.agent_data = agent_data

  def initialize(self):
    self.tempkey = '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'
    self.openai = OpenAI(temperature=0, openai_api_key=self.tempkey)
    self.df = pd.read_csv("csv_files/alyf.csv")
    self.agent = create_pandas_dataframe_agent(self.openai, self.df, verbose=True)
    self.agent_data = create_pandas_dataframe_agent(self.openai, self.df, verbose=True, return_intermediate_steps=True)
    return self

  def search_csv_natural(self, query):
    prompt = (
        """
            For the following query, if it requires drawing a table, reply as follows:
            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

            If the query requires creating a bar chart,  reply as follows:
            {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            If the query requires creating a line chart, reply as follows:
            {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            There can only be two types of chart, "bar" and "line".

            If it is just asking a question that requires neither, reply as follows:
            {"answer": "answer"}
            Example:
            {"answer": "The title with the highest rating is 'Gilead'"}

            If you do not know the answer, reply as follows:
            {"answer": "I do not know."}

            Return all output as a string.

            All strings in "columns" list and data list, should be in double quotes,

            For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}

            Lets think step by step.

            Below is the query.
            Query:
            """
        + query
    )
    obj = self.agent.run(prompt)
    string_data = obj.__str__()
    string_data = string_data.replace("'", "\"")
    json_data = json.loads(string_data)
    return json_data

  # CSV data when returned is not as parseable as SQL data is, prefer natural language when using CSV
  def search_csv_data(self, query):
    response = self.agent_data({"input":query})
    return response["intermediate_steps"][-1]