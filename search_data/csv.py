import pandas as pd
from langchain.openai import OpenAI
from langchain.agents import create_pandas_dataframe_agent

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
    self.df = pd.read_csv('/content/drive/My Drive/pipable/csv_data/alyf.csv')
    self.agent = create_pandas_dataframe_agent(self.openai, self.df, verbose=True)
    self.agent_data = create_pandas_dataframe_agent(self.openai, self.df, verbose=True, return_intermediate_steps=True)
    return self

  def search_csv_natural(self, query):
    return self.agent.run(query)

  # CSV data when returned is not as parseable as SQL data is, prefer natural language when using CSV
  def search_csv_data(self, query):
    response = self.agent_data({"input":query})
    return response["intermediate_steps"][-1]