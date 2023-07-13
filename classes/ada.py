import copy
from pprint import pprint

from IPython.display import HTML, display


class _ada():
  def __init__(self,ada_model=None,ada_tokenizer=None,ada_thread="",embedder=None,table=None,openaiAPIKEY=""):
    super().__init__()
    self.ada_model=ada_model
    self.ada_tokenizer = ada_tokenizer
    self.ada_thread=ada_thread
    self.openai_api_key = openaiAPIKEY

  def initialize(self):
    ada_thread=self.ada_thread
    return self

  def ask_ada(self,query):
    ada_thread = self.ada_thread
    temp_query = copy.deepcopy(query)
    if ada_thread == "":
      query=query
    else:
      query = ada_thread+". "+query
    import os

    import openai
    openai.api_key =self.openai_api_key
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
      messages=[
        {"role": "user","content":query }
      ]
    )
    text = completion.choices[0].message.content
    pprint(f"Ada: {text}")
    if ada_thread!="":
      self.ada_thread += ". "+text
    else:
      self.ada_thread = text
    pprint(display(HTML('<span style="color:#4285F4">Ada: Anything else I can help you with ? : </span>')))
    return text

  def reset_thread(self):
    self.ada_thread = ""