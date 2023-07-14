import copy

from IPython.display import HTML, display


class _llm():
  def __init__(self,llm_thread="",openaiAPIKEY=""):
    super().__init__()
    self.llm_thread=llm_thread
    self.openai_api_key = openaiAPIKEY

  def ask_llm(self,query):
    llm_thread = self.llm_thread
    temp_query = copy.deepcopy(query)
    if llm_thread == "":
      query=query
    else:
      query = llm_thread+". "+query
    import os

    import openai
    openai.api_key =self.openai_api_key
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
      messages=[
        {"role": "user","content":query }
      ]
    )
    text = completion.choices[0].message.content
    print(f"LLM: {text}")
    if llm_thread!="":
      self.llm_thread += ". "+text
    else:
      self.llm_thread = text
    return text

  def reset_thread(self):
    self.llm_thread = ""