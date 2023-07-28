import openai


class _llm():
  def __init__(self,llm_thread=[],openaiAPIKEY=""):
    super().__init__()
    self.llm_thread = llm_thread
    self.openai_api_key = openaiAPIKEY
    self.MAXLEN = 12000

  def reset_thread(self):
    self.llm_thread = []

  def _auto_minimal_query(self, query):
    querylen = len(query.split())
    if querylen > self.MAXLEN:
      return (1, "Query is too long. Please shorten it to less than 16384 tokens.")
    else:
      while querylen + len((' '.join(self.llm_thread).split())) > self.MAXLEN:
        self.llm_thread.pop(0)
      prequery = ' '.join(self.llm_thread)
      query = prequery + '\n' + query
      return (0, query)

  def ask_llm(self,query):
    flag, query = self._auto_minimal_query(query)
    if flag == 1:
      print(query)
      return (1, query)

    openai.api_key = self.openai_api_key
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",
      messages=[
        {"role": "user","content":query }
      ]
    )
    result = completion.choices[0].message.content
    print(f"LLM: {result}")

    self.llm_thread.append(result)

    return (0, result)
  
# returning 0 means success, 1 means error
# code to handle error is not written yet