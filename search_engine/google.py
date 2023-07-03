from googleapiclient.discovery import build
import re
from html import unescape
from pprint import pprint
from IPython.display import HTML
from IPython.display import display

class _google_search():
  def __init__(self,_google_search=None,frame=None,
               past_query=[],current_query=None,current_input=None,
               action=[],current_action=None,
               output=[],current_output=None,correct_output=[],programmable_search_engine_api_key="",past_results_urls=[],past_snippets=[]):
    super().__init__()
    self.past_query=past_query
    self.current_query=current_query
    self.current_input=current_input
    self._google_search = _google_search
    self.correct_output=correct_output
    self.past_results_urls = past_results_urls
    self.past_snippets = past_snippets
    self.programmable_search_engine_api_key = programmable_search_engine_api_key

  def initialise(self,google_api_key = "",search_engine_key=""):
    _google_search = build('customsearch', 'v1', developerKey=google_api_key)
    self.programmable_search_engine_api_key = search_engine_key
    self._google_search = _google_search
    return self

  def ask_google(self,query):
    def clean_text(text):
      clean_text = re.sub('<.*?>', '', text)
      clean_text = unescape(clean_text)
      clean_text = re.sub('[^a-zA-Z0-9\s]', '', clean_text)
      clean_text = re.sub('\s+', ' ', clean_text).strip()
      return clean_text
    pprint(display(HTML('<span style="color:#DB4437">Google Search Loaded: What is your query ?</span>')))
    self.past_query.append(query)
    pprint(display(HTML('<span style="color:#4285F4">Google search: fetching results : </span>')))
    gs = self._google_search
    print("Query :- {}".format(query))
    response = gs.cse().list(q=query, cx=self.programmable_search_engine_api_key).execute()
    ite=1
    list_of_snippets = []
    list_of_urls = []
    #print(response)
    for item in response['items']:
      cleantext = clean_text(item['snippet'])
      pprint("Link {}:- {}".format(ite,item['link']))
      pprint("Result {}:- {}".format(ite,cleantext))
      ite+=1
      list_of_urls.append(item['link'])
      list_of_snippets.append(cleantext)
      pprint(display(HTML('<span style="color:#DB4437">Google search: Next result : </span>')))
    self.past_results_urls.append(list_of_urls)
    self.past_snippets.append(list_of_snippets)
    pprint(display(HTML('<span style="color:#DB4437">Ada: Is there anything else I can do ? You can check my action catalogue to know all my skills and you can use .correct_output, .output , .querry , .input , .actions, .ada_thread methods to check our peer development thread. </span>')))
    pprint(display(HTML('<span style="color:#0F9D58">Ada: You can also ask google or open ai using .ask_google() or .ask_open_ai() methods </span>')))
    # return self

  def get_latest_google_search_results(self):
    num_of_results = len(self.past_snippets)
    if num_of_results == 0:
      raise Exception("No Results Found")
    latest_snippets = self.past_snippets[num_of_results-1]
    latest_urls = self.past_results_urls[num_of_results-1]
    return list(zip(latest_urls,latest_snippets))