import re
from html import unescape

from googleapiclient.discovery import build
from IPython.display import HTML, display


class _google_search():
  def __init__(self,_google_search=None,frame=None,
               past_query=[],current_query=None,current_input=None,
               action=[],current_action=None,
               output=[],current_output=None,correct_output=[],programmable_search_engine_api_key="",past_results=[]):
    super().__init__()
    self.past_query=past_query
    self.current_query=current_query
    self.current_input=current_input
    self._google_search = _google_search
    self.correct_output=correct_output
    self.past_results = past_results
    self.programmable_search_engine_api_key = programmable_search_engine_api_key

  def initialise(self,google_api_key = "",search_engine_key=""):
    _google_search = build('customsearch', 'v1', developerKey=google_api_key)
    self.programmable_search_engine_api_key = search_engine_key
    self._google_search = _google_search
    return self
  
  def get_latest_google_search_results(self):
    num_of_results = len(self.past_snippets)
    if num_of_results == 0:
      raise Exception("No Results Found")
    latest_snippets = self.past_snippets[num_of_results-1]
    latest_urls = self.past_results_urls[num_of_results-1]
    return list(zip(latest_urls,latest_snippets))

  def ask_google(self,query):
    def clean_text(text):
      clean_text = re.sub('<.*?>', '', text)
      clean_text = unescape(clean_text)
      clean_text = re.sub('[^a-zA-Z0-9\s]', '', clean_text)
      clean_text = re.sub('\s+', ' ', clean_text).strip()
      return clean_text
    print("Google search: {}".format(query))
    gs = self._google_search
    response = gs.cse().list(q=query, cx=self.programmable_search_engine_api_key).execute()

    if int(response['searchInformation']['totalResults']) == 0:
      return (1, "No results returned.")

    ite=1
    output = []
    for item in response['items']:
      cleantext = clean_text(item['snippet'])
      print("\nLink {}: {}".format(ite,item['link']))
      print("Result {}: {}".format(ite,cleantext))
      ite+=1
      output.append({"url":item['link'],"headline":cleantext})
    self.past_results.append(output)
    return (0, output)

# returning 0 means success, 1 means error
# code to handle error is not written yet