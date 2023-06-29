from pprint import pprint
from IPython.display import HTML
import pandas as pd
import numpy as np

class aggregated_stats():
  def __init__(self,frame="None"):
    super().__init__()
    self.frame = frame

  def initialize(self,path):
    self.frame = pd.read_csv(path,on_bad_lines = 'skip',sep = ",",skip_blank_lines = True,encoding = 'utf8',encoding_errors = "ignore")
    return self

  def get_stats(self,query):
    frame = self.frame
    pprint(display(HTML('<span style="color:#DB4437">pipable : Computing aggregated stats [25 percentile , 50 percentile , 75 percentile] </span>')))
    des=frame.describe(percentiles=[0.25,0.5,0.75,0.95], include=np.number)
    des
    return des

  def get_corr(self,query):
    frame=self.frame
    pprint(display(HTML('<span style="color:#DB4437">pipable : Checking correlation between features </span>')))
    corr=frame.corr(method='pearson', min_periods=1000, numeric_only=True)
    corr
    return corr