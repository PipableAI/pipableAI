import csv
import pandas as pd
import PyPDF2


class _data_reader:
  # commented unused initializations
  # uncomment as use case arises
  def __init__(self):#,human_prompt="None"):
    super().__init__()
    #self.expert_id = "load_data"     #id for the embedder class
    #self.expert_context = ["Load different type of files given path to the file. Ex : Load the file /a/b/c.csv , load the file /a/b/c.parquet , load pickle from .pickle"]
    #self.human_prompt = human_prompt #the human input in natural language that started the action chain
    #self.actions = {"read_pdf":self.read_pdf,"read_parquet":self.read_parquet,"read_csv":self.read_csv}
    #self.action_context = ["Load the pdf file from the path /a/b/c.pdf as frame","Load the paruqet file from the path /a/b/c.parquet as frame","Load the csv file from the path /a/b/c.csv as frame"]
    #self.action_ids = ["read_pdf","read_parquet","read_csv"]

  @staticmethod
  def read_csv(path):
    try:
      return pd.read_csv(filepath_or_buffer=path,quoting = csv.QUOTE_MINIMAL,sep=None, delimiter=None, header='infer', index_col=None, usecols=None, dtype=None, engine="python", converters=None, true_values=[], false_values=[], skiprows=0,nrows=None, na_filter=True, verbose=False, skip_blank_lines=True,compression='infer', lineterminator=None, quotechar='"', doublequote=True, escapechar=None, encoding="utf-8", encoding_errors='ignore', on_bad_lines='skip', delim_whitespace=False)
    except Exception as e:
      return e

  @staticmethod
  def read_parquet(path):
    try:
      return pd.read_parquet(path=path, engine='auto', columns=None)
    except Exception as e:
      return e

  @staticmethod
  def read_pdf(path):
    try:
      reader = PyPDF2.PdfReader(stream = path, strict= False)
      data = [reader.pages[i].extract_text() for i in range(len(reader.pages))]
      return pd.DataFrame({"content":data})
    except Exception as e:
      return e