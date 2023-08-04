import openai
import pandas as pd
import psycopg2
from pipable_utils import PIPABLE_LOGGER_CREATE
    
PIPABLE_LOG = PIPABLE_LOGGER_CREATE("PIP_POSTGRES", "DEBUG")

class _postgres_search():
  def __init__(self, openai_key = "", openai = None, file_path = {}, cursor = None, conn = None, db = None, agent = None):
    super().__init__()
    self.openai_key = openai_key
    self.PGname = file_path["pgdata"]
    self.PGhost = file_path["pghost"]
    self.PGuser = file_path["pguser"]
    self.PGpass = file_path["pgpass"]
    self.PGport = file_path["pgport"]
    self.PGsche = file_path["pgsche"]
    self.cur = cursor
    self.conn = conn
    self.pg_schema = ""
    # pgtables is a new config parameter introduced in the yaml file to scope down
    # the set of tables for this postgress expert to work with - instead of the whole schema
    # Example of how to describe this in the config taml file:
    # pgtables: alyf_members_provider_0epgz_view,alyf_asvs_provider_0epgz_view
    self.pgtables = file_path.get("pgtables", None)
    PIPABLE_LOG.info(f"Successfully initalized with {self.PGname}/{self.PGsche}")

  def autoschema(self):
    def generate_query(schema_name, table_names):
        table_names_str = ', '.join([f"'{table}'" for table in table_names]) if table_names else None
        table_name_clause = f"AND table_name IN ({table_names_str}) " if table_names else ''
        query = f"SELECT table_name, column_name, data_type " \
                f"FROM information_schema.columns " \
                f"WHERE table_schema = '{schema_name}' {table_name_clause}" \
                f"ORDER BY table_name;"
        return query
    query = generate_query(self.PGsche, self.pgtables.split(",") if self.pgtables else None)
    self.cur.execute(query)
    rows = self.cur.fetchall()
    self.pg_schema = '''
    Schema in this format: [(table_name, column_name, data_type),...]:\n
    '''
    PIPABLE_LOG.info(f"Schema retrieved - with {len(rows)} rows")
    self.pg_schema += str(rows)

  def initialize(self):
    self.conn = psycopg2.connect(
      user=self.PGuser,
      password=self.PGpass,
      host=self.PGhost,
      port=self.PGport,
      database=self.PGname,
      options=f"-c search_path={self.PGsche}"
    )
    self.cur = self.conn.cursor()
    self.autoschema()
    return self

  def search_data(self, query, context = ""):
    table_names = self.pgtables.split(",") if self.pgtables else None
    table_names = table_names_str = ', '.join([f"'{table}'" for table in table_names]) if table_names else None
    table_names_clause = f"Table Names: {table_names}\n" if table_names else ""
    prompt = (
        "Translate the following plain English query into a PostgreSQL query - with the following constraints:\n"
        + "(a) if you have a 'ORDER BY' clause in the query anywhere, make sure to include that column in the 'SELECT' clause. Otherwise the query will fail\n"
        + "(b) give me just the query and absolutely no comments and other stuff\n" 
        + f"Query: {query}\n"
        + table_names_clause
        + "Schema: \n"
        + self.pg_schema
    )
    print(prompt)
    openai.api_key =self.openai_key
    completion = openai.ChatCompletion.create(
      temperature=0.5,
      model="gpt-3.5-turbo",
      messages=[{"role": "user","content":prompt }]
    )
    obj = completion.choices[0].message.content
    PIPABLE_LOG.debug(f"Data-search query created '{' '.join(obj.splitlines())}' for query '{query}'")

    try:
      self.cur.execute(obj)
      df = pd.DataFrame(self.cur.fetchall())
      df.columns = [desc[0] for desc in self.cur.description]
      PIPABLE_LOG.info(f"Query run successful - dataframe with {len(df)} entries returned")
      # log success
      current_log = pd.DataFrame({
        "timestamp": [pd.Timestamp.now()],
        "query": [query],
        "datatype": ["postgres"],
        "database": [self.PGsche],
        "pipableAI": [""],
        "openAI": [obj],
        "success": [True],
        "error": [""]
      })
      temp = pd.read_parquet("logs.parquet", engine = 'pyarrow')
      pd.concat([temp, current_log], ignore_index = True).to_parquet("logs.parquet", engine = 'pyarrow')
      return (0, df)
    except Exception as e:
        PIPABLE_LOG.error(f"Query run failed with exception {repr(e)}....check log for more details")
        current_log = pd.DataFrame({
            "timestamp": [pd.Timestamp.now()],
            "query": [query],
            "datatype": ["postgres"],
            "database": [self.PGsche],
            "pipableAI": [""],
            "openAI": [obj],
            "success": [False],
            "error": [str(e)]
          })
        temp = pd.read_parquet("logs.parquet", engine = 'pyarrow')
        pd.concat([temp, current_log], ignore_index = True).to_parquet("logs.parquet", engine = 'pyarrow')
        return (1, str(e))