# Databricks notebook source
# MAGIC %sql
# MAGIC use catalog felixflory;
# MAGIC use schema ey_dbs_workshop_2024_10;
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION _genie_query(databricks_host STRING, 
# MAGIC                   databricks_token STRING,
# MAGIC                   space_id STRING,
# MAGIC                   question STRING,
# MAGIC                   contextual_history STRING)
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT 'This is a agent that you can converse with to get answers to questions. Try to provide simple questions and provide history if you had prior conversations.'
# MAGIC AS
# MAGIC $$
# MAGIC     import json
# MAGIC     import os
# MAGIC     import time
# MAGIC     from dataclasses import dataclass
# MAGIC     from datetime import datetime
# MAGIC     from typing import Optional
# MAGIC     
# MAGIC     import pandas as pd
# MAGIC     import requests
# MAGIC     
# MAGIC     
# MAGIC     @dataclass
# MAGIC     class GenieResult:
# MAGIC         space_id: str
# MAGIC         conversation_id: str
# MAGIC         question: str
# MAGIC         content: Optional[str]
# MAGIC         sql_query: Optional[str] = None
# MAGIC         sql_query_description: Optional[str] = None
# MAGIC         sql_query_result: Optional[pd.DataFrame] = None
# MAGIC         error: Optional[str] = None
# MAGIC     
# MAGIC         def to_json_results(self):
# MAGIC             result = {
# MAGIC                 "space_id": self.space_id,
# MAGIC                 "conversation_id": self.conversation_id,
# MAGIC                 "question": self.question,
# MAGIC                 "content": self.content,
# MAGIC                 "sql_query": self.sql_query,
# MAGIC                 "sql_query_description": self.sql_query_description,
# MAGIC                 "sql_query_result": self.sql_query_result.to_dict(
# MAGIC                     orient="records") if self.sql_query_result is not None else None,
# MAGIC                 "error": self.error,
# MAGIC             }
# MAGIC             jsonified_results = json.dumps(result)
# MAGIC             return f"Genie Results are: {jsonified_results}"
# MAGIC     
# MAGIC         def to_string_results(self):
# MAGIC             results_string = self.sql_query_result.to_dict(orient="records") if self.sql_query_result is not None else None
# MAGIC             return ("Genie Results are: \n"
# MAGIC                     f"Space ID: {self.space_id}\n"
# MAGIC                     f"Conversation ID: {self.conversation_id}\n"
# MAGIC                     f"Question That Was Asked: {self.question}\n"
# MAGIC                     f"Content: {self.content}\n"
# MAGIC                     f"SQL Query: {self.sql_query}\n"
# MAGIC                     f"SQL Query Description: {self.sql_query_description}\n"
# MAGIC                     f"SQL Query Result: {results_string}\n"
# MAGIC                     f"Error: {self.error}")
# MAGIC     
# MAGIC     class GenieClient:
# MAGIC     
# MAGIC         def __init__(self, *,
# MAGIC                      host: Optional[str] = None,
# MAGIC                      token: Optional[str] = None,
# MAGIC                      api_prefix: str = "/api/2.0/genie/spaces"):
# MAGIC             self.host = host or os.environ.get("DATABRICKS_HOST")
# MAGIC             self.token = token or os.environ.get("DATABRICKS_TOKEN")
# MAGIC             assert self.host is not None, "DATABRICKS_HOST is not set"
# MAGIC             assert self.token is not None, "DATABRICKS_TOKEN is not set"
# MAGIC             self._workspace_client = requests.Session()
# MAGIC             self._workspace_client.headers.update({"Authorization": f"Bearer {self.token}"})
# MAGIC             self._workspace_client.headers.update({"Content-Type": "application/json"})
# MAGIC             self.api_prefix = api_prefix
# MAGIC             self.max_retries = 300
# MAGIC             self.retry_delay = 1
# MAGIC             self.new_line = "\r\n"
# MAGIC     
# MAGIC         def _make_url(self, path):
# MAGIC             return f"{self.host.rstrip('/')}/{path.lstrip('/')}"
# MAGIC     
# MAGIC         def start(self, space_id: str, start_suffix: str = "") -> str:
# MAGIC             path = self._make_url(f"{self.api_prefix}/{space_id}/start-conversation")
# MAGIC             resp = self._workspace_client.post(
# MAGIC                 url=path,
# MAGIC                 headers={"Content-Type": "application/json"},
# MAGIC                 json={"content": "starting conversation" if not start_suffix else f"starting conversation {start_suffix}"},
# MAGIC             )
# MAGIC             resp = resp.json()
# MAGIC             print(resp)
# MAGIC             return resp["conversation_id"]
# MAGIC     
# MAGIC         def ask(self, space_id: str, conversation_id: str, message: str) -> GenieResult:
# MAGIC             path = self._make_url(f"{self.api_prefix}/{space_id}/conversations/{conversation_id}/messages")
# MAGIC             # TODO: cleanup into a separate state machine
# MAGIC             resp_raw = self._workspace_client.post(
# MAGIC                 url=path,
# MAGIC                 headers={"Content-Type": "application/json"},
# MAGIC                 json={"content": message},
# MAGIC             )
# MAGIC             resp = resp_raw.json()
# MAGIC             message_id = resp.get("message_id", resp.get("id"))
# MAGIC             if message_id is None:
# MAGIC                 print(resp, resp_raw.url, resp_raw.status_code, resp_raw.headers)
# MAGIC                 return GenieResult(content=None, error="Failed to get message_id")
# MAGIC     
# MAGIC             attempt = 0
# MAGIC             query = None
# MAGIC             query_description = None
# MAGIC             content = None
# MAGIC     
# MAGIC             while attempt < self.max_retries:
# MAGIC                 resp_raw = self._workspace_client.get(
# MAGIC                     self._make_url(f"{self.api_prefix}/{space_id}/conversations/{conversation_id}/messages/{message_id}"),
# MAGIC                     headers={"Content-Type": "application/json"},
# MAGIC                 )
# MAGIC                 resp = resp_raw.json()
# MAGIC                 status = resp["status"]
# MAGIC                 if status == "COMPLETED":
# MAGIC                     try:
# MAGIC     
# MAGIC                         query = resp["attachments"][0]["query"]["query"]
# MAGIC                         query_description = resp["attachments"][0]["query"].get("description", None)
# MAGIC                         content = resp["attachments"][0].get("text", {}).get("content", None)
# MAGIC                     except Exception as e:
# MAGIC                         return GenieResult(
# MAGIC                             space_id=space_id,
# MAGIC                             conversation_id=conversation_id,
# MAGIC                             question=message,
# MAGIC                             content=resp["attachments"][0].get("text", {}).get("content", None)
# MAGIC                         )
# MAGIC                     break
# MAGIC     
# MAGIC                 elif status == "EXECUTING_QUERY":
# MAGIC                     self._workspace_client.get(
# MAGIC                         self._make_url(
# MAGIC                             f"{self.api_prefix}/{space_id}/conversations/{conversation_id}/messages/{message_id}/query-result"),
# MAGIC                         headers={"Content-Type": "application/json"},
# MAGIC                     )
# MAGIC                 elif status in ["FAILED", "CANCELED"]:
# MAGIC                     return GenieResult(
# MAGIC                         space_id=space_id,
# MAGIC                         conversation_id=conversation_id,
# MAGIC                         question=message,
# MAGIC                         content=None,
# MAGIC                         error=f"Query failed with status {status}"
# MAGIC                     )
# MAGIC                 elif status != "COMPLETED" and attempt < self.max_retries - 1:
# MAGIC                     time.sleep(self.retry_delay)
# MAGIC                 else:
# MAGIC                     return GenieResult(
# MAGIC                         space_id=space_id,
# MAGIC                         conversation_id=conversation_id,
# MAGIC                         question=message,
# MAGIC                         content=None,
# MAGIC                         error=f"Query failed or still running after {self.max_retries * self.retry_delay} seconds"
# MAGIC                     )
# MAGIC                 attempt += 1
# MAGIC             resp = self._workspace_client.get(
# MAGIC                 self._make_url(
# MAGIC                     f"{self.api_prefix}/{space_id}/conversations/{conversation_id}/messages/{message_id}/query-result"),
# MAGIC                 headers={"Content-Type": "application/json"},
# MAGIC             )
# MAGIC             resp = resp.json()
# MAGIC             columns = resp["statement_response"]["manifest"]["schema"]["columns"]
# MAGIC             header = [str(col["name"]) for col in columns]
# MAGIC             rows = []
# MAGIC             output = resp["statement_response"]["result"]
# MAGIC             if not output:
# MAGIC                 return GenieResult(
# MAGIC                     space_id=space_id,
# MAGIC                     conversation_id=conversation_id,
# MAGIC                     question=message,
# MAGIC                     content=content,
# MAGIC                     sql_query=query,
# MAGIC                     sql_query_description=query_description,
# MAGIC                     sql_query_result=pd.DataFrame([], columns=header),
# MAGIC                 )
# MAGIC             for item in resp["statement_response"]["result"]["data_typed_array"]:
# MAGIC                 row = []
# MAGIC                 for column, value in zip(columns, item["values"]):
# MAGIC                     type_name = column["type_name"]
# MAGIC                     str_value = value.get("str", None)
# MAGIC                     if str_value is None:
# MAGIC                         row.append(None)
# MAGIC                         continue
# MAGIC                     match type_name:
# MAGIC                         case "INT" | "LONG" | "SHORT" | "BYTE":
# MAGIC                             row.append(int(str_value))
# MAGIC                         case "FLOAT" | "DOUBLE" | "DECIMAL":
# MAGIC                             row.append(float(str_value))
# MAGIC                         case "BOOLEAN":
# MAGIC                             row.append(str_value.lower() == "true")
# MAGIC                         case "DATE":
# MAGIC                             row.append(datetime.strptime(str_value, "%Y-%m-%d").date())
# MAGIC                         case "TIMESTAMP":
# MAGIC                             row.append(datetime.strptime(str_value, "%Y-%m-%d %H:%M:%S"))
# MAGIC                         case "BINARY":
# MAGIC                             row.append(bytes(str_value, "utf-8"))
# MAGIC                         case _:
# MAGIC                             row.append(str_value)
# MAGIC                 rows.append(row)
# MAGIC     
# MAGIC             query_result = pd.DataFrame(rows, columns=header)
# MAGIC             return GenieResult(
# MAGIC                 space_id=space_id,
# MAGIC                 conversation_id=conversation_id,
# MAGIC                 question=message,
# MAGIC                 content=content,
# MAGIC                 sql_query=query,
# MAGIC                 sql_query_description=query_description,
# MAGIC                 sql_query_result=query_result,
# MAGIC             )
# MAGIC     
# MAGIC     
# MAGIC     assert databricks_host is not None, "host is not set"
# MAGIC     assert databricks_token is not None, "token is not set"
# MAGIC     assert space_id is not None, "space_id is not set"
# MAGIC     assert question is not None, "question is not set"
# MAGIC     assert contextual_history is not None, "contextual_history is not set"
# MAGIC     client = GenieClient(host=databricks_host, token=databricks_token)
# MAGIC     conversation_id = client.start(space_id)
# MAGIC     formatted_message = f"""Use the contextual history to answer the question. The history may or may not help you. Use it if you find it relevant.
# MAGIC     
# MAGIC     Contextual History: {contextual_history}
# MAGIC     
# MAGIC     Question to answer: {question}
# MAGIC     """
# MAGIC     
# MAGIC     result = client.ask(space_id, conversation_id, formatted_message)
# MAGIC     
# MAGIC     return result.to_string_results()
# MAGIC
# MAGIC $$;

# COMMAND ----------

"the question to ask about The data provided spans various aspects of financial reports, including:

The two spreadsheets contain the following information:

Trial balance P&L: This sheet contains detailed profit and loss (P&L) data, including revenue streams like sales from projects and services. The data is presented over multiple fiscal years (FY17, FY18, and FY19), with columns showing revenue from different accounts such as 'Sales revenues resulting from projects,' 'Sales revenues - other business,' and 'Revenue from lease/rental contracts.'
Effects from IFRS 16: Likely contains data about the impacts of IFRS 16 accounting standards, although this sheet wasn't fully previewed yet.

Trial balance Net assets This sheet includes details about the company's net assets, particularly related to fixed assets and financial assets, such as equity investments, shares in companies, and subsidiaries. It includes fiscal data for FY17, FY18, and FY19, showing asset values in euros.
Effects from IFRS 16 Similar to the second sheet in the first file, this likely contains information about how IFRS 16 impacts net assets.
These files appear to be part of financial reporting, specifically related to revenue streams, asset holdings, and the application of accounting standards. 

This data can be leveraged for a wide range of analyses, such as sentiment analysis, market trend analysis, customer satisfaction and engagement analysis, financial performance analysis, and user behavior analysis on digital platforms."

# COMMAND ----------

from mlflow.utils.databricks_utils import get_databricks_host_creds

# COMMAND ----------

creds = get_databricks_host_creds()
creds.token, creds.host

UC_CATALOG = "felixflory"
UC_SCHEMA = "ey_dbs_workshop_2024_10"
GENIE_SPACE = "01ef9217da36130ba9e15a0c0eeae810"

QUESTION_COMMENT = "the question to ask"
CONTEXTUAL_HISTORY_COMMENT = """
provide relavant history to be able to answer this question, assume genie doesnt 
keep track of history. Use 'no relevant history' if there is nothing relevant 
to answer the question."""
GENIE_FUNC_DESCRIPTION = """
This is an agent that you can converse with to get answers to questions about the customer Ad Astra. It can answer questions about net equity assets for fiscal year 2019 for the customer Ad Astra. Assets can be classified into different levels, such as level 1, level 2, level 3, and level 4. The agent can also provide information about the company's financial performance, such as year over year earnings, investment performance, and other metrics. Try to provide simple questions and provide history if you had prior conversations.

"""

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {UC_CATALOG}.{UC_SCHEMA}.chat_with_genie(
  question STRING COMMENT '{QUESTION_COMMENT}',
  contextual_history STRING COMMENT "{CONTEXTUAL_HISTORY_COMMENT}")
RETURNS STRING
LANGUAGE SQL
COMMENT "{GENIE_FUNC_DESCRIPTION}"
RETURN SELECT {UC_CATALOG}.{UC_SCHEMA}._genie_query(
  '{creds.host}',
  '{creds.token}',
  '{GENIE_SPACE}',
  question, -- retrieved from function
  contextual_history -- retrieved from function
)
""")

# COMMAND ----------

# %sql

# use catalog felixflory;
# use schema ey_dbs_workshop_2024_10;


# CREATE OR REPLACE FUNCTION chat_with_ad_astra_genie(question STRING COMMENT "the question to ask about customer reports",
#                   contextual_history STRING COMMENT "provide relavant history to be able to answer this question, assume genie doesnt keep track of history. Use 'no relevant history' if there is nothing relevant to answer the question.")
# RETURNS STRING
# LANGUAGE SQL
# COMMENT 'This is a agent that you can converse with to get answers to questions about customer reports. Try to provide simple questions and provide history if you had prior conversations.' 
# RETURN SELECT _genie_query(
#   'https://e2-demo-field-eng.cloud.databricks.com/',
#   secret('felix-flory', 'pat'),
#   '01ef9217da36130ba9e15a0c0eeae810',
#   question, -- retrieved from function
#   contextual_history -- retrieved from function
# );

# COMMAND ----------

# MAGIC %md
# MAGIC I had to run the following in a query editor

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION felixflory.ey_dbs_workshop_2024_10.ai_search(question STRING COMMENT "the question to ask about customer reports, their corporate earnings reports")
# MAGIC RETURNS STRING
# MAGIC LANGUAGE SQL
# MAGIC COMMENT 'This is a search agent for anything regarding company revenue, financial reports and coporate earnings' 
# MAGIC RETURN SELECT string(collect_set(chunked_text)) from vector_search(index => "felixflory.ey_dbs_workshop_2024_10.ey_dbs_app_poc_chunked_docs_gold_index", query => question, num_results => 5);
# MAGIC
# MAGIC select felixflory.ey_dbs_workshop_2024_10.ai_search("what whas Zorch revenue in 2019");

# COMMAND ----------


