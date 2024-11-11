# Databricks notebook source
# MAGIC %md
# MAGIC # Parse & Extract from Multiple PDFs
# MAGIC
# MAGIC Example: [NVDA Form-4](https://d18rn0p25nwr6d.cloudfront.net/CIK-0001045810/1494210c-49d2-4108-95bc-9a18e0de9ae5.pdf)

# COMMAND ----------

# MAGIC %pip install transformers==4.30.2 "unstructured[pdf,docx]==0.10.30" langchain==0.1.5 llama-index==0.9.3 --quiet
# MAGIC %pip install databricks-vectorsearch==0.22 pydantic==1.10.9 mlflow==2.10.1 --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %sh sudo rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* && sudo apt-get clean && sudo apt-get update && sudo apt-get install poppler-utils tesseract-ocr -y

# COMMAND ----------

display(dbutils.fs.ls("/Volumes/davidhuang_test/pdf_parsing/sec_pdf_docs"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract Texts & Tables with Unstructured.IO

# COMMAND ----------

from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from pyspark.sql.types import StructType, StructField, StringType
import glob
import re
import requests
import io

# COMMAND ----------

# grab form-4 pdfs
directory = glob.glob('/Volumes/davidhuang_test/pdf_parsing/sec_pdf_docs/nvidia-form-4-*.pdf')
directory

# COMMAND ----------

# partition each using hi-res
data = []
for file_name in directory:
    print("parsing:", file_name)
    sections = partition_pdf(file_name, strategy="hi_res", infer_table_structure=True)
    pdf_txt = ''
    for section in sections:
        if(section.metadata is not None):
            if(section.metadata.text_as_html is not None):
                pdf_txt += section.metadata.text_as_html + '\n'
            else:
                pdf_txt += section.text + '\n'
    data.append({"file_name": file_name, "pdf_content": pdf_txt})

# COMMAND ----------

# create dataframe
df = spark.createDataFrame(data)
display(df)

# COMMAND ----------

# show in HTML
from IPython.display import HTML

html_code = """<table><thead><th rowspan="2">L Title of Security (Instr. 3)</th><th rowspan="2">2. Trans. Date</th><th rowspan="2">|2A. Deemed Execution Date, if any</th><th>3. Trans. | (Instr. 8)</th><th>Code</th><th colspan="3">|4. Securities Acquired (A) Disposed of (D) (Instr. 3, 4 and 5)</th><th colspan="2" rowspan="2">or]5. Amount of Securities Beneficially Owned Following Reported Transaction(s) (instr, 3 and 4)</th><th rowspan="2">|6. Ownership Form: Direct (D) lor Indirect (1) (Instr. 4)</th><th rowspan="2">7. Nature of | Indirect Beneficial |Ownership |(Instr. 4)</th></thead><thead><th></th><th></th><th></th><th>Code</th><th>| V |</th><th>Amount]</th><th>(A) or (D) |</th><th>Price</th><th colspan="2"></th><th></th><th></th></thead><tr><td>Common Stock</td><td>3/19/2024</td><td></td><td>yi.</td><td>Vv</td><td>1,200}</td><td>D</td><td>$0 (.</td><td></td><td>2,968,428</td><td>I</td><td>The Lori Lym Huang 2016 Annuity Trus Il Agreement</td></tr><tr><td>Common Stock</td><td>3/19/2024</td><td></td><td>gy |</td><td>ov}</td><td>1200]</td><td></td><td>$02)</td><td></td><td>2,968,428</td><td>| 1</td><td>The Jen-Hsun [Huang 2016 Annuity Trust Il Agreement</td></tr><tr><td>Common Stock</td><td>3/19/2024</td><td></td><td>ym</td><td>[ov]</td><td>2a00f</td><td>a 7</td><td>go</td><td></td><td>60,483,228</td><td>| 1</td><td>ust 2</td></tr><tr><td>Common Stock</td><td>3/20/2024</td><td></td><td>F</td><td></td><td>74,9952)</td><td>Dd</td><td>[9903.72</td><td></td><td>3,147,883</td><td>D</td><td></td></tr><tr><td>Common Stock</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>4,048,956</td><td>| I</td><td>By y a Partnership</td></tr><tr><td>Common Stock</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>2,228,000</td><td>| 1</td><td>By [irrevocable Trust!</td></tr><tr><td>. Common Stock</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>. 5,007,800}</td><td>1</td><td>By Irrevocable | Remainder Trust {2</td></tr></table>"""

HTML(html_code)

# COMMAND ----------

# save to catalog
df.write.mode("overwrite").saveAsTable("davidhuang_test.pdf_parsing.form_4_parsed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract Info from Dataframe

# COMMAND ----------

# get chat model
from langchain_community.chat_models import ChatDatabricks
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", temperature=0)

# COMMAND ----------

# set system template
SYS_TEMPLATE = """You are a helpful assistant that can read and understand SEC Form-4 (Statement of Change of Benefitial Ownership of Securities). You will be given information below as CONTEXT, and you will be asked to EXTRACT INFORMATION. Your response MUST be in correct JSON format, with no explanations. 

Following this structure: "Name and Address of Reporitng Person": <your response>, "Issuer Name and Ticker or Trading Symbol": <your response>, "Date of Earliest Transaction (MM/DD/YYYY)": <your response>, "Relationship of Reporting Person(s) to Issuer": <your response>, "Title of Security": [<your response>], "Trans. Date": [<your response>], "Trans. Code": [<your response>], "Amount of Securities Aquired or Disposed": [<your response>],
"(A) or (D)": [<your response>], "Price": [<your response>], "Securities Aquired or Disposed, Amount of Securities Beneficially Owned Following Reported Transactions": [<your response>], "Ownership Form: Direct (D) or Indirect (I): [<your response>], "Nature of Indirect Beneficial Ownership": [<your response>]

CONTEXT: {context} 

ANSWER:
"""

# COMMAND ----------

# function use template to call LLM
def extract_with_llm(context, template):
    return chat_model.invoke(template.format(context=context)).content

# COMMAND ----------

from pyspark.sql.functions import lit
import pandas as pd

# Convert the Spark DataFrame column to a pandas DataFrame column
pdf_content = df.select("pdf_content").toPandas()["pdf_content"]

llm_output = [extract_with_llm(x, SYS_TEMPLATE) for x in pdf_content]
llm_output_cleaned = [x.replace("```json", "").replace("```", "").replace("\n", "") for x in llm_output]

print(llm_output_cleaned[0])

# COMMAND ----------

import json

json.loads(llm_output_cleaned[0])

# COMMAND ----------

llm_output_json = [json.loads(x) for x in llm_output_cleaned]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Output Results to Dataframe

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, FloatType, BooleanType

# Define the schema for the struct
schema = StructType([
    StructField("Name and Address of Reporting Person", StringType()),
    StructField("Issuer Name and Ticker or Trading Symbol", StringType()),
    StructField("Date of Earliest Transaction (MM/DD/YYYY)", StringType()),
    StructField("Relationship of Reporting Person(s) to Issuer", StringType()),
    StructField("Title of Security", ArrayType(StringType())),
    StructField("Trans. Date", ArrayType(StringType())),
    StructField("Trans. Code", ArrayType(StringType())),
    StructField("Amount of Securities Aquired or Disposed", ArrayType(StringType())),
    StructField("(A) or (D)", ArrayType(StringType())),
    StructField("Price", ArrayType(StringType())),
    StructField("Securities Aquired or Disposed, Amount of Securities Beneficially Owned Following Reported Transactions", ArrayType(StringType())),
    StructField("Ownership Form: Direct (D) or Indirect (I)", ArrayType(StringType())),
    StructField("Nature of Indirect Beneficial Ownership", ArrayType(StringType()))
])

new_df = spark.createDataFrame(llm_output_json, schema)

display(new_df)

# COMMAND ----------

new_column_headers = [
    "name_and_address_of_reporting_person",
    "issuer_name_and_ticker_or_trading_symbol",
    "date_of_earliest_transaction",
    "relationship_of_reporting_person_to_issuer",
    "title_of_security",
    "trans_date",
    "trans_code",
    "amount_of_securities_aquired_or_disposed",
    "acquired_or_disposed",
    "price",
    "amount_of_securities_beneficially_owned_following_reported_transactions",
    "ownership_direct_or_indirect",
    "nature_of_indirect_beneficial_ownership",
]

# COMMAND ----------

(
    new_df.toDF(*new_column_headers)
    .write.mode("overwrite")
    .saveAsTable("davidhuang_test.pdf_parsing.form_4_extracted")
)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC select * from davidhuang_test.pdf_parsing.form_4_extracted 

# COMMAND ----------


