# Databricks notebook source
# Install the unstructured library
%pip install unstructured

# Import the necessary modules
from unstructured.partition.pdf import partition_pdf

# Define the path to your PDF file
pdf_path = "./ey_dbs_one_page_pdf.pdf"

# Parse the PDF and extract tables
elements = partition_pdf(pdf_path, infer_table_structure=True, strategy='hi_res')

# Display the extracted elements
for element in elements:
    print(element)

# COMMAND ----------

# MAGIC %sh
# MAGIC apt-get update
# MAGIC apt-get install -y poppler-utils

# COMMAND ----------

# MAGIC %pip install unstructured[local-inference] 

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install PyMuPDF

# COMMAND ----------

import fitz  # PyMuPDF
import shutil
import tempfile

# Create a temporary directory
temp_dir = tempfile.mkdtemp()

# Define the path to your PDF file
pdf_path = "/Volumes/yash_gupta_catalog/ey_dbs/pdf_files/Project Ad Astra_Report_2020-12-10_1155.pdf"

# Copy the PDF to the temporary directory
shutil.copy(pdf_path, temp_dir)

# Path to the copied PDF in the temporary directory
temp_pdf_path = f"{temp_dir}/Project Ad Astra_Report_2020-12-10_1155.pdf"

# Open the PDF
doc = fitz.open(temp_pdf_path)

# Example: Extract text from the first page
first_page_text = doc[0].get_text()

# Remember to close the document
doc.close()

# Now you can work with the text extracted from the PDF

# COMMAND ----------

first_page_text

# COMMAND ----------

import nltk

# Load 'punkt' tokenizer to confirm the path works
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
print("NLTK punkt tokenizer loaded successfully.")

# COMMAND ----------

# MAGIC %pip install -U -qqq nltk==3.8.1

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import nltk
nltk.download('punkt', download_dir='/local_disk0/nltk_data')

# COMMAND ----------

import nltk
nltk.data.path.append('/local_disk0/nltk_data')

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

def get_keywords(col):
    import nltk
    nltk.download('punkt', download_dir='/local_disk0/nltk_data')
    nltk.data.path.append('/local_disk0/nltk_data')
    sentences = nltk.sent_tokenize(col)
    return sentences

get_keywords_udf = udf(get_keywords, StringType())

# COMMAND ----------

# Proceed with the rest of your code
from unstructured.partition.pdf import partition_pdf

# Returns a List[Element] present in the pages of the parsed pdf document
elements = partition_pdf(
    "./ey_dbs_one_page_pdf.pdf",
    infer_table_structure=True,
    strategy='hi_res'
)

display(elements)

# COMMAND ----------

tables = [el for el in elements if el.category == "Table"]

# COMMAND ----------

print(tables[0].text)
print(tables[0].text)
