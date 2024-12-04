# Databricks notebook source
# MAGIC %sh
# MAGIC apt install -y ghostscript

# COMMAND ----------

!pip install PyPDF2==2.12.1

# COMMAND ----------

!pip install "camelot-py[base]"

# COMMAND ----------

!pip install opencv-python

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import camelot
tables = camelot.read_pdf('/Volumes/drewfurgiuele/default/pdfs/H3256_001_000_SOB.pdf', flavor="stream", pages='2', split_text=True)
tables

# COMMAND ----------

camelot.plot(tables[1], kind='contour').show()

# COMMAND ----------

camelot.plot(tables[1], kind='grid').show()

# COMMAND ----------

camelot.plot(tables[1], kind='textedge').show()

# COMMAND ----------

tables[1]
tables[1].parsing_report
{
    'accuracy': 99.02,
    'whitespace': 12.24,
    'order': 1,
    'page': 1
}

# COMMAND ----------

tables[1].df

# COMMAND ----------


