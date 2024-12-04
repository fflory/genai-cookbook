# Databricks notebook source
!pip install nltk==3.8.1

# COMMAND ----------

!pip install unstructured[all-docs]

# COMMAND ----------

# MAGIC %sh
# MAGIC apt install -y libmagic-dev poppler-utils

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /root/ntlk_data

# COMMAND ----------

from unstructured.partition.auto import partition

elements = partition("/Volumes/drewfurgiuele/default/pdfs/output/H3256_001_000_SOB_pg2.png")

# COMMAND ----------


