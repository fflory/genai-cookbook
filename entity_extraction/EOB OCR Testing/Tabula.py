# Databricks notebook source
!pip install -q tabula-py jpype1

# COMMAND ----------

!java -version

# COMMAND ----------

import tabula

# COMMAND ----------

tabula.environment_info()

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /local_disk0/.ephemeral_nfs/envs/pythonEnv-aee20a89-7c8a-49cc-be7a-f705eeaab226/lib/python3.10/site-packages/tabula

# COMMAND ----------

pdf_path = "/Volumes/drewfurgiuele/default/pdfs/H3256_001_000_SOB.pdf"
dfs = tabula.read_pdf(pdf_path, stream=True, pages='all')

# COMMAND ----------

dfs

# COMMAND ----------


