# Databricks notebook source
!pip install extractable

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import extractable as ex

input_file = "/Volumes/drewfurgiuele/default/pdfs/H3256_001_000_SOB.pdf"
output_dir = "/Volumes/drewfurgiuele/default/pdfs/output/extractable/"

# Extract tables from a PDF file
ex.extract(input_file=input_file, output_dir=output_dir, mode=ex.Mode.PRESENTATION)

# COMMAND ----------

# MAGIC %sh
# MAGIC cat /Volumes/drewfurgiuele/default/pdfs/output/extractable/table_1.xml

# COMMAND ----------

df = spark.read.option("rowTag", "tr").format("xml").load("/Volumes/drewfurgiuele/default/pdfs/output/extractable/table_1.xml")
df.printSchema()
df.show(truncate=False)

# COMMAND ----------

!pip install img2table

# COMMAND ----------


