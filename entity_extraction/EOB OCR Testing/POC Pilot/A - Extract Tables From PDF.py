# Databricks notebook source
# MAGIC %md
# MAGIC # Extracting Table Contents From PDF Documents
# MAGIC
# MAGIC This notebook is designed to take a PDF document, or series of PDF documents, and use a popular open source Python library, Extractable (https://github.com/SuleyNL/Extractable) to identity the regions of each page the contain a tabular chart. The library also makes attempts at identifying row and column layouts as well, but with more complicated charts it may not correctly identify them. 
# MAGIC
# MAGIC The goal of this library is instead rely on it's ability to quite accurately extract table regions from each page of a PDF, and create images of them. The library can also provide output files that numerically identify each chart on each page too, if you want to extract the images youreself.

# COMMAND ----------

!pip install extractable

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import extractable as ex

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC After we install our library on the cluster, we'll provide on PDF and specify where we want our output files saved. One important note here: these output files are actually metadata; files that contain the coordinates on each page of a PDF. The library will also extract the images themeslves to a temporary location, which we'll use in a later step:

# COMMAND ----------

input_file = "/Volumes/drewfurgiuele/default/pdfs/H3256_001_000_SOB.pdf"
output_dir = "/Volumes/drewfurgiuele/default/pdfs/output/extractable/"

# Extract tables from a PDF file
tables = ex.extract(input_file=input_file, output_dir=output_dir, mode=ex.Mode.PRESENTATION, output_filetype=ex.Filetype.PDF)

# COMMAND ----------

# MAGIC %md
# MAGIC After our process finishes, the object returned by the library contains many different properties. We'll use the temporary folder to then copy the images to a Databricks Volume that we can use later:

# COMMAND ----------

tables.data

# COMMAND ----------

tables.data["pdf_images"]

# COMMAND ----------

tables.temp_dir

# COMMAND ----------

# MAGIC %sh
# MAGIC cp /tmp/tmpkbtiz9mm/*.jpg /Volumes/drewfurgiuele/default/pdfs/output/extractable/temp_images/

# COMMAND ----------

# MAGIC %md
# MAGIC We can also take the images and use a native `binaryFile` Spark DataFrame reader to write our image and image metadata to a Delta table:

# COMMAND ----------

table_images = spark.read.format('binaryFile').load('/Volumes/drewfurgiuele/default/pdfs/output/extractable/temp_images/')

# COMMAND ----------

display(table_images)

# COMMAND ----------

table_images.write.saveAsTable("drewfurgiuele.default.pdf_images")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from drewfurgiuele.default.pdf_images
