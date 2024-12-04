# Databricks notebook source
from transformers import pipeline
from PIL import Image

# COMMAND ----------

source_pdfs = spark.read.table("drewfurgiuele.default.pdf_images").select("path","content").where("path = 'dbfs:/Volumes/drewfurgiuele/default/pdfs/output/extractable/temp_images/H3256_001_000_SOB_page_2.jpg'")

# COMMAND ----------

display(source_pdfs)

# COMMAND ----------

image = Image.open('/Volumes/drewfurgiuele/default/pdfs/output/extractable/temp_images/H3256_001_000_SOB_page_2.jpg')

# COMMAND ----------

image

# COMMAND ----------

#image = Image.open(requests.get(url, stream=True).raw)

doc_question_answerer = pipeline("document-question-answering", model="impira/layoutlm-document-qa")
preds = doc_question_answerer(
    question="What is the maximum out-of-pocket out-of-network cost?",
    image=image,
)
preds

# COMMAND ----------

doc_question_answerer = pipeline("document-question-answering", model="impira/layoutlm-document-qa")
preds = doc_question_answerer(
    question="What is the percent of co-insurance amount for an ambulance?",
    image=image,
)
preds

# COMMAND ----------


