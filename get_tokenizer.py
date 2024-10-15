# Databricks notebook source
# MAGIC %pip install transformers
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %sql
# MAGIC -- create catalog if not exists main;
# MAGIC USE CATALOG main;
# MAGIC CREATE schema if not exists main.huggingface_felix_flory;
# MAGIC use schema huggingface_felix_flory;
# MAGIC CREATE VOLUME if not exists hf_data;

# COMMAND ----------

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-large-en-v1.5")

# COMMAND ----------

# save to UC volume
tokenizer.save_pretrained(
    "/Volumes/felixflory/cookbook_felix_october/tokenizer/gte-large-en-v1.5")

# COMMAND ----------

# MAGIC %sh
# MAGIC #!/bin/bash
# MAGIC
# MAGIC # Create tar file
# MAGIC tar cvfz /Volumes/felixflory/cookbook_felix_october/tars/gte-large-en-v1.5.tar.gz /Volumes/felixflory/cookbook_felix_october/tokenizer/gte-large-en-v1.5

# COMMAND ----------

from transformers import AutoTokenizer

# load from UC volume
tokenizer = AutoTokenizer.from_pretrained("/Volumes/felixflory/cookbook_felix_october/tokenizer/gte-large-en-v1.5")

# COMMAND ----------

tokens = tokenizer.tokenize("The quick red fox jumped over the lazy brown dog.")
print(tokens)

# COMMAND ----------

token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)

# COMMAND ----------

encoded_tokens = tokenizer.encode("The quick red fox jumped over the lazy brown dog.", add_special_tokens=True)
print(encoded_tokens)
