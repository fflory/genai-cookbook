# Databricks notebook source
# MAGIC %pip install transformers==4.30.2 "unstructured[pdf,docx,databricks-volumes]" -U -qq
# MAGIC %pip install databricks-vectorsearch pydantic==1.10.9 lxml==4.9.3 databricks-agents -qqqq
# MAGIC %pip install mlflow-skinny mlflow mlflow[gateway] -U
# MAGIC %pip install langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4 llama-index==0.10.43 -qqqq
# MAGIC %pip install tiktoken outlines -q
# MAGIC %pip install openai tenacity -qq
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../entity_extraction/00-helpers

# COMMAND ----------

# MAGIC %run ../rag_app_sample_code/A_POC_app/pdf_uc_volume/00_config

# COMMAND ----------

install_ocr_on_nodes()

# COMMAND ----------

_sp = SOURCE_PATH+'/project_churches/Project Churches - FINAL Red Flag Report 181121.pdf'

# COMMAND ----------

pdf_files = [f.name for f in dbutils.fs.ls(_sp)]
print(pdf_files)

# COMMAND ----------

from unstructured.partition.pdf import partition_pdf
from databricks.sdk import WorkspaceClient
import pandas as pd

w = WorkspaceClient()

# COMMAND ----------

raw_pdf_elements = partition_pdf(
      filename= _sp, # pdf_files, #f"/Volumes/{catalog}/{schema}/{volume}/{pdf_files[1]}",
      #extract_images_in_pdf=True,
      infer_table_structure=True,
      lenguages=["eng"],
      strategy="hi_res",
      extract_image_block_types=["Table", "Image"],
      #extract_image_block_to_payload=False,
      # Chunking params to aggregate text blocks
      # Attempt to create a new chunk 3800 chars
      # Attempt to keep chunks > 2000 chars
      # Hard max on chunks                 
      # max_characters=4000,
      # new_after_n_chars=3800,
      # combine_text_under_n_chars=2000,
      # chunking_strategy="basic",
      #image_output_dir_path=f"/Volumes/{catalog}/{schema}/{volume}/parsed_images_from_pdf/",
      extract_image_block_output_dir=f"/Volumes/felixflory/ey_dbs_workshop_2024_10/raw_data/pdf_table_output/"
)

# COMMAND ----------

df_ex = pd.DataFrame([e.to_dict() for e in raw_pdf_elements])
df_ex['page_number'] = df_ex['metadata'].apply(lambda x: x['page_number'])
df_ex['parent_id'] = df_ex['metadata'].apply(lambda x: x.get("parent_id", None))
display(df_ex)

# COMMAND ----------

display(df_ex[df_ex['page_number'] == 7])
