# Databricks notebook source
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

raw_pdf_elements = partition_pdf(
      filename= pdf_files, #f"/Volumes/{catalog}/{schema}/{volume}/{pdf_files[1]}",
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
      extract_image_block_output_dir=f"/Volumes/{catalog}/{schema}/{volume}/parsed_image_from_pdf/"
)
