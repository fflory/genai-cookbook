# Databricks notebook source
# MAGIC %run ../rag_app_sample_code/00_global_config

# COMMAND ----------

import pyspark.sql.functions as F
import pandas as pd
import re

# COMMAND ----------

# MAGIC %md
# MAGIC # Process file names and metadata

# COMMAND ----------

class SampleFiles:
    def __init__(self, uc_path, metadata_path):
        self.uc_path = uc_path
        self.metadata_path = metadata_path
        self.file_names = self.get_file_names(uc_path)
        # Convert the list of file paths into a pandas DataFrame
        self.file_names_pddf = pd.DataFrame(self.file_names, columns=["file_path"])
        self.file_names_pddf["file_name"] = self.file_names_pddf["file_path"].apply(
            lambda x: self.get_file_names_without_extension([x])[0]
        )

        # get the EY provided sample metadata
        self.metadata_pddf = pd.read_csv(self.metadata_path)
        self.metadata_pddf.columns = (
            self.metadata_pddf.columns.str.replace(
                r"[@#()$ -]", "_", regex=True
            ).str.lower()
        )

        # Convert the Pandas DataFrames to Spark DataFrames
        self.file_names_spdf = spark.createDataFrame(self.file_names_pddf)
        self.metadata_spdf   = spark.createDataFrame(self.metadata_pddf)

        # Join the Spark DataFrames on the file_name column
        self.output_spdf = self.file_names_spdf.join(
            self.metadata_spdf,
            self.file_names_spdf.file_name
            == self.metadata_spdf.file_name,
        ).drop(self.file_names_spdf.file_name)

    # extract the file_name to match to the samples
    def get_file_names_without_extension(self, files):
        file_names = [f.split("/")[-1] for f in files]
        file_names_without_extension = [
            re.sub(r"\.(xlsx|pdf|docx|pptx)$", "", f) for f in file_names
        ]
        return file_names_without_extension

    def get_file_names(self, uc_path):
        files = []
        # List all files and directories at the current path
        for file_info in dbutils.fs.ls(uc_path):
            if file_info.isDir():
                # Recursively list files in subdirectories
                files.extend(self.get_file_names(file_info.path))
            else:
                # Add file path to the list
                files.append(file_info.path)
        return files

# COMMAND ----------

metadata_path = (
  "/".join(SOURCE_PATH.split("/")[:-1])
  + "/info/"
  + "Databricks Use-cases_ver0.1.xlsx - Copy of Sample Data.csv")

# COMMAND ----------

sample_files = SampleFiles(
    uc_path=SOURCE_PATH,
    metadata_path=metadata_path,
)
sample_files.output_spdf.write.format("delta").mode("overwrite").saveAsTable(
    f"{UC_CATALOG}.{UC_SCHEMA}.sample_files_metadata"
)
metadata_df = spark.table(f"{UC_CATALOG}.{UC_SCHEMA}.sample_files_metadata")
display(metadata_df)

# COMMAND ----------

file_types = [r.file_type for r in metadata_df.select("file_type").distinct().collect()]
categories_string = ", ".join(file_types[:-1]) + ", or "+ file_types[-1]
print(categories_string)

# COMMAND ----------

# MAGIC %md
# MAGIC # Process PDF content

# COMMAND ----------

def add_pdf_content():
    pdf_df = metadata_df.filter(F.col("file_extension") == F.lit(".pdf"))

    raw_text = spark.table(
        f"{UC_CATALOG}.{UC_SCHEMA}.ey_dbs_app_poc_parsed_docs_silver"
    )
    pdf_content_df = raw_text.join(
        pdf_df, raw_text["path"] == pdf_df["file_path"], "inner"
    ).drop("file_path")
    pdf_content_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(
        f"{UC_CATALOG}.{UC_SCHEMA}.pdf_content_df"
    )
    pdf_content_df = spark.table(f"{UC_CATALOG}.{UC_SCHEMA}.pdf_content_df")
    return pdf_content_df


pdf_content_df = add_pdf_content()

# COMMAND ----------

display(pdf_content_df.select("file_name", "file_extension", "ccs", "file_type", "doc_parsed_contents"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Prompt Engineering

# COMMAND ----------

prompt_template = f"""
Classify this document into {categories_string}.

Example 1: Metadata: Filename - "Draft Project Alpha Financial DD Report_05.18.2018"; CCS - "Buy & Integrate"; File extension - "pdf"; Text: "Reliance RestrictedDraft Financial due diligence reportProject Alpha..." Category: FDD Report  

Example 2: Metadata: Filename - "Draft Project Alpha Transaction Foundations Databook"; CCS - "Buy & Integrate"; File extension - "pdf"; Text: "Page 1Project Alpha..." Category: Databook  

Now classify the following document, only provide the class, do not provide any reason or other context:

Metadata: Filename - %s; CCS - %s; File extension - %s, Text: %s"""

# COMMAND ----------

# llm_endpoint = "yosegi_fine_tuned_dbsupport"
# llm_endpoint = "databricks-meta-llama-3-1-405b-instruct"
# llm_endpoint = "Yash_GPT"
llm_endpoint = "databricks-meta-llama-3-1-70b-instruct"

# COMMAND ----------

# MAGIC %md
# MAGIC # LLM based Classification

# COMMAND ----------

classified_df = (
    pdf_content_df.withColumn(
        "query",
        F.format_string(
            prompt_template,
            F.col("file_name"),
            F.col("ccs"),
            F.col("file_extension"),
            F.expr("substring(doc_parsed_contents.parsed_content, 1, 1000)"),
        ),
    )
    .withColumn("class", F.expr(f"ai_query('{llm_endpoint}', query)"))
    # .limit(10)
)
classified_df.write.mode("overwrite").saveAsTable(
    f"{UC_CATALOG}.{UC_SCHEMA}.documents_classified"
)

# COMMAND ----------

classified_df = spark.table(f"{UC_CATALOG}.{UC_SCHEMA}.documents_classified")
is_match = F.lower(F.col("file_type")).eqNullSafe(F.lower(F.col("class"))).alias("is_match")
display(classified_df.select("file_type", "class", is_match, "path"))
