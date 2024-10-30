# Databricks notebook source
# MAGIC %run ../rag_app_sample_code/00_global_config

# COMMAND ----------

import pyspark.sql.functions as F
import pandas as pd
import re

# COMMAND ----------

class SampleFiles:
    def __init__(self, uc_path, sample_data):
        self.uc_path = uc_path
        self.sample_data = sample_data
        self.files = self.list_files(uc_path)
        # Convert the list of file paths into a pandas DataFrame
        self.files_df = pd.DataFrame(self.files, columns=["file_path"])
        self.files_df["file_name"] = self.files_df["file_path"].apply(
            lambda x: self.get_file_names_without_extension([x])[0]
        )

        # get the EY provided sample data
        self.use_case_sample_data_files = pd.read_csv(self.sample_data)
        self.use_case_sample_data_files.columns = (
            self.use_case_sample_data_files.columns.str.replace(
                r"[@#()$ -]", "_", regex=True
            ).str.lower()
        )

        # Convert the Pandas DataFrames to Spark DataFrames
        self.files_df_spark = spark.createDataFrame(self.files_df)
        self.use_case_sample_data_files_spark = spark.createDataFrame(
            self.use_case_sample_data_files
        )

        # Join the Spark DataFrames on the file_name column
        self.joined_df = self.files_df_spark.join(
            self.use_case_sample_data_files_spark,
            self.files_df_spark.file_name
            == self.use_case_sample_data_files_spark.file_name,
        ).drop(self.files_df_spark.file_name)

    # extract the file_name to match to the samples
    def get_file_names_without_extension(self, files):
        file_names = [f.split("/")[-1] for f in files]
        file_names_without_extension = [
            re.sub(r"\.(xlsx|pdf|docx|pptx)$", "", f) for f in file_names
        ]
        return file_names_without_extension

    def list_files(self, uc_path):
        files = []
        # List all files and directories at the current path
        for file_info in dbutils.fs.ls(uc_path):
            if file_info.isDir():
                # Recursively list files in subdirectories
                files.extend(self.list_files(file_info.path))
            else:
                # Add file path to the list
                files.append(file_info.path)
        return files

# COMMAND ----------



# COMMAND ----------

sample_files = SampleFiles(
    uc_path = SOURCE_PATH,
    sample_data = "/".join(SOURCE_PATH.split("/")[:-1])
    + "/info/"
    + "Databricks Use-cases_ver0.1.xlsx - Copy of Sample Data.csv",
)
sample_files.joined_df.write.format("delta").mode("overwrite").saveAsTable(f"{UC_CATALOG}.{UC_SCHEMA}.sample_files_delta_table")
sample_files_df = spark.table(f"{UC_CATALOG}.{UC_SCHEMA}.sample_files_delta_table")
display(sample_files_df)

# COMMAND ----------

file_types = [r.file_type for r in sample_files_df.select("file_type").distinct().collect()]

# COMMAND ----------

# categories = [
#     "Engagement Agreement/SoW",
#     "Databook",
#     "Red Flag Report",
#     "RFI",
#     "Information Request List",
#     "FDD Report",
#     "Strategy Document",
#     "Growth Analysis",
#     "Questionnaire",
#     "Net Assets",
#     "PL",
#     "Company overview",
#     "Census",
#     "AR Aging",
#     "Trial Balance",
#     "Opportunity Overview",
# ]

# COMMAND ----------

categories_string = ", ".join(file_types[:-1]) + ", or "+ file_types[-1]
print(categories_string)

# COMMAND ----------

_pdf_df = sample_files_df.filter(F.col("file_extension") == F.lit(".pdf"))


# COMMAND ----------

display(_pdf_df)

# COMMAND ----------

lt = spark.table(f"{UC_CATALOG}.{UC_SCHEMA}.ey_dbs_app_poc_parsed_docs_silver") 
parsed_slvr_sample_info = lt.join(_pdf_df, lt["path"] == _pdf_df["file_path"], "inner")
parsed_slvr_sample_info.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(f"{UC_CATALOG}.{UC_SCHEMA}.ey_dbs_app_poc_parsed_docs_silver_sample")
ey_dbs_app_poc_parsed_docs_silver_sample = spark.table(f"{UC_CATALOG}.{UC_SCHEMA}.ey_dbs_app_poc_parsed_docs_silver_sample")
display(ey_dbs_app_poc_parsed_docs_silver_sample)

# COMMAND ----------

# ; CCS - %s; File extension - %s; Text: %s

# COMMAND ----------

prompt_template = f"""
Classify this document into {categories_string}.

Example 1: Metadata: Filename - "Draft Project Alpha Financial DD Report_05.18.2018"; CCS - "Buy & Integrate"; File extension - "pdf"; Text: "Reliance RestrictedDraft Financial due diligence reportProject Alpha..." Category: FDD Report  

Example 2: Metadata: Filename - "Draft Project Alpha Transaction Foundations Databook"; CCS - "Buy & Integrate"; File extension - "pdf"; Text: "Page 1Project Alpha..." Category: Databook  

Now classify the following document, only provide the class, do not provide any reason or other context:

Metadata: Filename - %s; CCS - %s; File extension - %s, Text: %s"""

# COMMAND ----------

# prompt_template = f"""
# Classify this document into {categories_string}.

# Example 1: Metadata: Filename - "Draft Project Alpha Financial DD Report_05.18.2018"; CCS - "Buy & Integrate"; File extension - "pdf"; Text: "Reliance RestrictedDraft Financial due diligence reportProject Alpha 18 May 2018 ..." Category: FDD Report  

# Example 2: Metadata: Filename - "Draft Project Alpha Transaction Foundations Databook"; CCS - "Buy & Integrate"; File extension - "pdf"; Text: "Page 1Project Alpha Category: Databook  

# Now classify the following document: Metadata: Filename - "Draft Project Alpha Transaction Foundations Databook"; CCS - "Buy & Integrate"; File extension - "pdf"; Text: 
# """
# print(prompt)

# COMMAND ----------

llm_endpoint = "yosegi_fine_tuned_dbsupport"
llm_endpoint = "databricks-meta-llama-3-1-405b-instruct"
llm_endpoint = 'databricks-meta-llama-3-1-70b-instruct'
llm_endpoint = "Yash_GPT"

# COMMAND ----------

llama_405_df = (
  ey_dbs_app_poc_parsed_docs_silver_sample.withColumn(
    "query",
    F.format_string(
        prompt_template,
        F.col("file_name"),
        F.col("ccs"),
        F.col("file_extension"),
        F.expr("substring(doc_parsed_contents.parsed_content, 1, 1000)"),
    ),
)
.withColumn(
    "class", F.expr(f"ai_query('{llm_endpoint}', query)")
)
.limit(10))

# COMMAND ----------

llama_70_df = (
  ey_dbs_app_poc_parsed_docs_silver_sample.withColumn(
    "query",
    F.format_string(
        prompt_template,
        F.col("file_name"),
        F.col("ccs"),
        F.col("file_extension"),
        F.expr("substring(doc_parsed_contents.parsed_content, 1, 1000)"),
    ),
)
.withColumn(
    "class", F.expr(f"ai_query('{llm_endpoint}', query)")
)
.limit(10))

# COMMAND ----------

display(llama_70_df)

# COMMAND ----------

llm_endpoint = "Yash_GPT"
yash_GPT = (
    ey_dbs_app_poc_parsed_docs_silver_sample.withColumn(
        "query",
        F.format_string(
            prompt_template,
            F.col("file_name"),
            F.col("ccs"),
            F.col("file_extension"),
            F.expr("substring(doc_parsed_contents.parsed_content, 1, 1000)"),
        ),
    )
    .withColumn(
        "class", F.expr(f"ai_query('{llm_endpoint}', query)")
    )
    .limit(12)
)
display(yash_GPT)

# COMMAND ----------

# ai_query_sql = f"""
# SELECT 
#   *,
#   ai_query(
#     'databricks-meta-llama-3-1-405b-instruct',
#     CONCAT('{prompt}', doc_parsed_contents.parsed_content)
#   ) AS class
# FROM {UC_CATALOG}.{UC_SCHEMA}.ey_dbs_app_poc_parsed_docs_silver_sample
# limit 4
# """
# print(ai_query_sql)
