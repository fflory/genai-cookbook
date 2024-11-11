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
# MAGIC # Process Excel content
# MAGIC

# COMMAND ----------

import zipfile

# Path to the Excel file
file_path = "/Volumes/felixflory/ey_dbs_workshop_2024_10/raw_data/project_ad_astra/1.6.3.1 Ad Astra - Net assets details - 02.11.2020.xlsx"

def read_xls_files(file_path):
    file_path = file_path.replace("dbfs:", "")
    # Open the .xlsx file as a zip archive
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        # List all files in the archive (usually includes several XML files and other folders)
        xml_files = [file for file in zip_ref.namelist() if file.endswith('.xml')]
        
        # Print names of XML files for reference
        # print("XML files in the .xlsx archive:")
        # for xml_file in xml_files:
        #     print(xml_file)

        # Read the contents of each XML file
        xml_contents = {}
        for xml_file in xml_files:
            with zip_ref.open(xml_file) as file:
                # Decode and read the XML content
                xml_content = file.read().decode('utf-8')
                xml_contents[xml_file] = xml_content

    return xml_contents

xml_contents = read_xls_files(file_path)

xml_contents_flat = " ".join(f"{key}: {content[:500]}" for key, content in xml_contents.items())
len(xml_contents_flat)


# # Display the XML contents of each file (or you can use specific files if needed)
# for xml_file, content in xml_contents.items():
#     print(f"Contents of {xml_file}:")
#     print(content[:500])  # Print the first 500 characters for preview
#     print("\n---\n")


# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# Define a UDF to apply the read_xls_files function
@udf(StringType())
def read_xls_files_udf(file_path):
    return str(read_xls_files(file_path))

# Apply the UDF to the dataframe
excel_data_df = metadata_df.filter("file_extension == '.xlsx'")
excel_df_plus = excel_data_df.withColumn("xml_contents", read_xls_files_udf(excel_data_df.file_path))
display(excel_df_plus)

# COMMAND ----------

# MAGIC %md
# MAGIC # Process PDF content

# COMMAND ----------

def add_pdf_content():
    pdf_df = metadata_df.filter(F.col("file_extension") == F.lit(".xlsx"))

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

display(pdf_content_df)

# COMMAND ----------

display(pdf_content_df.select("file_name", "file_extension", "ccs", "file_type", "doc_parsed_contents"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Prompt Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC - **Be clear, specific, and direct**
# MAGIC   - “Respond with the following JSON schema encased in code blocks (```)”
# MAGIC - **Use examples (few shot learning)**
# MAGIC   - Add examples of desired inputs / outputs to your prompt
# MAGIC - **Chain-of-Thought**
# MAGIC   - Allow your model to “think”. 
# MAGIC   - Direct your model to output a summary of their results before generating the JSON structure.
# MAGIC - **Use the system prompt to give the LLM a specific role**
# MAGIC   - “You are an expert call transcript analyst working for AT&T”.
# MAGIC - **Instruct the LLM how to handle edge cases**
# MAGIC   - “If a label is not specifically addressed in the transcript summary, assign that label a False”.
# MAGIC - **Prefill responses**
# MAGIC   - If you want JSON output in code blocks, end your user prompt with ```, allowing the LLM to complete the JSON output
# MAGIC
# MAGIC https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview
# MAGIC

# COMMAND ----------

categories_string

# COMMAND ----------

prompt_template = f"""

You work for Ernst and Yong and are an expert in tagging and classifying documents. . 

Classify this document into {categories_string}.

Here are some examples of what you can expect to be asked:

Example 1: Metadata: Filename - "Draft Project Alpha Financial DD Report_05.18.2018"; CCS - "Buy & Integrate"; File extension - "pdf"; Text: "Reliance RestrictedDraft Financial due diligence reportProject Alpha..." Category: FDD Report  

Example 2: Metadata: Filename - "Draft Project Alpha Transaction Foundations Databook"; CCS - "Buy & Integrate"; File extension - "pdf"; Text: "Page 1Project Alpha..." Category: Databook  

Before responding with the final class output take a moment to caputure a few sentences of short notes which you can use to help accurately answer the above question. 

Now classify the following document, only provide the class, do not provide any reason or other context:

Metadata: Filename - %s; CCS - %s; File extension - %s, Text: %s

Rembember to only output the class, do not provide any reason or other context.
"""

# COMMAND ----------

prompt_template = f"""

You work for Ernst and Yong and are an expert in tagging and classifying documents. . 

Classify this document into {categories_string}.

Here are some examples of what you can expect to be asked:

Example 1: Metadata: Filename - "Draft Project Alpha Financial DD Report_05.18.2018"; CCS - "Buy & Integrate"; File extension - "pdf"; Text: "Reliance RestrictedDraft Financial due diligence reportProject Alpha..." Category: FDD Report  

Example 2: Metadata: Filename - "Draft Project Alpha Transaction Foundations Databook"; CCS - "Buy & Integrate"; File extension - "pdf"; Text: "Page 1Project Alpha..." Category: Databook  

Before responding with the final class output take a moment to caputure a few sentences of short notes which you can use to help accurately answer the above question. 

Now classify the following document, only provide the class, do not provide any reason or other context:

Metadata: Filename - %s, CCS - %s; File extension - %s, Text: %s

Rembember to only output the class, do not provide any reason or other context.
"""

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
    metadata_df_plus.withColumn(
        "query",
        F.format_string(
            prompt_template,
            F.col("file_name"),
            F.col("ccs"),
            F.col("file_extension"),
            F.expr("substring(xml_contents, 1, 1000)"),
        ),
    )
    .withColumn("class", F.expr(f"ai_query('{llm_endpoint}', query)"))
    # .limit(10)
)
classified_df.write.mode("overwrite").saveAsTable(
    f"{UC_CATALOG}.{UC_SCHEMA}.documents_classified_xlsx"
)

# COMMAND ----------

classified_df = spark.table(f"{UC_CATALOG}.{UC_SCHEMA}.documents_classified_xlsx")
is_match = F.lower(F.col("file_type")).eqNullSafe(F.lower(F.col("class"))).alias("is_match")
display(classified_df.select(
  is_match,
  F.col("file_type"), 
  F.col("class"), 
  F.col("file_name"),
  F.col("ccs"),
  F.col("file_extension"),
  # F.col("path")
  ))

# COMMAND ----------


