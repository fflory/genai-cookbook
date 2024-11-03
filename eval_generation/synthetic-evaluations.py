# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-vectorsearch databricks-sdk langchain==0.2.11 langchain_core==0.2.23 langchain_community==0.2.10 --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../rag_app_sample_code/00_global_config

# COMMAND ----------

# MAGIC %md
# MAGIC # CUJ 1: Generate evaluations from documents, and evaluate your model.
# MAGIC
# MAGIC NOTE: To properly test this CUJ, it would be best if you used your own RAG chain. You can follow the genai cookbook to create
# MAGIC one if you haven't already: https://ai-cookbook.io/
# MAGIC
# MAGIC Alternatively, you can use my model serving endpoint with databricks docs.
# MAGIC
# MAGIC Databricks RAG model serving endpoint: [`agents_mosaic_catalog-lilac_schema-databricks_rag_cleaned`](https://e2-dogfood.staging.cloud.databricks.com/ml/endpoints/agents_mosaic_catalog-lilac_schema-databricks_rag_cleaned?o=6051921418418893)
# MAGIC
# MAGIC Databricks Docs in Delta: [`mosaic_catalog.lilac_schema.databricks_rag_gaic_docs_cleaned`](https://e2-dogfood.staging.cloud.databricks.com/explore/data/mosaic_catalog/lilac_schema/databricks_rag_gaic_docs_cleaned?o=6051921418418893&activeTab=sample)
# MAGIC

# COMMAND ----------

chunks = spark.table(f"{UC_CATALOG}.{UC_SCHEMA}.ey_dbs_app_poc_parsed_docs_silver")

# COMMAND ----------



# COMMAND ----------

display(chunks)

# COMMAND ----------

import mlflow
from databricks.agents.eval import generate_evals_df
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# docs = (chunks
#         .withColumn("row_num", F.row_number().over(Window.partitionBy("path").orderBy(F.rand())))
#         .filter("row_num <= 10")
#         .select(F.col("chunked_text").alias("content"), F.col("path").alias("doc_uri"))
#         )
docs = (chunks
        # .withColumn("row_num", F.row_number().over(Window.partitionBy("path").orderBy(F.rand())))
        # .filter("row_num <= 10")
        .select(F.col("doc_parsed_contents")["parsed_content"].alias("content"), F.col("path").alias("doc_uri"))
        )
# docs = docs.toPandas()
display(docs)

# COMMAND ----------

# Export a Delta table to a Parquet file in a UC volume
parquet_output_path = f"{SOURCE_PATH}/eval_docs_input"

docs.repartition(1).write.mode("overwrite").parquet(parquet_output_path)

# COMMAND ----------

# MAGIC %md
# MAGIC I had to run the generation of query answer pairs in a testing workspace
# MAGIC https://e2-dogfood.staging.cloud.databricks.com/editor/notebooks/2557620792184964?o=6051921418418893#command/2557620792186115

# COMMAND ----------

path_to_parquet_files = f"/Volumes/felixflory/ey_dbs_workshop_2024_10/raw_data/eval_docs_outut/"
evals = spark.read.parquet(path_to_parquet_files)
evals.write.mode("overwrite").saveAsTable(EVALUATION_SET_FQN)
eval_df = spark.table(EVALUATION_SET_FQN)
display(eval_df)

# COMMAND ----------

# Define the agent as a function that calls the model serving endpoint for the Llama 3.1 model.
def dbx_rag_agent(input):
    client = mlflow.deployments.get_deploy_client("databricks")
    input.pop("databricks_options", None)
    return client.predict(
        endpoint="agents_main-davidhuang-davidhuang_agent_quick_start",
        inputs=input,
    )

# Evaluate the model using the newly generated evaluation set. After the function call completes, click the UI link to see the results.
with mlflow.start_run():
  results = mlflow.evaluate(
      model=dbx_rag_agent, 
      data=evals, 
      model_type="databricks-agent"
  )

# COMMAND ----------

with mlflow.start_run():
    # Evaluate
    eval_results = mlflow.evaluate(
        data=evals, # Your evaluation set
        model='runs:/d79a07fc4f8140988477d40c8ad504de/chain',
        model_type="databricks-agent", # activate Mosaic AI Agent Evaluation
    )

# COMMAND ----------


