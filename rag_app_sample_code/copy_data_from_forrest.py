# Databricks notebook source
!cp -r /Volumes/field_ai_examples/rag/financebench/* /Volumes/felixflory/rag_felixflory/source_docs

# COMMAND ----------

!cp /Volumes/main/felix_flory_financebench/data/financebench_eval_set.snappy.parquet /Volumes/felixflory/rag_felixflory/financebench_eval_parquet

# COMMAND ----------

input_parquet_filename = '/Volumes/felixflory/rag_felixflory/financebench_eval_parquet/financebench_eval_set.snappy.parquet'

# COMMAND ----------

# MAGIC %run ./00_global_config

# COMMAND ----------

# output_table = f"{UC_CATALOG}.{UC_SCHEMA}.my_agent_app_evaluation_set"
# spark.read.parquet(input_parquet_filename).write.mode("overwrite").saveAsTable(EVALUATION_SET_FQN)

# COMMAND ----------

output_table = f"{UC_CATALOG}.{UC_SCHEMA}.financebench_eval_set"
spark.read.parquet(input_parquet_filename).write.mode("overwrite").saveAsTable(output_table)
