# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents
# MAGIC %restart_python or dbutils.library.restartPython()

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC # Cleanup
# MAGIC
# MAGIC manually delete serving endpoints, experiments, models, and checkpoints in UC volume, and the index

# COMMAND ----------

# MAGIC %run ./00_global_config

# COMMAND ----------



# COMMAND ----------

from databricks import agents

# COMMAND ----------

active_deployments = agents.list_deployments()

active_deployment = next((item for item in active_deployments if item.model_name == UC_MODEL_NAME), None) 

if active_deployment.review_app_url:
  print(f"Review App URL: {active_deployment.review_app_url}")

# COMMAND ----------

active_deployment

# COMMAND ----------

# agents.delete_deployment(model_name='felixflory.rag_felixflory.my_agent_app')

# COMMAND ----------

# _tables_to_delete_df = spark.sql(f"show tables in {UC_CATALOG}.{UC_SCHEMA}").filter(f"tableName like '%{RAG_APP_NAME}%'")
# display(_tables_to_delete_df)

# COMMAND ----------

# _tables_to_delete = _tables_to_delete_df.select("tableName").rdd.map(lambda r: r.tableName).collect()
# for t in _tables_to_delete:
#     spark.sql(f"DROP TABLE IF EXISTS {UC_CATALOG}.{UC_SCHEMA}.{t}")

# COMMAND ----------

# MAGIC %environment
# MAGIC "client": "1"
# MAGIC "base_environment": ""
