# Databricks notebook source



# COMMAND ----------

dbutils.fs.cp(
    "/Volumes/field_ai_examples/alphaleger/financebench",
    "/Volumes/felixflory/financebench/financebench",
    recurse=True,
)
spark.sql(
    """
    CREATE TABLE felixflory.financebench.financebench_agent_app_evaluation_set
    AS SELECT * FROM field_ai_examples.alphaleger.managed_evaluation_set
"""
)
# COMMAND ----------
