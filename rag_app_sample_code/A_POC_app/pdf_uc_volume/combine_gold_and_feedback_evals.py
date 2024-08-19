# Databricks notebook source
import pyspark.sql.functions as F
col_map = {"question": "request", "financebench_id": "request_id", "answer": "expected_response"}
question_selection = ["financebench_id_00299", "financebench_id_03029", "financebench_id_04412"]
# gold_to_feedback_map = {
#   "financebench_id_00299": "0b429aca-9e5f-43cd-9220-7f70bdd81025",
#   "financebench_id_04412": "33d514eb-1d37-4ea8-9222-6ff8eda710c5"}

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType

gold_to_feedback_schema = StructType([
    StructField("gold_id", StringType(), True),
    StructField("feedback_id", StringType(), True)
])

gold_to_feedback_data = [
  ("financebench_id_00299", "0b429aca-9e5f-43cd-9220-7f70bdd81025"),
  ("financebench_id_04412", "33d514eb-1d37-4ea8-9222-6ff8eda710c5")]

gold_to_feedback_df = spark.createDataFrame(gold_to_feedback_data, schema=gold_to_feedback_schema)
display(gold_to_feedback_df)

# COMMAND ----------

finbench_eval_withkey = (
  finbench_eval
  .join(gold_to_feedback_df, finbench_eval.request_id == gold_to_feedback_df.gold_id, "left")
)
display(finbench_eval_withkey)

# COMMAND ----------

eval_df_spark = spark.createDataFrame(eval_df)

finbench_eval_with_feedback = finbench_eval_withkey.join(eval_df_spark, finbench_eval_withkey.feedback_id == eval_df_spark.request_id, "left")

display(finbench_eval_with_feedback)
