# Databricks notebook source
import pyspark.sql.functions as F

col_map = {"question": "request", "financebench_id": "request_id", "answer": "expected_response"}
examples_pass_llm_judgment = [
  'financebench_id_08286',
  'financebench_id_00684',
  'financebench_id_04417',
  'financebench_id_01981',
  'financebench_id_00288',
  'financebench_id_01091',
  'financebench_id_01290',
  'financebench_id_10285',
  'financebench_id_04254',
  'financebench_id_01244',
  'financebench_id_09724',
  'financebench_id_00839',
  'financebench_id_00723',
  'financebench_id_04980',
  'financebench_id_00882',
  'financebench_id_00746',
  'financebench_id_04302',
  'financebench_id_04784']
question_selection = examples_pass_llm_judgment + [
  "financebench_id_00299", "financebench_id_03029", "financebench_id_04412"
  ]

finbench_eval_raw = spark.table("felixflory.rag_felixflory.financebench_eval_set")
finbench_eval_raw_count = finbench_eval_raw.count()
sample_rate = 20 / (finbench_eval_raw_count + len(question_selection))

eval_df = (
  finbench_eval_raw
  .withColumnsRenamed(col_map)
  .withColumn("expected_retrieved_context", F.array(F.struct(F.col("doc_uri"))))
  .select(*(list(col_map.values()) + ["expected_retrieved_context"]))
  .filter(F.col("request_id").isin(question_selection) | 
          (F.rand(4345) <= F.lit(sample_rate)))
)
display(eval_df)
