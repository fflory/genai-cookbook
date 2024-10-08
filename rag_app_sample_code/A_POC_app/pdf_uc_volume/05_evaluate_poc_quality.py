# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import pandas as pd
from databricks import agents

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC # Load your evaluation set from the previous step

# COMMAND ----------

df = spark.table(EVALUATION_SET_FQN)
eval_df = df.toPandas()
display(eval_df)

# COMMAND ----------

# If you did not collect feedback from your stakeholders, and want to evaluate using a manually curated set of questions, you can use the structure below.

eval_data = [
    {
        ### REQUIRED
        # Question that is asked by the user
        "request": "What is the difference between reduceByKey and groupByKey in Spark?",

        ### OPTIONAL
        # Optional, user specified to identify each row
        "request_id": "your-request-id",
        # Optional: correct response to the question
        # If provided, Agent Evaluation can compute additional metrics.
        "expected_response": "There's no significant difference.",
        # Optional: Which documents should be retrieved.
        # If provided, Agent Evaluation can compute additional metrics.
        "expected_retrieved_context": [
            {
                # URI of the relevant document to answer the request
                # Must match the contents of `document_uri` in your chain config / Vec
                "doc_uri": "doc_uri_2_1",
            },
        ],
    }
]

# Uncomment this row to use the above data instead of your evaluation set
# eval_df = pd.DataFrame(eval_data)

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluate the POC application

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get the MLflow run of the POC application 

# COMMAND ----------

runs = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT_NAME], filter_string=f"run_name = '{POC_CHAIN_RUN_NAME}'", output_format="list", 
                          order_by=["attributes.start_time desc"], max_results=1)

if len(runs) != 1:
    raise ValueError(f"Found {len(runs)} run with name {POC_CHAIN_RUN_NAME}.  Ensure the run name is accurate and try again.")

poc_run = runs[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the correct Python environment for the POC's app
# MAGIC
# MAGIC TODO: replace this with env_manager=virtualenv once that works

# COMMAND ----------

pip_requirements = mlflow.pyfunc.get_model_dependencies(f"runs:/{poc_run.info.run_id}/chain")

# COMMAND ----------

# MAGIC %pip install -qqqq -r $pip_requirements

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run evaluation on the POC app

# COMMAND ----------

# MAGIC %md
# MAGIC eval with ground truth (not from feedback)

# COMMAND ----------

with mlflow.start_run(run_name="finbench_eval2"):
    # Evaluate
    eval_results = mlflow.evaluate(
        data=eval_df,
        model=f"runs:/{poc_run.info.run_id}/chain",  # replace `chain` with artifact_path that you used when calling log_model.  By default, this is `chain`.
        model_type="databricks-agent",
        # extra_metrics=[answer_relevance_metric]
    )

# COMMAND ----------

finbench_results = eval_results.tables['eval_results']
# [i for i in finbench_results.loc[finbench_results['response/llm_judged/correctness/rating'] == 'yes']['request_id']]
finbench_results

# COMMAND ----------

with mlflow.start_run(run_id=poc_run.info.run_id):
    # Evaluate
    eval_results = mlflow.evaluate(
        data=eval_df,
        model=f"runs:/{poc_run.info.run_id}/chain",  # replace `chain` with artifact_path that you used when calling log_model.  By default, this is `chain`.
        model_type="databricks-agent",
        # extra_metrics=[answer_relevance_metric]
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Look at the evaluation results
# MAGIC
# MAGIC You can explore the evaluation results using the above links to the MLflow UI.  If you prefer to use the data directly, see the cells below.

# COMMAND ----------

eval_results.metrics

# COMMAND ----------

# Evaluation results including LLM judge scores/rationales for each row in your evaluation set
per_question_results_df = eval_results.tables['eval_results']

# You can click on a row in the `trace` column to view the detailed MLflow trace
display(per_question_results_df)

# COMMAND ----------

# Evaluation results including LLM judge scores/rationales for each row in your evaluation set
per_question_results_df = eval_results.tables['eval_results']

# You can click on a row in the `trace` column to view the detailed MLflow trace
display(per_question_results_df)
