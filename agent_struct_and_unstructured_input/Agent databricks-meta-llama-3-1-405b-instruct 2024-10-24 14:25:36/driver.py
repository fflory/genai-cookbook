# Databricks notebook source
# MAGIC %md
# MAGIC # Driver notebook
# MAGIC
# MAGIC This is an auto-generated notebook created by an AI Playground export. We generated three notebooks in the same folder:
# MAGIC - [agent]($./agent): contains the code to build the agent.
# MAGIC - [config.yml]($./config.yml): contains the configurations.
# MAGIC - [**driver**]($./driver): logs, evaluate, registers, and deploys the agent.
# MAGIC
# MAGIC This notebook uses Mosaic AI Agent Framework ([AWS](https://docs.databricks.com/en/generative-ai/retrieval-augmented-generation.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/retrieval-augmented-generation)) to deploy the agent defined in the [agent]($./agent) notebook. The notebook does the following:
# MAGIC 1. Logs the agent to MLflow
# MAGIC 2. Evaluate the agent with Agent Evaluation
# MAGIC 3. Registers the agent to Unity Catalog
# MAGIC 4. Deploys the agent to a Model Serving endpoint
# MAGIC
# MAGIC ## Prerequisities
# MAGIC
# MAGIC - Address all `TODO`s in this notebook.
# MAGIC - Review the contents of [config.yml]($./config.yml) as it defines the tools available to your agent, the LLM endpoint, and the agent prompt.
# MAGIC - Review and run the [agent]($./agent) notebook in this folder to view the agent's code, iterate on the code, and test outputs.
# MAGIC
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See docs ([AWS](https://docs.databricks.com/en/generative-ai/deploy-agent.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/deploy-agent)) for details

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-agents mlflow langchain==0.2.16 langgraph-checkpoint==1.0.12  langchain_core langchain-community==0.2.16 langgraph==0.2.16 pydantic
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the `agent` as an MLflow model
# MAGIC Log the agent as code from the [agent]($./agent) notebook. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).

# COMMAND ----------

# Log the model to MLflow
import os
import mlflow

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "What are the total net equity assets for fiscal year 2019 for the customer Ad Astra?"
        }
    ]
}

with mlflow.start_run():
    logged_agent_info = mlflow.langchain.log_model(
        lc_model=os.path.join(
            os.getcwd(),
            'agent',
        ),
        pip_requirements=[
            f"langchain==0.2.16",
            f"langchain-community==0.2.16",
            f"langgraph-checkpoint==1.0.12",
            f"langgraph==0.2.16",
            f"pydantic",
        ],
        model_config="config.yml",
        artifact_path='agent',
        input_example=input_example,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with [Agent Evaluation](https://learn.microsoft.com/azure/databricks/generative-ai/agent-evaluation/)
# MAGIC
# MAGIC You can edit the requests or expected responses in your evaluation dataset and run evaluation as you iterate your agent, leveraging mlflow to track the computed quality metrics.

# COMMAND ----------

import pandas as pd

eval_examples = [
    {
        "request": {
            "messages": [
                {
                    "role": "user",
                    "content": "What are the total net equity assets for fiscal year 2019 for the customer Ad Astra?"
                }
            ]
        },
        "expected_response": None
    },
    {
        "request": {
            "messages": [
                {
                    "role": "user",
                    "content": "What are the total net equity assets for fiscal year 2019 for the customer Ad Astra?"
                },
                {
                    "role": "assistant",
                    "content": "This question requires information about the customer Ad Astra's financial performance, specifically their net equity assets for fiscal year 2019. The felixflory.ey_dbs_workshop_2024_10.chat_with_genie function is the most relevant tool to answer this question as it can provide information about the company's financial performance, including year over year earnings, investment performance, and other metrics.\n\n<uc_function_call>{\"id\":\"kdCC2ws23PhwKul5bCSa\",\"name\":\"felixflory.ey_dbs_workshop_2024_10.chat_with_genie\",\"arguments\":\"{\\\"question\\\":\\\"What are the total net equity assets for fiscal year 2019 for the customer Ad Astra?\\\",\\\"contextual_history\\\":\\\"no relevant history\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"kdCC2ws23PhwKul5bCSa\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"Genie Results are: \\\\nSpace ID: 01ef9217da36130ba9e15a0c0eeae810\\\\nConversation ID: 01ef923d7e731607b148e7c34bc588b6\\\\nQuestion That Was Asked: Use the contextual history to answer the question. The history may or may not help you. Use it if you find it relevant.\\\\n        \\\\n        Contextual History: no relevant history\\\\n        \\\\n        Question to answer: What are the total net equity assets for fiscal year 2019 for the customer Ad Astra?\\\\n        \\\\nContent: None\\\\nSQL Query: SELECT SUM(`fy19`) AS total_net_equity_assets_fy19 FROM `felixflory`.`ey_dbs_workshop_2024_10`.`ad_astra_1_6_3_1_ad_astra___net_assets_details___02_11_2020_xlsx_effects_from_ifrs_16` WHERE `net_assets` ILIKE '%equity%'\\\\nSQL Query Description: This query calculates the total net equity assets for fiscal year 2019 for the customer Ad Astra. It filters the data to include only the net assets categorized as 'equity'.\\\\nSQL Query Result: [{'total_net_equity_assets_fy19': 1.234672854182996}]\\\\nError: None\\\"]]}\"}</uc_function_result>\n\nThe total net equity assets for fiscal year 2019 for the customer Ad Astra are 1.234672854182996."
                },
                {
                    "role": "user",
                    "content": "this is not about net equity effects - please try again?"
                }
            ]
        },
        "expected_response": None
    }
]

eval_dataset = pd.DataFrame(eval_examples)
display(eval_dataset)

# COMMAND ----------

import mlflow
import pandas as pd

with mlflow.start_run(run_id=logged_agent_info.run_id):
    eval_results = mlflow.evaluate(
        f"runs:/{logged_agent_info.run_id}/agent",  # replace `chain` with artifact_path that you used when calling log_model.
        data=eval_dataset,  # Your evaluation dataset
        model_type="databricks-agent",  # Enable Mosaic AI Agent Evaluation
    )

# Review the evaluation results in the MLFLow UI (see console output), or access them in place:
display(eval_results.tables['eval_results'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "main"
schema = "default"
model_name = ""
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents

# Deploy the model to the review app and a model serving endpoint
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version)
