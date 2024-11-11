# Databricks notebook source
import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")
client.create_endpoint(
    name="felix-gpt-40-endpoint",
    config={
        "served_entities": [{
            "name": "gpt-4o-mini",
            "external_model": {
                "name": "gpt-4o-mini",
                "provider": "openai",
                "task": "llm/v1/chat",
                "openai_config": {
                    "openai_api_key": "{{secrets/felix-flory/oai}}"
                }
            }
        }]
    }
)
# client.delete_endpoint(endpoint="felix-gpt-40-endpoint")

# COMMAND ----------

client.get_endpoint(endpoint="Yash_GPT_4o")

# COMMAND ----------

# get Databricks credentials

from mlflow.utils.databricks_utils import get_databricks_host_creds

creds = get_databricks_host_creds()

creds.token, creds.host

# use openai's client to make calls to Databricks Foundation Model API
import openai
client = openai.OpenAI(
    api_key=creds.token,
    base_url=creds.host + '/serving-endpoints',
)

# COMMAND ----------

chat_completion = client.chat.completions.create(
  messages=[
  {
    "role": "system",
    "content": "You are an AI assistant"
  },
  {
    "role": "user",
    "content": "print 2 + 3"
  }
  ],
  model="felix-gpt-40-endpoint",
  max_tokens=756
)

# COMMAND ----------

completions_response = client.predict(
    endpoint="felix-gpt-40-endpoint",
    inputs={
        "messages": [
            {
                "role": "user",
                "content": "print 2 + 3"
            }
        ]
    }
)

# COMMAND ----------




# COMMAND ----------

# MAGIC %md
# MAGIC
