# Databricks notebook source
# MAGIC %md
# MAGIC # Agent proof of concept.
# MAGIC
# MAGIC By the end of this notebook, you will have created a POC of your Agent that you can interact with, and ask questions.
# MAGIC
# MAGIC This means:
# MAGIC - We will have a mlflow model registered in the "Models" tab on the Databricks menu on the left. Models that are registered are just assets that can be instantiated from another notebook, but are not served on an endpoint. These models can be invoked with `mlflow.invoke()`.
# MAGIC - We will have a served model registered in the "Serving" tab on the Databricks menu on the left. This means that the model is served and can be accessed via a UI or a REST API for anyone in the workspace.
# MAGIC

# COMMAND ----------

# Versions of Databricks code are not locked since Databricks ensures changes are backwards compatible.
%pip install -qqqq -U databricks-agents databricks-vectorsearch databricks-sdk mlflow mlflow-skinny
# Restart to load the packages into the Python environment
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import the global configuration

# COMMAND ----------

# MAGIC %run ./00_global_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent configuration

# COMMAND ----------

# MAGIC %run ./agents/agent_config

# COMMAND ----------

retriever_config = RetrieverToolConfig(
    vector_search_index=f"{UC_CATALOG}.{UC_SCHEMA}.{AGENT_NAME}_chunked_docs_index",
    vector_search_schema=RetrieverSchemaConfig(
        primary_key="chunk_id",
        chunk_text="content_chunked",
        document_uri="doc_uri",
        additional_metadata_columns=[],
    ),
    parameters=RetrieverParametersConfig(num_results=5, query_type="ann"),
    vector_search_threshold=0.1,
    chunk_template="Passage text: {chunk_text}\nPassage metadata: {metadata}\n\n",
    prompt_template="""Use the following pieces of retrieved context to answer the question.\nOnly use the passages from context that are relevant to the query to answer the question, ignore the irrelevant passages.  When responding, cite your source, referring to the passage by the columns in the passage's metadata.\n\nContext: {context}""",
    tool_description_prompt="Search for documents that are relevant to a user's query about the [REPLACE WITH DESCRIPTION OF YOUR DOCS].",
)

# `llm_endpoint_name`: Model Serving endpoint with the LLM for your Agent.
#     - Either an [Foundational Models](https://docs.databricks.com/en/machine-learning/foundation-models/index.html) Provisioned Throughput / Pay-Per-Token or [External Model](https://docs.databricks.com/en/generative-ai/external-models/index.html) of type `/llm/v1/chat` with support for [function calling](https://docs.databricks.com/en/machine-learning/model-serving/function-calling.html).  Supported models: `databricks-meta-llama-3-70b-instruct` or any of the Azure OpenAI / OpenAI models.

llm_config = LLMConfig(
    # https://docs.databricks.com/en/machine-learning/foundation-models/index.html
    llm_endpoint_name="databricks-meta-llama-3-1-70b-instruct",
    # Define a template for the LLM prompt.  This is how the RAG chain combines the user's question and the retrieved context.
    llm_system_prompt_template=(
        """You are a helpful assistant that answers questions by calling tools.  Provide responses ONLY based on the information from tools that are explictly specified to you.  If you do not have a relevant tool for a question, respond with 'Sorry, I'm not trained to answer that question'."""
    ),
    # Parameters that control how the LLM responds.
    llm_parameters=LLMParametersConfig(temperature=0.01, max_tokens=1500),
)

agent_config = AgentConfig(
    retriever_tool=retriever_config,
    llm_config=llm_config,
    input_example={
        "messages": [
            {
                "role": "user",
                "content": "What is RAG?",
            },
        ]
    },
)

agent_config.dict()

# COMMAND ----------

# MAGIC %run ./validators/validate_agent_config

# COMMAND ----------

validate_retriever_config(retriever_config)
validate_llm_config(llm_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set the MLflow experiement name
# MAGIC
# MAGIC Used to store the Agent's model

# COMMAND ----------

import mlflow

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import the Agent's code

# COMMAND ----------

# MAGIC %run ./agents/function_calling_agent_w_retriever_tool

# COMMAND ----------

# MAGIC %md
# MAGIC ## Iteratively vibe check & adjust the Agent's configuration
# MAGIC
# MAGIC If you need to adjust the Agent's code, you can do so in the `./agents/function_calling_agent_w_retriever_tool` Notebook

# COMMAND ----------

# 1st turn of converastion
first_turn_input = {
    "messages": [
        {"role": "user", "content": f"what was walmarts revenue in 2019?"},
    ]
}
# # Load the Agent
# agent = AgentWithRetriever()
_config = mlflow.models.ModelConfig(
            development_config="./agents/generated_configs/agent.yaml"
        )
_model_serving_client = get_deploy_client("databricks")
_retriever_tool = VectorSearchRetriever(_config.get("retriever_tool"))
_vector_search_schema = _config.get("retriever_tool").get(
    "vector_search_schema"
)
# I have to run the followin twice
mlflow.models.set_retriever_schema(
    primary_key=_vector_search_schema.get("primary_key"),
    text_column=_vector_search_schema.get("chunk_text"),
    doc_uri=_vector_search_schema.get("doc_uri"),
)
_tool_functions = {
  "retrieve_documents": _retriever_tool,
  }

_chat_history = None

# COMMAND ----------

# response = agent.predict(model_input=first_turn_input)
context = None
model_input = first_turn_input
params = None

# COMMAND ----------

messages=model_input.get("messages")

# COMMAND ----------

# Parse `messages` array into the user's query & the chat history
with mlflow.start_span(name="parse_input", span_type="PARSER") as span:
    span.set_inputs({"messages": messages})
    # user_query = self.extract_user_query_string(messages)
    user_query = messages[-1]["content"]
    # Save the history inside the Agent's internal state
    # self.chat_history = self.extract_chat_history(messages)
    _chat_history= [] + messages[:-1]
    span.set_outputs(
        {"user_query": user_query, "chat_history": _chat_history}
    )

# COMMAND ----------

system_prompt = _config.get("llm_config").get("llm_system_prompt_template")

messages = (
    [{"role": "system", "content": system_prompt}]
    + _chat_history  # append chat history for multi turn
    + [{"role": "user", "content": user_query}]
)

# COMMAND ----------

# # Call the LLM to recursively calls tools and eventually deliver a generation to send back to the user
# (
#     model_response,
#     messages_log_with_tool_calls,
# ) = self.recursively_call_and_run_tools(messages=messages)

# COMMAND ----------

max_iter=10
kwargs = {"messages": messages}

messages = kwargs["messages"]
del kwargs["messages"]
i = 0

# while i < max_iter: first iteration
# response = chat_completion(messages=messages, tools=True)
tools=True

# COMMAND ----------

# def chat_completion(self, messages: List[Dict[str, str]], tools: bool = False):
endpoint_name = _config.get("llm_config").get("llm_endpoint_name")
llm_options = _config.get("llm_config").get("llm_parameters")

# Trace the call to Model Serving
traced_create = mlflow.trace(
    _model_serving_client.predict,
    name="chat_completions_api",
    span_type="CHAT_MODEL",
)

if tools:
    # Get all tools
    print("tools is True")

tools = [_retriever_tool.get_tool_definition()]

inputs = {
    "messages": messages,
    "tools": tools,
    **llm_options,
}
# else:
#     inputs = {
#         "messages": messages,
#         **llm_options,
#     }

inputs

# COMMAND ----------

# Use the traced_create to make the prediction
response = traced_create(
        endpoint=endpoint_name,
        inputs=inputs,
    )
# chat_completion end

# COMMAND ----------

response

# COMMAND ----------

# continue while i < max_iter: first iteration loop
assistant_message = response.choices[0]["message"]
# print(assistant_message)
tool_calls = assistant_message.get("tool_calls")
print(tool_calls)
tool_messages = []
# for tool_call in tool_calls:
tool_call = tool_calls[0]
function = tool_call["function"]
args = json.loads(function["arguments"])
# result = execute_function(function["name"], args)
function_name = function["name"]
# def execute_function(self, function_name, args):

the_function = _tool_functions.get(function_name)
result = the_function(**args)
print(result)

# COMMAND ----------

tool_message = {
    "role": "tool",
    "tool_call_id": tool_call["id"],
    "content": result,
}
tool_messages.append(tool_message)
# for tool_call in tool_calls: end of first and only iteration  

# COMMAND ----------

assistant_message_dict = assistant_message.copy()
del assistant_message_dict["content"]
messages = (
    messages
    + [
        assistant_message_dict,
    ]
    + tool_messages
)
# end while i < max_iter: for first iteration

# COMMAND ----------

import pprint
print(pprint.pformat(messages, indent=1, width=200, depth=5, compact=True, sort_dicts=False, underscore_numbers=False))

# COMMAND ----------

# while i < max_iter: second iteration

# def chat_completion(self, messages: List[Dict[str, str]], tools: bool = False): tools = True
endpoint_name = _config.get("llm_config").get("llm_endpoint_name")
llm_options = _config.get("llm_config").get("llm_parameters")


# Trace the call to Model Serving
traced_create = mlflow.trace(
    _model_serving_client.predict,
    name="chat_completions_api",
    span_type="CHAT_MODEL",
)

# if tools:
    # Get all tools
tools = [_retriever_tool.get_tool_definition()]

inputs = {
    "messages": messages,
    "tools": tools,
    **llm_options,
}
# else:
#     inputs = {
#         "messages": messages,
#         **llm_options,
#     }

# COMMAND ----------

print(pprint.pformat(inputs, indent=1, width=200, depth=5, compact=True, sort_dicts=False, underscore_numbers=False))

# COMMAND ----------

# Use the traced_create to make the prediction
response = traced_create(
        endpoint=endpoint_name,
        inputs=inputs,
    )
# chat_completion end

# COMMAND ----------

print(pprint.pformat(response, indent=1, width=200, depth=5, compact=True, sort_dicts=False, underscore_numbers=False))

# COMMAND ----------

assistant_message = response.choices[0]["message"]

tool_calls = assistant_message.get("tool_calls")
print(tool_calls)

if tool_calls is None:
  print("tool_calls is None")

# return 
(
            model_response,
            messages_log_with_tool_calls,
        ) = (response, messages)

# recursively_call_and_run_tools end

messages_log_with_tool_calls.append(model_response.choices[0]["message"])
messages_log_with_tool_calls = messages_log_with_tool_calls[1:]

# return 
rv = {
            "content": model_response.choices[0]["message"]["content"],
            # this should be returned back to the Review App (or any other front end app) and stored there so it can be passed back to this stateless agent with the next turns of converastion.
            "messages": messages_log_with_tool_calls,
        }
# end of predict 

# COMMAND ----------

print(pprint.pformat(rv, indent=1, width=200, depth=5, compact=True, sort_dicts=False, underscore_numbers=False))

# COMMAND ----------

# # Load the Agent
# agent = AgentWithRetriever()

# # Example for testing multiple turns of converastion

# # 1st turn of converastion
# first_turn_input = {
#     "messages": [
#         {"role": "user", "content": f"what was walmarts revenue in 2019?"},
#     ]
# }

# response = agent.predict(model_input=first_turn_input)

# # 2nd turn of converastion
# new_messages = response["messages"]
# new_messages.append({"role": "user", "content": f"how do i use it?"})
# # print(type(new_messages))
# second_turn_input = {"messages": new_messages}
# response = agent.predict(model_input=second_turn_input)
# print(response["content"])

# COMMAND ----------

# agent.retriever_tool.similarity_search("what is walmart's revenue in 2019?")
# agent.get_messages_array(model_input=first_turn_input)
# agent.tool_functions.get("retrieve_documents")
# agent.chat_history # None
# agent.config._read_config()
# agent.config.get("llm_config").get("llm_system_prompt_template")

# agent.recursively_call_and_run_tools(max_iter=3,messages=first_turn_input["messages"])

# COMMAND ----------

response

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the Agent POC to the Review App

# COMMAND ----------

from mlflow.models.resources import (
    DatabricksVectorSearchIndex,
    DatabricksServingEndpoint,
)
from mlflow.models.signature import ModelSignature
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest
import yaml
from databricks import agents
from databricks import vector_search

databricks_resources = [
    DatabricksServingEndpoint(endpoint_name=llm_config.llm_endpoint_name),
    DatabricksVectorSearchIndex(index_name=retriever_config.vector_search_index),
]

# Specify the full path to the Agent notebook
model_file = "agents/function_calling_agent_w_retriever_tool"
model_path = os.path.join(os.getcwd(), model_file)

with mlflow.start_run(run_name=POC_CHAIN_RUN_NAME):
    model_info = mlflow.pyfunc.log_model(
        python_model=model_path,
        model_config=agent_config.dict(),
        artifact_path="agent",
        input_example=agent_config.input_example,
        resources=databricks_resources,
        signature=ModelSignature(
            inputs=ChatCompletionRequest(),
            outputs=StringResponse(),
        ),
    )

# COMMAND ----------

### Invoke the logged model
model = mlflow.pyfunc.load_model(model_info.model_uri)
model.predict(agent_config.input_example)

# COMMAND ----------

# Use Unity Catalog as the model registry
mlflow.set_registry_uri("databricks-uc")

# Register the model to the Unity Catalog
uc_registered_model_info = mlflow.register_model(
    model_uri=model_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

from databricks import agents
import time
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate


# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version)

# Wait for the Review App to be ready
print("\nWaiting for endpoint to deploy.  This can take 15 - 20 minutes.", end="")

print(f"\n\nReview App: {deployment_info.review_app_url}")
