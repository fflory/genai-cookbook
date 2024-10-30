# Databricks notebook source
# MAGIC %pip install openai presidio-analyzer presidio-anonymizer

# COMMAND ----------

from presidio_analyzer import EntityRecognizer, RecognizerResult
from openai import OpenAI
import os
from getpass import getpass

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = getpass()

# COMMAND ----------

class LLMRecognizer(EntityRecognizer):
    def __init__(self, supported_entities=None, name="LLMRecognizer", version="0.1"):
        supported_entities = supported_entities or ["PERSON", "LOCATION", "ORGANIZATION"]
        super().__init__(supported_entities=supported_entities, name=name, version=version)

    def load(self):
        # Set up your Databricks model serving endpoint
        self.client = OpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
            base_url="https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints"
        )

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        # Use your model serving endpoint for entity recognition
        response = self.client.completions.create(
            prompt=f"Extract named entities from the text: {text}",
            model="Yash_GPT",
            max_tokens=100
        )

        # Parse the response and extract entity spans
        for entity in response['choices'][0]['text'].split("\n"):
            if entity:
                entity_type, entity_value = entity.split(":")
                start = text.find(entity_value.strip())
                end = start + len(entity_value.strip())
                results.append(RecognizerResult(entity_type.strip(), start, end, score=0.85))
        
        return results

# COMMAND ----------

from presidio_analyzer import AnalyzerEngine

# Initialize the Analyzer Engine
analyzer = AnalyzerEngine()

# Add the custom recognizer
llm_recognizer = LLMRecognizer()
analyzer.registry.add_recognizer(llm_recognizer)


# COMMAND ----------

# Ensure you have a valid access token
access_token = "YOUR_VALID_ACCESS_TOKEN"

# Configure the analyzer with the access token
analyzer = AnalyzerEngine()

text = "Barack Obama was the president of the United States."
results = analyzer.analyze(
    text=text,
    entities=["PERSON", "LOCATION", "ORGANIZATION"],
    language="en"
)

for result in results:
    print(f"Entity: {result.entity_type}, Text: {text[result.start:result.end]}")
