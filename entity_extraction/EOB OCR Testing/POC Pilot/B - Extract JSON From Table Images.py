# Databricks notebook source
# MAGIC %md
# MAGIC # Extracting PDF Table Data via a MultiModal Chat Bot
# MAGIC
# MAGIC This example is built around the LLaVa model via HuggingFace: https://huggingface.co/docs/transformers/en/model_doc/llava Specifically, it's using the LLaVa-Next model, leveraging mistralai/Mistral-7B-Instruct-v0.2 as LLM (https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf). The idea of this example is to load a given image, and using prompts similar to a conversational LLM, ask it to extract the table in the image into a different data structure, like a JSON document. From there, you can extact key/value pairs from the resulting response. The responses would then be saved to a traditional data store and could be retrieved for any other use you'd have of them.
# MAGIC
# MAGIC This model does perform very well, but could perform even better if fine-tuned on your own document set.

# COMMAND ----------

!pip install git+https://github.com/huggingface/transformers

# COMMAND ----------

!pip install 'accelerate>=0.26.0'

# COMMAND ----------

!pip install bitsandbytes

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
import torch
import gc

# COMMAND ----------

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# COMMAND ----------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    quantization_config=quantization_config,
    attn_implementation="flash_attention_2"
)
model.to(device)

# COMMAND ----------

from PIL import Image
import requests

# COMMAND ----------

dbutils.fs.ls("/Volumes/felixflory/ey_dbs_workshop_2024_10/images/")

# COMMAND ----------


image = Image.open('/Volumes/felixflory/ey_dbs_workshop_2024_10/images/page-9.png')
image

# COMMAND ----------

#import io
#import base64 as b64
#from PIL import Image

#@udf ("string")
#def extract_json_via_vision_llm(content):
prompt = "[INST] <image>\nYou are a helpful assistant that mimicks the behavior of an API. The image you are reviewing is a table and you must take the image and convert the structure a JSON object using the rows and columns in the table. You should respond like an API and just provide the raw JSON document. Omit any other resoponses except the JSON.[/INST]"
#buffer = io.BytesIO()
#image = Image.open(io.BytesIO(content))
inputs = processor(image, prompt, return_tensors="pt").to(device)
output = model.generate(**inputs, max_new_tokens=700)
result = processor.decode(output[0], skip_special_tokens=True)
json_only = result.split("```")
json_only[1]

# COMMAND ----------

import json
json_only = result.split("```")[1].split("json\n")
json_dict = json.loads(json_only[1])
json_dict

# COMMAND ----------

json_dict["Medical Benefits"]["Out-of-network"]["Outpatient"]["Occupational Therapy"]
