# Databricks notebook source
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

torch.cuda.empty_cache()
gc.collect()

# COMMAND ----------

quantization_config = BitsAndBytesConfig(
    llm_int8_enable_fp32_cpu_offload=True
    #load_in_4bit=True,
    #bnb_4bit_quant_type="nf4",
    #bnb_4bit_compute_dtype=torch.float16,
)

# COMMAND ----------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    quantization_config=quantization_config
)
model.to(device)

# COMMAND ----------

from PIL import Image
import requests

image = Image.open("/Volumes/drewfurgiuele/default/pdfs/output/H3256_001_000_SOB_pg2.png")
prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"

inputs = processor(prompt, image, return_tensors="pt").to(device)
output = model.generate(**inputs, max_new_tokens=200)
                        

# COMMAND ----------

print(processor.decode(output[0], skip_special_tokens=True))

# COMMAND ----------

image = Image.open("//Volumes/drewfurgiuele/default/pdfs/output/justatable.png")
prompt = "[INST] <image>\nSummarize the table in the image into a JSON representation of the table. The table has two header rows. The has only 3 columns. The first column is the key, and the two other columns are for in network and out of network costs. Some of the rows span both of the value columns. There is a small line between the rows and the columns are seperated by whitespace. Only return the JSON structure and no additional information. You should only populate the values of the JSON keys in dollar figures. The values in the JSON document should only be dollar figures. If two dollar figures are found, use both in the value of the key an seperate them with a comma. If no dollar figure exists, replace the value with NA. Any of the dollar figures that are found should not have decimal places.[/INST]"

inputs = processor(prompt, image, return_tensors="pt").to(device)
output = model.generate(**inputs, max_new_tokens=500)

# COMMAND ----------

print(processor.decode(output[0], skip_special_tokens=True))

# COMMAND ----------


