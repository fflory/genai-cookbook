# Databricks notebook source
# MAGIC %md
# MAGIC # Document Table Data Extraction via Donut Models using Batch Question Answering
# MAGIC
# MAGIC This example use a different model approach, notibly based on the Donut models: https://huggingface.co/docs/transformers/main/en/model_doc/donut
# MAGIC
# MAGIC This example provides an example as to where all of your images could be processed using a Transformers Pipeline approach, where multiple images and questions could be provided in one pipeline and multiple answers returned. Just like with the LLaVa example, once the individual data points are extracted they could be saved to a Delta table and used to populate any number of documents, or used in other applications like traditional LLM chatbots as well.

# COMMAND ----------

!nvidia-smi

# COMMAND ----------

from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import re
import torch
import glob
import numpy as np

# COMMAND ----------

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
processor.tokenizer.padding_side = 'left'
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

# COMMAND ----------

# move model to GPU if it's available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# COMMAND ----------

images = []
i = 0
for f in glob.iglob("/Volumes/felixflory/ey_dbs_workshop_2024_10/images/*.png"):
    if i < 3:
        images.append(Image.open(f))
    i += 1

# images = images[1:3]
#images = np.array(images)

# COMMAND ----------

images[2]

# COMMAND ----------


image = images[2]
width, height = image.size


# COMMAND ----------

# prepare encoder inputs
pixel_values = processor(images=images, return_tensors="pt").pixel_values
batch_size = pixel_values.shape[0]
batch_size

# COMMAND ----------

# MAGIC %md
# MAGIC After all of our images are loaded, we can use the `docvqa` task prompt to ask multiple questions of the multiple images. Each response is saved in a list in the response:

# COMMAND ----------

# prepare decoder inputs
task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
questions = [
  # "What are all the adjusted NWC values ?"
  "Extract all the NWC values"
  # "What is the in-network co-pay for Durable Medical Equipement, or DME?", 
  # "What is the out of network co-pay for hospice?",
  # "What is the maximum out of network out-of-pocket amount in the plan?"
  ]
prompts = [task_prompt.replace("{user_input}", question) for question in questions]
decoder_input_ids = processor.tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt").input_ids

# COMMAND ----------

outputs = model.generate(
    pixel_values.to(device),
    decoder_input_ids=decoder_input_ids.to(device),
    max_length=model.decoder.config.max_position_embeddings,
    early_stopping=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    num_beams=1,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
)

# COMMAND ----------

sequences = processor.batch_decode(outputs.sequences)

# COMMAND ----------

for seq in sequences:
  sequence = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
  sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
  print(processor.token2json(sequence))

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Using cord-v2 Tasks
# MAGIC
# MAGIC Another approach is to use CORD-V2 tasks here to extract all the data from a given chart, similar to LLaVa. In this example, the model used was on a particular type of document (menus and ingrediants) but, again, if used in a fine tuning scenario could be made to extract similar structures for your specific documents with a high degree of accuracy. For reference, see the following: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Donut/CORD/Fine_tune_Donut_on_a_custom_dataset_(CORD)_with_PyTorch_Lightning.ipynb

# COMMAND ----------

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

# COMMAND ----------

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# COMMAND ----------

task_prompt = "<s_cord-v2>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

# COMMAND ----------

pixel_values = processor(images[10], return_tensors="pt").pixel_values

# COMMAND ----------

outputs = model.generate(
    pixel_values.to(device),
    decoder_input_ids=decoder_input_ids.to(device),
    max_length=model.decoder.config.max_position_embeddings,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
)

# COMMAND ----------



# COMMAND ----------

sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
print(processor.token2json(sequence))
