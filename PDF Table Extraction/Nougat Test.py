# Databricks notebook source
# MAGIC %pip install python-Levenshtein

# COMMAND ----------

# MAGIC %pip install huggingface_hub

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install torch transformers pillow
# MAGIC

# COMMAND ----------

# MAGIC %pip install "nougat-ocr[api, dataset]"

# COMMAND ----------

from transformers import VisionEncoderDecoderModel, AutoProcessor
import torch
from PIL import Image

# COMMAND ----------

model_name = "facebook/nougat-base"
processor = AutoProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# COMMAND ----------

image_path = "./cropped_table.jpeg"
#image_path = "./ey_dbs_one_page_pdf.jpg"
image = Image.open(image_path)

# COMMAND ----------

if image.mode != "RGB":
    image = image.convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values
pixel_values = pixel_values.to(device)

generated_ids = model.generate(
    pixel_values, 
    max_length=512, 
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    no_repeat_ngram_size=2,
    repetition_penalty=1.2
    )
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Save or process the extracted text as needed
with open(f"./cropped_table.md", "w") as f:
    f.write(generated_text)

# COMMAND ----------

import re
from PIL import Image
from transformers import NougatProcessor, VisionEncoderDecoderModel
import torch

processor = NougatProcessor.from_pretrained("facebook/nougat-base")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# prepare local PDF image for the model
filepath = "./cropped_table.jpeg"
image = Image.open(filepath)
pixel_values = processor(image, return_tensors="pt").pixel_values

# generate transcription (here we only generate 30 tokens)
outputs = model.generate(
    pixel_values.to(device),
    min_length=1,
    max_new_tokens=2000,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
)

sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
sequence = processor.post_process_generation(sequence, fix_markdown=False)
# note: we're using repr here such for the sake of printing the \n characters, feel free to just print the sequence
print(repr(sequence))

# COMMAND ----------

sequence

# COMMAND ----------

from mlflow import pyfunc
import mlflow
from mlflow.models.signature import infer_signature
from transformers import VisionEncoderDecoderModel, NougatProcessor
import torch
from PIL import Image
import numpy as np

class HuggingFaceModelWrapper(pyfunc.PythonModel):
    def load_context(self, context):
        self.processor = NougatProcessor.from_pretrained("facebook/nougat-base")
        self.model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def predict(self, context, model_input):
        image = Image.open(model_input)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        outputs = self.model.generate(
            pixel_values.to(self.device),
            min_length=1,
            max_new_tokens=30,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
        )
        sequence = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        sequence = self.processor.post_process_generation(sequence, fix_markdown=False)
        return sequence

# Example input and output for signature inference
example_image_path = "./ey_dbs_one_page_pdf.jpg"
example_image = Image.open(example_image_path)
example_pixel_values = NougatProcessor.from_pretrained("facebook/nougat-base")(example_image, return_tensors="pt").pixel_values
example_output = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base").generate(example_pixel_values)

# Convert tensor output to numpy array
example_output_np = example_output.cpu().numpy()

# Infer signature
signature = infer_signature(np.array([example_image_path]), example_output_np)

# Log the model
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="huggingface_model",
        python_model=HuggingFaceModelWrapper(),
        conda_env={
            'channels': ['defaults'],
            'dependencies': [
                'python=3.8.5',
                'pip',
                {
                    'pip': [
                        'mlflow',
                        'torch',
                        'transformers',
                        'Pillow'
                    ],
                },
            ],
            'name': 'mlflow-env'
        },
        signature=signature
    )

# Register the model
mlflow.set_registry_uri("databricks-uc")
model_uri = f"runs:/{run.info.run_id}/huggingface_model"
model_details = mlflow.register_model(model_uri=model_uri, name="yash_gupta_catalog.ey_dbs.nougat_base")

# COMMAND ----------

example_output_np

# COMMAND ----------

from PIL import Image

# Assuming the model and wrapper have been properly loaded and registered as per the previous cells
# Load the model from the registry
model_name = "yash_gupta_catalog.ey_dbs.nougat_base"
model_version = 4  # Assuming version 1 for simplicity; adjust as necessary
loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

# Prepare the input image (PDF is not supported based on the previous context, so we use JPG)
image_path = "./ey_dbs_one_page_pdf.jpg"  # Replace with the path to your JPG image

# Use the loaded model to predict
result = loaded_model.predict(image_path)

# Display the result
display(result)
