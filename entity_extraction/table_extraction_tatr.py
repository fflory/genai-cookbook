# Databricks notebook source
# %pip install tatr
%pip install timm
%restart_python

# COMMAND ----------

from transformers import TableTransformerModel, TableTransformerConfig

# Initializing a Table Transformer microsoft/table-transformer-detection style configuration
configuration = TableTransformerConfig()

# Initializing a model from the microsoft/table-transformer-detection style configuration
model = TableTransformerModel(configuration)

# Accessing the model configuration
configuration = model.config

# COMMAND ----------

from transformers import AutoImageProcessor, TableTransformerModel
from huggingface_hub import hf_hub_download
from PIL import Image

file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_pdf.png")
image = Image.open(file_path).convert("RGB")

image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
model = TableTransformerModel.from_pretrained("microsoft/table-transformer-detection")

# prepare image for the model
inputs = image_processor(images=image, return_tensors="pt")

# forward pass
outputs = model(**inputs)

# the last hidden states are the final query embeddings of the Transformer decoder
# these are of shape (batch_size, num_queries, hidden_size)
last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)

# COMMAND ----------

from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import torch
from PIL import Image

file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_pdf.png")
image = Image.open(file_path).convert("RGB")

image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
    0
]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )

# COMMAND ----------

# RAG_APP_NAME rag_app
# UC_CATALOG felixflory
# UC_SCHEMA ey_dbs_workshop_2024_10
# UC_MODEL_NAME felixflory.ey_dbs_workshop_2024_10.rag_app
# VECTOR_SEARCH_ENDPOINT one-env-shared-endpoint-13
SOURCE_PATH = '/Volumes/felixflory/ey_dbs_workshop_2024_10/raw_docs'
# EVALUATION_SET_FQN felixflory.ey_dbs_workshop_2024_10.rag_app_evaluation_set
# MLFLOW_EXPERIMENT_NAME /Users/felix.flory@databricks.com/rag_app
# POC_DATA_PIPELINE_RUN_NAME data_pipeline_poc
# POC_CHAIN_RUN_NAME poc

# COMMAND ----------

_sp = SOURCE_PATH+'/project_churches/Project Churches - FINAL Red Flag Report 181121.pdf'

# COMMAND ----------

from tatr import TATR

# Load the PDF
pdf_path = _sp
tatr = TATR(pdf_path)

# Extract tables
tables = tatr.extract_tables()

# Process tables
for i, table in enumerate(tables):
    print(f"Table {i+1}")
    print(table.to_dataframe())  # Convert to a DataFrame

# Save to CSV
table.to_dataframe().to_csv("output_table.csv", index=False)
