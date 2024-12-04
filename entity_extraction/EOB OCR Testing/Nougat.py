# Databricks notebook source
from huggingface_hub import hf_hub_download
import re
from PIL import Image

from transformers import NougatProcessor, VisionEncoderDecoderModel, NougatTokenizerFast
from datasets import load_dataset
import torch

# COMMAND ----------

processor = NougatProcessor.from_pretrained("facebook/nougat-base")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")
tokenizer = NougatTokenizerFast.from_pretrained("facebook/nougat-base")

# COMMAND ----------

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# COMMAND ----------

# prepare PDF image for the model
image = Image.open("/Volumes/drewfurgiuele/default/pdfs/output/H3256_001_000_SOB_pg2.png").convert('RGB')
pixel_values = processor(image, return_tensors="pt").pixel_values

# COMMAND ----------

# generate transcription (here we only generate 30 tokens)
outputs = model.generate(
    pixel_values.to(device),
    min_length=1,
    max_new_tokens=1000
)

# COMMAND ----------

sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
sequence = processor.post_process_generation(sequence, fix_markdown=False)
corrected_tables = tokenizer.correct_tables(sequence)


# COMMAND ----------

# note: we're using repr here such for the sake of printing the \n characters, feel free to just print the sequence
print(sequence)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Summary of Benefits
# MAGIC
# MAGIC ### January 1, 2024 - December 31, 2024
# MAGIC
# MAGIC This is a summary of what we cover and what you pay. For a complete list of covered services, limitations and exclusions, review the Evidence of Coverage (EOC) at **myUHCMedicare.com** or call Customer Service for help. After you enroll in the plan, you will get more information on how to view your plan details online.
# MAGIC
# MAGIC ### UHC Dual Complete GA-S001 (PPO D-SNP)
# MAGIC
# MAGIC \begin{tabular}{l l} \hline \hline \multicolumn{1}{c}{**Medical premium, deductible and limits**} \\ \hline  & **In-network** & **Out-of-network** \\ \hline \multicolumn{1}{c}{**Monthly plan premium**} & \$0 \\  & & You may need to continue to pay your Medicare Part B premium \\ \hline \multicolumn{1}{c}{**Annual medical deductible**} & Your deductible is \$0 or the Original Medicare Part B deductible amount, combined in and out-of-network. The 2023 Original Medicare deductible amount is \$226. The 2024 amount will be set by CMS in the fall of 2023. Our plan will provide updated rates as soon as they are released. \\ \hline \multicolumn{1}{c}{**Maximum out-of-pocket amount** (does not include prescription drugs)} & \$0 \\  & This is the most you will pay out-of-pocket each year for Medicare-covered services and supplies received from network providers. \\ \hline \multicolumn{1}{c}{**Medicare cost-sharing**} & If you have full Medicaid benefits or are a Qualified Medicare Beneficiary (QMB), you will pay \$0 for your Medicare-covered services as noted by the cost-sharing in this chart

# COMMAND ----------


