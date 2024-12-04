# Databricks notebook source
!pip install pdf2image

# COMMAND ----------

# MAGIC %sh
# MAGIC sudo apt-get install -y poppler-utils

# COMMAND ----------

from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

# COMMAND ----------

images = convert_from_path('/Volumes/drewfurgiuele/default/pdfs/H3256_001_000_SOB.pdf')

# COMMAND ----------

images

# COMMAND ----------

from PIL import Image
import os
import tempfile

# COMMAND ----------

with tempfile.TemporaryDirectory() as temp_dir:
  # convert pdf to multiple image
  images = convert_from_path("/Volumes/drewfurgiuele/default/pdfs/H3256_001_000_SOB.pdf", output_folder=temp_dir)

  # save images to temporary directory
  temp_images = []
  for i in range(len(images)):
      image_path = f'{temp_dir}/{i}.jpg'
      images[i].save(image_path, 'JPEG')
      temp_images.append(image_path)

  # read images into pillow.Image
  imgs = list(map(Image.open, temp_images))

min_img_width = min(i.width for i in imgs)

# find total height of all images
total_height = 0
for i, img in enumerate(imgs):
    total_height += imgs[i].height

# create new image object with width and total height
merged_image = Image.new(imgs[0].mode, (min_img_width, total_height))

y = 0
for img in imgs:
    merged_image.paste(img, (0, y))
    y += img.height

# save merged image
merged_image.save("/Volumes/drewfurgiuele/default/pdfs/output/H3256_001_000_SOB.png", "PNG")

# COMMAND ----------

merged_image

# COMMAND ----------

!pip install transformers

# COMMAND ----------

from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

# COMMAND ----------

from PIL import Image

#Image2 = Image.open(r'/Volumes/drewfurgiuele/default/pdfs/output/H3256_001_000_SOB.png').convert("RGB")
Image2 = Image.open(r'/Volumes/drewfurgiuele/default/pdfs/output/one_page_pdf.png').convert("RGB")
print(type(Image2))

# COMMAND ----------

pixel_values = processor(Image2, return_tensors="pt").pixel_values

print(pixel_values.shape)

# COMMAND ----------

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(device)

# COMMAND ----------

task_prompt = "<s_cord-v2>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

pixel_values = processor(Image2, return_tensors="pt").pixel_values

outputs = model.generate(
    pixel_values.to(device),
    decoder_input_ids=decoder_input_ids.to(device),
    max_length=500000,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=False,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
    output_scores=True,
)

# COMMAND ----------

import re

sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
print(sequence)

# COMMAND ----------

processor.token2json(sequence)
