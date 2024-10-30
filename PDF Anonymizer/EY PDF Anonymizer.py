# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Install libraries & dependencies

# COMMAND ----------

# MAGIC %pip install python-pptx
# MAGIC %pip install Pillow
# MAGIC %pip install presidio-analyzer
# MAGIC %pip install presidio-image-redactor
# MAGIC %pip install PyMuPDF

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Import Necessary Libraries

# COMMAND ----------

# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
import fitz  # PyMuPDF for PDF processing
from PIL import Image
import io
import os
from presidio_analyzer import AnalyzerEngine
from presidio_image_redactor import ImageRedactorEngine
import pytesseract  # OCR library for text extraction

# COMMAND ----------

# MAGIC %md
# MAGIC #Redaction and Mapping only

# COMMAND ----------

# Import necessary libraries
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import fitz  # PyMuPDF for PDF processing
from PIL import Image, ImageDraw, ImageFont  # Pillow for image manipulation
import pytesseract  # OCR library for text extraction
import io
import os
from presidio_analyzer import AnalyzerEngine, RecognizerResult

# Global variables for Unity Catalog
catalog_name = "yash_gupta_catalog"
schema_name = "ey_dbs"
volume_name = "dbs_pptx"

# Full file path to the PDF in Unity Catalog Volume
file_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/Project Churches - FINAL Red Flag Report 181121.pdf"

# Function to extract pages as images with enhanced quality
def extract_pdf_pages_as_images(pdf_path, zoom_x=2.0, zoom_y=2.0):
    doc = fitz.open(pdf_path)  # Open the PDF using PyMuPDF (fitz)
    page_images = []
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # Load each page
        mat = fitz.Matrix(zoom_x, zoom_y)  # Set zoom factor for higher resolution
        pix = page.get_pixmap(matrix=mat)  # Convert page to image with zoom
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)  # Convert to PIL Image
        page_images.append(img)
    
    return page_images

# Function to dynamically calculate a font size that fits inside the bounding box
def get_dynamic_font_size(draw, text, box_width, box_height, initial_size=4):
    font_size = initial_size
    font = ImageFont.load_default()  # Use default font
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

    while text_width > box_width or text_height > box_height:
        font_size -= 0.5  # Decrease font size
        if font_size <= 0.5:  # Avoid going too small
            break
        font = ImageFont.load_default()  # Use default font
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    
    return font

# Function to redact PII by overlaying solid black bounding boxes with entity_unique_id
def redact_pii_from_images(images):
    analyzer = AnalyzerEngine()
    redacted_entities_list = []
    schema = StructType([
        StructField("doc_name", StringType(), True),
        StructField("page_num", IntegerType(), True),
        StructField("text", StringType(), True)
    ])

    extracted_text_df = spark.createDataFrame([], schema)
    entity_mapping = {}
    entity_counter = {}

    for i, img in enumerate(images):
        # Use pytesseract to extract text and bounding boxes from the image
        extracted_text = pytesseract.image_to_string(img)
        boxes = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)  # Get bounding box information as a dictionary

        
        # Append extracted text to the table yash_gupta_catalog.ey_dbs.raw_documents
        new_row = spark.createDataFrame([(file_path, i + 1, extracted_text)], schema)
        extracted_text_df = extracted_text_df.union(new_row)
    
        # Analyze the extracted text with Presidio
        results = analyzer.analyze(text=extracted_text, language='en')  # Specify the language

        # Create an ImageDraw object to draw on the image
        draw = ImageDraw.Draw(img)

        # Iterate over the detected PII results and draw bounding boxes with unique ID
        for result in results:
            entity_text = extracted_text[result.start:result.end]  # Extract entity text using start and end positions
            if entity_text not in entity_mapping:
                prefix = result.entity_type[:3].upper()
                entity_counter[prefix] = entity_counter.get(prefix, 0) + 1
                entity_unique_id = f"{prefix}_{entity_counter[prefix]}"
                entity_mapping[entity_text] = entity_unique_id
            else:
                entity_unique_id = entity_mapping[entity_text]

            redacted_entities_list.append((entity_text, result.entity_type, entity_unique_id))

            # Find the bounding box for the entire entity
            for j in range(len(boxes['text'])):
                if boxes['text'][j] == entity_text:
                    x, y, w, h = boxes['left'][j], boxes['top'][j], boxes['width'][j], boxes['height'][j]
                    box_width, box_height = w, h

                    # Draw solid black rectangle as bounding box
                    draw.rectangle([x, y, x + w, y + h], fill="black")

                    # Calculate appropriate font size to fit within the bounding box
                    font = get_dynamic_font_size(draw, entity_unique_id, box_width, box_height)

                    # Overlay entity_unique_id inside the bounding box
                    draw.text((x, y), entity_unique_id, fill="white", font=font)
                    break  # Break after finding the first match to avoid multiple boxes
    
    extracted_text_df.write.format("delta").mode("append").saveAsTable("yash_gupta_catalog.ey_dbs.raw_documents")
    return redacted_entities_list, images

# Function to save redacted images and entities to a Delta table with separate columns
def save_redacted_results(redacted_images, redacted_entities):
    # Create a DataFrame from the redacted entities with separate columns
    entity_df = spark.createDataFrame(redacted_entities, ["entity_text", "entity_type", "entity_unique_id"])

    # Define the table name using the catalog and schema
    table_name = f"{catalog_name}.{schema_name}.entity_mapping_table"
    
    # Write the DataFrame as a Delta table to the specified catalog and schema
    entity_df.write.format("delta").mode("overwrite").saveAsTable(table_name)

    # If the table does not exist yet, this will create it
    spark.sql(f"CREATE TABLE IF NOT EXISTS {table_name} USING DELTA")

    print(f"Data successfully written to {table_name}")

    # Save the redacted images
    output_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/redacted_output_v2"
    os.makedirs(output_path, exist_ok=True)  # Ensure the output directory exists
    for idx, img in enumerate(redacted_images):
        img.save(os.path.join(output_path, f"redacted_page_{idx+1}.png"))

# Main workflow
def main():
    # Step 1: Read PDF from Unity Catalog
    pdf_path = file_path  # DBFS path for Databricks

    # Step 2: Extract pages as images
    page_images = extract_pdf_pages_as_images(pdf_path)

    # Step 3: Redact PII from page images by overlaying bounding boxes with entity_unique_id
    redacted_entities, redacted_images = redact_pii_from_images(page_images)

    # Step 4: Save the redacted entities and images to an output path (Delta table in this case)
    save_redacted_results(redacted_images, redacted_entities)

# Run the main function
main()

# COMMAND ----------

# MAGIC %md
# MAGIC #Redaction & variable overlay

# COMMAND ----------

from PIL import Image
import io
# Define the path to the redacted output images
redacted_output_path = "/Volumes/yash_gupta_catalog/ey_dbs/dbs_pptx/redacted_output_v2/"

# Load the redacted images from the specified path
redacted_images_df = spark.read.format("binaryFile").load(f"{redacted_output_path}/*.png")

# Collect the content of each image and display row by row
redacted_images_content = redacted_images_df.select("content").collect()

for row in redacted_images_content:
    img = Image.open(io.BytesIO(row["content"]))
    display(img)

# COMMAND ----------

# MAGIC %md
# MAGIC # Version 2

# COMMAND ----------

# Import necessary libraries
from pyspark.sql.functions import lit
import fitz  # PyMuPDF for PDF processing
from PIL import Image, ImageDraw, ImageFont  # Pillow for image manipulation
import pytesseract  # OCR library for text extraction
import io
import os
from presidio_analyzer import AnalyzerEngine, RecognizerResult
import re

# Global variables for Unity Catalog
catalog_name = "yash_gupta_catalog"
schema_name = "ey_dbs"
volume_name = "dbs_pptx"

# Full file path to the PDF in Unity Catalog Volume
file_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/Project Churches - FINAL Red Flag Report 181121.pdf"

# Function to extract pages as images with enhanced quality
def extract_pdf_pages_as_images(pdf_path, zoom_x=2.0, zoom_y=2.0):
    doc = fitz.open(pdf_path)  # Open the PDF using PyMuPDF (fitz)
    page_images = []
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # Load each page
        mat = fitz.Matrix(zoom_x, zoom_y)  # Set zoom factor for higher resolution
        pix = page.get_pixmap(matrix=mat)  # Convert page to image with zoom
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)  # Convert to PIL Image
        page_images.append(img)
    
    return page_images

# Function to dynamically calculate a font size that fits inside the bounding box
def get_dynamic_font_size(draw, text, box_width, box_height, initial_size=4):
    font_size = initial_size
    font = ImageFont.load_default()  # Use default font
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

    while text_width > box_width or text_height > box_height:
        font_size -= 0.5  # Decrease font size
        if font_size <= 0.5:  # Avoid going too small
            break
        font = ImageFont.load_default()  # Use default font
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    
    return font

# Function to preprocess image for better OCR accuracy
def preprocess_image(img):
    img = img.convert("L")  # Convert to grayscale
    img = img.point(lambda x: 0 if x < 140 else 255)  # Binarize image
    return img

# Function to redact PII by overlaying solid black bounding boxes with entity_unique_id
def redact_pii_from_images(images):
    analyzer = AnalyzerEngine()
    redacted_entities_list = []
    entity_mapping = {}
    entity_counter = {}

    for i, img in enumerate(images):
        # Preprocess image for better OCR accuracy
        preprocessed_img = preprocess_image(img)
        
        # Use pytesseract to extract text and bounding boxes from the image
        extracted_text = pytesseract.image_to_string(preprocessed_img)
        
        boxes = pytesseract.image_to_data(preprocessed_img, output_type=pytesseract.Output.DICT)  # Get bounding box information as a dictionary

        # Analyze the extracted text with Presidio
        results = analyzer.analyze(text=extracted_text, entities=["PERSON", "LOCATION", "ORGANIZATION"], language='en')  # Specify the entities and language

        # Create an ImageDraw object to draw on the image
        draw = ImageDraw.Draw(img)

        # Iterate over the detected PII results and draw bounding boxes with unique ID
        for result in results:
            entity_text = extracted_text[result.start:result.end]  # Extract entity text using start and end positions
            if entity_text not in entity_mapping:
                prefix = result.entity_type[:3].upper()
                entity_counter[prefix] = entity_counter.get(prefix, 0) + 1
                entity_unique_id = f"{prefix}_{entity_counter[prefix]}"
                entity_mapping[entity_text] = entity_unique_id
            else:
                entity_unique_id = entity_mapping[entity_text]

            redacted_entities_list.append((entity_text, result.entity_type, entity_unique_id))

            # Find the bounding box for the entire entity
            for j in range(len(boxes['text'])):
                if re.search(re.escape(entity_text), boxes['text'][j], re.IGNORECASE):
                    x, y, w, h = boxes['left'][j], boxes['top'][j], boxes['width'][j], boxes['height'][j]
                    box_width, box_height = w, h

                    # Draw solid black rectangle as bounding box
                    draw.rectangle([x, y, x + w, y + h], fill="black")

                    # Calculate appropriate font size to fit within the bounding box
                    font = get_dynamic_font_size(draw, entity_unique_id, box_width, box_height)

                    # Overlay entity_unique_id inside the bounding box
                    draw.text((x, y), entity_unique_id, fill="white", font=font)
                    break  # Break after finding the first match to avoid multiple boxes

    return redacted_entities_list, images

# Function to save redacted images and entities to a Delta table with separate columns
def save_redacted_results(redacted_images, redacted_entities):
    # Create a DataFrame from the redacted entities with separate columns
    entity_df = spark.createDataFrame(redacted_entities, ["entity_text", "entity_type", "entity_unique_id"])

    # Define the table name using the catalog and schema
    table_name = f"{catalog_name}.{schema_name}.entity_mapping_table"
    
    # Write the DataFrame as a Delta table to the specified catalog and schema
    entity_df.write.format("delta").mode("overwrite").saveAsTable(table_name)

    # If the table does not exist yet, this will create it
    spark.sql(f"CREATE TABLE IF NOT EXISTS {table_name} USING DELTA")

    print(f"Data successfully written to {table_name}")

    # Save the redacted images
    output_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/redacted_output_v2"
    os.makedirs(output_path, exist_ok=True)  # Ensure the output directory exists
    for idx, img in enumerate(redacted_images):
        img.save(os.path.join(output_path, f"redacted_page_{idx+1}.png"))

# Main workflow
def main():

    spark.sql("")

    # Step 1: Read PDF from Unity Catalog
    pdf_path = file_path  # DBFS path for Databricks

    # Step 2: Extract pages as images
    page_images = extract_pdf_pages_as_images(pdf_path)

    # Step 3: Redact PII from page images by overlaying bounding boxes with entity_unique_id
    redacted_entities, redacted_images = redact_pii_from_images(page_images)

    # Step 4: Save the redacted entities and images to an output path (Delta table in this case)
    save_redacted_results(redacted_images, redacted_entities)

# Run the main function
main()
