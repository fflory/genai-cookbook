# Databricks notebook source
!pip install pytesseract

# COMMAND ----------

!sudo apt install -y tesseract-ocr

# COMMAND ----------

!sudo apt install -y libtesseract-dev

# COMMAND ----------

from PIL import Image
import pytesseract

print(pytesseract.image_to_string(Image.open('/Volumes/drewfurgiuele/default/pdfs/output/one_page_pdf.png')))

# COMMAND ----------

xml = pytesseract.image_to_alto_xml('/Volumes/drewfurgiuele/default/pdfs/output/one_page_pdf.png')

# COMMAND ----------

xml

# COMMAND ----------

hocr = pytesseract.image_to_pdf_or_hocr('/Volumes/drewfurgiuele/default/pdfs/output/one_page_pdf.png', extension='hocr')

# COMMAND ----------

hocr

# COMMAND ----------

from pytesseract import Output

d = pytesseract.image_to_data('/Volumes/drewfurgiuele/default/pdfs/output/one_page_pdf.png', output_type=Output.DICT)
print(d.keys())

# COMMAND ----------


