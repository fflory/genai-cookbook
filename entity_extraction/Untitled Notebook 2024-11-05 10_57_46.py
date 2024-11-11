# Databricks notebook source
import fitz  # PyMuPDF
import numpy as np
import matplotlib.pyplot as plt

# Path to the PDF file
pdf_path = '/Workspace/Users/yash.gupta@databricks.com/Demo Folder/DBS Benchmarks/PDF Table Extraction/Project Churches - FINAL Red Flag Report 181121.pdf'

# Open the PDF file
pdf_document = fitz.open(pdf_path)

# Extract pages as images in full resolution
images = []
for page_num in range(len(pdf_document)):
    page = pdf_document.load_page(page_num)
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))  # Adjust matrix for higher resolution
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    images.append(img)
	
plt.imshow(images[0])
plt.axis('off')
plt.show()
