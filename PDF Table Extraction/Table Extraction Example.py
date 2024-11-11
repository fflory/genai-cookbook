# Databricks notebook source
# MAGIC %sh
# MAGIC sudo apt-get update
# MAGIC sudo apt-get install tesseract-ocr

# COMMAND ----------

# MAGIC %pip install --upgrade typing_extensions

# COMMAND ----------

# MAGIC %pip install torch torchvision

# COMMAND ----------

# MAGIC %pip install ultralyticsplus==0.0.28 ultralytics==8.0.43

# COMMAND ----------

# MAGIC %pip install pytesseract transformers

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from ultralyticsplus import YOLO, render_result

# load model
model = YOLO('keremberke/yolov8m-table-extraction')

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# set image
image = './ey_dbs_one_page_pdf.jpg'

# perform inference
results = model.predict(image)

# observe results
print(results[0].boxes)
render = render_result(model=model, image=image, result=results[0])
display(render)

# COMMAND ----------

# MAGIC %pip install pymupdf PyPDF2

# COMMAND ----------

pdf_path = "/Volumes/felixflory/ey_dbs_workshop_2024_10/raw_data/project_churches/Project Churches - FINAL Red Flag Report 181121.pdf"

# COMMAND ----------

import fitz  # PyMuPDF
import numpy as np
import matplotlib.pyplot as plt

# Path to the PDF file
# pdf_path = '/Workspace/Users/yash.gupta@databricks.com/Demo Folder/DBS Benchmarks/PDF Table Extraction/Project Churches - FINAL Red Flag Report 181121.pdf'

# Open the PDF file
pdf_document = fitz.open(pdf_path)

# Extract pages as images in full resolution
images = []
for page_num in range(len(pdf_document)):
    page = pdf_document.load_page(page_num)
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))  # Adjust matrix for higher resolution
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    images.append(img)

# Zoom into the center of the image by 10x
zoom_factor = 3
center_x, center_y = images[0].shape[1] // 2, images[0].shape[0] // 2
half_width, half_height = images[0].shape[1] // (2 * zoom_factor), images[0].shape[0] // (2 * zoom_factor)
zoomed_image = images[0][center_y - half_height:center_y + half_height, center_x - half_width:center_x + half_width]

# Display the zoomed image using matplotlib
plt.imshow(zoomed_image)
plt.axis('off')
plt.show()

# COMMAND ----------

import cv2
import numpy as np

# Load the original image
original_image = cv2.imread('./ey_dbs_one_page_pdf.jpg', cv2.IMREAD_UNCHANGED)

# Get the coordinates of the bounding box
box = results[0].boxes[0].xyxy  # Assuming you want to crop the first detected object
print(box)

# Extract the coordinates of the top-left and bottom-right corners
x1, y1, x2, y2 = map(int, box[0])

# Crop the region of interest (ROI) from the original image
cropped_image = original_image[y1:y2, x1:x2]

# Save or display the cropped image
cv2.imwrite('./cropped_table.jpeg', cropped_image, [cv2.IMWRITE_JPEG_QUALITY, 100])

# COMMAND ----------

# MAGIC %pip install python-doctr

# COMMAND ----------

# MAGIC %pip install pytesseract pillow
# MAGIC
# MAGIC import pytesseract
# MAGIC from PIL import Image
# MAGIC
# MAGIC # Load the image using PIL
# MAGIC image_path = './cropped_table.jpeg'
# MAGIC image = Image.open(image_path)
# MAGIC
# MAGIC # Perform OCR using Tesseract with improved configuration
# MAGIC custom_config = r'--oem 3 --psm 11'  # OEM 3 for LSTM OCR Engine, PSM 11 for sparse text with OSD
# MAGIC ocr_result = pytesseract.image_to_string(image, config=custom_config)
# MAGIC
# MAGIC print(ocr_result)

# COMMAND ----------

import os
import json

# Let's pick the desired backend
# os.environ['USE_TF'] = '1'
os.environ['USE_TORCH'] = '1'

import matplotlib.pyplot as plt

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Read the file
doc = DocumentFile.from_images("./cropped_table.jpeg")
print(f"Number of pages: {len(doc)}")

# Instantiate a pretrained model
predictor = ocr_predictor(pretrained=True)

result = predictor(doc)

# JSON export
json_export = result.export()

# Define a function to remove fields recursively
def remove_fields(obj, fields):
    if isinstance(obj, list):
        for item in obj:
            remove_fields(item, fields)
    elif isinstance(obj, dict):
        for key in list(obj.keys()):
            if key in fields:
                del obj[key]
            else:
                remove_fields(obj[key], fields)

# Function to remove 'geometry' key from 'blocks' and 'lines'
def remove_geometry(data):
    if isinstance(data, list):
        for item in data:
            remove_geometry(item)
    elif isinstance(data, dict):
        if 'geometry' in data:
            del data['geometry']
        for key, value in data.items():
            remove_geometry(value)

# Fields to remove
fields_to_remove = ['confidence', 'page_idx', 'dimensions', 'orientation', 'language', 'artefacts']

# Remove the specified fields
remove_fields(json_export, fields_to_remove)

# Remove 'geometry' from 'blocks' and 'lines'
for page in json_export['pages']:
    for block in page['blocks']:
        if 'geometry' in block:
            del block['geometry']
        for line in block.get('lines', []):
            if 'geometry' in line:
                del line['geometry']

# Convert the modified data back to JSON
modified_json = json.dumps(json_export, separators=(',', ':'))

# Print the modified JSON
print(modified_json)

#save to a file OCR_Result.txt
output_file_path = "OCR_Result.txt"

# Open the file in write mode and write the JSON data
with open(output_file_path, "w") as output_file:
    output_file.write(modified_json)

print(f"Modified JSON data saved to {output_file_path}")

# COMMAND ----------

prompt = f"""Your objective is to analyze the provided data in JSON format.

JSON Data:
{modified_json}

Reply to the user in JSON format, incorporating the key-value pairs
"""

# COMMAND ----------

print(prompt)

# COMMAND ----------

from openai import OpenAI
import os

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
# Alternatively in a Databricks notebook you can use this:
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
#DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url="https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints"
)

chat_completion = client.chat.completions.create(
  messages=[
  {
    "role": "system",
    "content": prompt  # Ensure you define or replace "Your prompt here" with the actual prompt variable or string
  },
  {
    "role": "user",
    "content": "Extract the data in a table format"
  }
  ],
  model="databricks-meta-llama-3-1-405b-instruct",
  max_tokens=4096  # Reduced max_tokens to 4096
)

print(chat_completion.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC Your objective is to analyze the provided data in JSON format.
# MAGIC
# MAGIC JSON Data:
# MAGIC [Document(page_content='{"pages":[{"blocks":[{"objectness_score":0.8377201004071279,"lines":[{"objectness_score":0.79041388630867,"words":[{"value":"Adjusted","geometry":[[0.01045866935483869,0.0],[0.14712229082661288,0.0302734375]],"objectness_score":0.7569854855537415,"crop_orientation":{"value":0}},{"value":"EBITDA","geometry":[[0.16140057963709675,0.0],[0.2827660345262097,0.025390625]],"objectness_score":0.8238422870635986,"crop_orientation":{"value":0}}]},{"objectness_score":0.8216062188148499,"words":[{"value":"urrency","geometry":[[0.014538180443548376,0.048828125],[0.15426143523185482,0.0830078125]],"objectness_score":0.7786068320274353,"crop_orientation":{"value":0}},{"value":"44000","geometry":[[0.16038070186491937,0.048828125],[0.2582889679939516,0.0791015625]],"objectness_score":0.8646056056022644,"crop_orientation":{"value":0}}]},{"objectness_score":0.887136181195577,"words":[{"value":"FY19A","geometry":[[0.638703377016129,0.0498046875],[0.7386513986895161,0.0771484375]],"objectness_score":0.8685144186019897,"crop_orientation":{"value":0}},{"value":"FYZ0A","geometry":[[0.7610887096774194,0.0498046875],[0.8610367313508065,0.0771484375]],"objectness_score":0.9184199571609497,"crop_orientation":{"value":0}},{"value":"FY21A","geometry":[[0.8814342867943548,0.048828125],[0.9844419417842742,0.0791015625]],"objectness_score":0.8744741678237915,"crop_orientation":{"value":0}}]},{"objectness_score":0.8526530563831329,"words":[{"value":"Reported","geometry":[[0.014538180443548376,0.0966796875],[0.15324155745967744,0.126953125]],"objectness_score":0.8155468106269836,"crop_orientation":{"value":0}},{"value":"BITDA","geometry":[[0.16446021295362906,0.09765625],[0.2827660345262097,0.123046875]],"objectness_score":0.8897593021392822,"crop_orientation":{"value":0}}]},{"objectness_score":0.8804900646209717,"words":[{"value":"1,237","geometry":[[0.6560412991431451,0.0966796875],[0.7376315209173387,0.1279296875]],"objectness_score":0.8804900646209717,"crop_orientation":{"value":0}}]},{"objectness_score":0.8776907324790955,"words":[{"value":"1,263","geometry":[[0.7794465095766129,0.0966796875],[0.8589969758064516,0.1279296875]],"objectness_score":0.8776907324790955,"crop_orientation":{"value":0}}]},{"objectness_score":0.8543958067893982,"words":[{"value":"1,143","geometry":[[0.9018318422379032,0.095703125],[0.9824021862399194,0.126953125]],"objectness_score":0.8543958067893982,"crop_orientation":{"value":0}}]},{"objectness_score":0.8435123364130656,"words":[{"value":"Due","geometry":[[0.011478547127016125,0.14453125],[0.07471096900201613,0.171875]],"objectness_score":0.8480840921401978,"crop_orientation":{"value":0}},{"value":"dwgence","geometry":[[0.08592962449596775,0.1455078125],[0.22259324596774194,0.1767578125]],"objectness_score":0.7900530695915222,"crop_orientation":{"value":0}},{"value":"adjustments","geometry":[[0.23585165700604838,0.1484375],[0.4204495337701613,0.1748046875]],"objectness_score":0.8923998475074768,"crop_orientation":{"value":0}}]},{"objectness_score":0.828395140171051,"words":[{"value":"Rleversal","geometry":[[0.014538180443548376,0.1923828125],[0.13794339087701613,0.2216796875]],"objectness_score":0.7894170880317688,"crop_orientation":{"value":0}},{"value":"of","geometry":[[0.1481421685987903,0.1923828125],[0.18179813508064518,0.2216796875]],"objectness_score":0.8232448697090149,"crop_orientation":{"value":0}},{"value":"rent","geometry":[[0.17057947958669356,0.1943359375],[0.2429908014112903,0.2216796875]],"objectness_score":0.8456294536590576,"crop_orientation":{"value":0}},{"value":"AASB","geometry":[[0.25012994581653225,0.1923828125],[0.3480382119455645,0.2236328125]],"objectness_score":0.8692613840103149,"crop_orientation":{"value":0}},{"value":"16)","geometry":[[0.3561972341229839,0.1923828125],[0.4051513671875,0.224609375]],"objectness_score":0.8144229054450989,"crop_orientation":{"value":0}}]},{"objectness_score":0.8073583245277405,"words":[{"value":"66","geometry":[[0.6835779989919355,0.19140625],[0.7376315209173387,0.2216796875]],"objectness_score":0.8073583245277405,"crop_orientation":{"value":0}}]},{"objectness_score":0.8333876132965088,"words":[{"value":"171","geometry":[[0.8049434538810484,0.19140625],[0.8569572202620968,0.220703125]],"objectness_score":0.8333876132965088,"crop_orientation":{"value":0}}]},{"objectness_score":0.8410371541976929,"words":[{"value":"189","geometry":[[0.9293685420866935,0.1923828125],[0.981382308467742,0.2216796875]],"objectness_score":0.8410371541976929,"crop_orientation":{"value":0}}]},{"objectness_score":0.7941164523363113,"words":[{"value":"Transaction","geometry":[[0.012498424899193561,0.2412109375],[0.17873850176411288,0.267578125]],"objectness_score":0.7690224647521973,"crop_orientation":{"value":0}},{"value":"costs","geometry":[[0.19097703503024194,0.2421875],[0.2684877457157258,0.2685546875]],"objectness_score":0.8467686176300049,"crop_orientation":{"value":0}},{"value":"-","geometry":[[0.2725672568044355,0.2509765625],[0.2888853011592742,0.2646484375]],"objectness_score":0.7238874435424805,"crop_orientation":{"value":0}},{"value":"Perks","geometry":[[0.29704432333669356,0.2412109375],[0.3796544228830645,0.267578125]],"objectness_score":0.8367872834205627,"crop_orientation":{"value":0}}]},{"objectness_score":0.7985784411430359,"words":[{"value":"40","geometry":[[0.9436468308971775,0.2412109375],[0.9824021862399194,0.2685546875]],"objectness_score":0.7985784411430359,"crop_orientation":{"value":0}}]},{"objectness_score":0.8249719738960266,"words":[{"value":"ATOcash","geometry":[[0.00943879158266131,0.287109375],[0.15732106854838712,0.3203125]],"objectness_score":0.781613826751709,"crop_orientation":{"value":0}},{"value":"iow","geometry":[[0.16446021295362906,0.2890625],[0.22361312373991937,0.3173828125]],"objectness_score":0.8617035746574402,"crop_orientation":{"value":0}},{"value":"0081","geometry":[[0.23381190146169356,0.2900390625],[0.3113226121471774,0.3173828125]],"objectness_score":0.8315985202789307,"crop_orientation":{"value":0}}]},{"objectness_score":0.8462278246879578,"words":[{"value":"63)","geometry":[[0.8008639427923387,0.2890625],[0.8610367313508065,0.3232421875]],"objectness_score":0.8462278246879578,"crop_orientation":{"value":0}}]},{"objectness_score":0.837712824344635,"words":[{"value":"(38)","geometry":[[0.9242691532258065,0.2890625],[0.9824021862399194,0.322265625]],"objectness_score":0.837712824344635,"crop_orientation":{"value":0}}]},{"objectness_score":0.8379850536584854,"words":[{"value":"Total","geometry":[[0.011478547127016125,0.33984375],[0.0889892578125,0.3681640625]],"objectness_score":0.862191915512085,"crop_orientation":{"value":0}},{"value":"due","geometry":[[0.09714827998991937,0.3388671875],[0.15528131300403225,0.3681640625]],"objectness_score":0.8160064220428467,"crop_orientation":{"value":0}},{"value":"diligence","geometry":[[0.16649996849798387,0.3408203125],[0.30316358996975806,0.3701171875]],"objectness_score":0.8693274855613708,"crop_orientation":{"value":0}},{"value":"adjustments","geometry":[[0.3154021232358871,0.33984375],[0.5030596333165323,0.3701171875]],"objectness_score":0.8044143915176392,"crop_orientation":{"value":0}}]},{"objectness_score":0.8621871471405029,"words":[{"value":"166","geometry":[[0.682558121219758,0.3388671875],[0.7376315209173387,0.3671875]],"objectness_score":0.8621871471405029,"crop_orientation":{"value":0}}]},{"objectness_score":0.8265699148178101,"words":[{"value":"109","geometry":[[0.8049434538810484,0.3388671875],[0.8610367313508065,0.3681640625]],"objectness_score":0.8265699148178101,"crop_orientation":{"value":0}}]},{"objectness_score":0.8124813437461853,"words":[{"value":"191","geometry":[[0.9273287865423387,0.3388671875],[0.9803624306955645,0.3681640625]],"objectness_score":0.8124813437461853,"crop_orientation":{"value":0}}]},{"objectness_score":0.856190025806427,"words":[{"value":"Adjusted","geometry":[[0.013518302671370941,0.3896484375],[0.14712229082661288,0.419921875]],"objectness_score":0.8387911915779114,"crop_orientation":{"value":0}},{"value":"BIIDA","geometry":[[0.1624204574092742,0.390625],[0.2786865234375,0.4130859375]],"objectness_score":0.8735888600349426,"crop_orientation":{"value":0}}]},{"objectness_score":0.8817239999771118,"words":[{"value":"1,403","geometry":[[0.6570611769153226,0.3896484375],[0.7345718876008065,0.4169921875]],"objectness_score":0.8817239999771118,"crop_orientation":{"value":0}}]},{"objectness_score":0.8021516799926758,"words":[{"value":",372","geometry":[[0.7804663873487903,0.3876953125],[0.860016853578629,0.4189453125]],"objectness_score":0.8021516799926758,"crop_orientation":{"value":0}}]},{"objectness_score":0.803520679473877,"words":[{"value":"1,335","geometry":[[0.9018318422379032,0.3876953125],[0.981382308467742,0.419921875]],"objectness_score":0.803520679473877,"crop_orientation":{"value":0}}]},{"objectness_score":0.8384233713150024,"words":[{"value":"Pro","geometry":[[0.011478547127016125,0.4375],[0.06859170236895162,0.4658203125]],"objectness_score":0.7972131371498108,"crop_orientation":{"value":0}},{"value":"forma","geometry":[[0.07675072454637094,0.4384765625],[0.16649996849798387,0.46484375]],"objectness_score":0.8381108045578003,"crop_orientation":{"value":0}},{"value":"adjustments","geometry":[[0.17669874621975806,0.44140625],[0.361296622983871,0.4677734375]],"objectness_score":0.8799461722373962,"crop_orientation":{"value":0}}]},{"objectness_score":0.8163160085678101,"words":[{"value":"FMA","geometry":[[0.013518302671370941,0.484375],[0.0828699911794355,0.51171875]],"objectness_score":0.8462024331092834,"crop_orientation":{"value":0}},{"value":"agreements","geometry":[[0.09000913558467744,0.4873046875],[0.26032872353830644,0.5166015625]],"objectness_score":0.7864295840263367,"crop_orientation":{"value":0}}]},{"objectness_score":0.8583030700683594,"words":[{"value":"(049)","geometry":[[0.6621605657762097,0.4853515625],[0.7355917653729839,0.517578125]],"objectness_score":0.8583030700683594,"crop_orientation":{"value":0}}]},{"objectness_score":0.8354290127754211,"words":[{"value":"851)","geometry":[[0.7845458984375,0.484375],[0.860016853578629,0.517578125]],"objectness_score":0.8354290127754211,"crop_orientation":{"value":0}}]},{"objectness_score":0.8451314568519592,"words":[{"value":"49)","geometry":[[0.9059113533266129,0.4853515625],[0.981382308467742,0.5185546875]],"objectness_score":0.8451314568519592,"crop_orientation":{"value":0}}]},{"objectness_score":0.8365898430347443,"words":[{"value":"Departed","geometry":[[0.014538180443548376,0.533203125],[0.14304277973790325,0.5634765625]],"objectness_score":0.8320962190628052,"crop_orientation":{"value":0}},{"value":"doctors","geometry":[[0.15426143523185482,0.5341796875],[0.26032872353830644,0.560546875]],"objectness_score":0.8410834670066833,"crop_orientation":{"value":0}}]},{"objectness_score":0.8537074327468872,"words":[{"value":"(153)","geometry":[[0.6621605657762097,0.5322265625],[0.7355917653729839,0.5654296875]],"objectness_score":0.8537074327468872,"crop_orientation":{"value":0}}]},{"objectness_score":0.9146141409873962,"words":[{"value":"236)","geometry":[[0.7845458984375,0.5322265625],[0.8589969758064516,0.564453125]],"objectness_score":0.9146141409873962,"crop_orientation":{"value":0}}]},{"objectness_score":0.8304181098937988,"words":[{"value":"(87)","geometry":[[0.9242691532258065,0.53125],[0.981382308467742,0.5654296875]],"objectness_score":0.8304181098937988,"crop_orientation":{"value":0}}]},{"objectness_score":0.8197602033615112,"words":[{"value":"Departed","geometry":[[0.014538180443548376,0.5830078125],[0.14406265751008063,0.6142578125]],"objectness_score":0.8151406645774841,"crop_orientation":{"value":0}},{"value":"Seneral","geometry":[[0.15732106854838712,0.5830078125],[0.26950762348790325,0.6103515625]],"objectness_score":0.8060038089752197,"crop_orientation":{"value":0}},{"value":"Manager","geometry":[[0.28174615675403225,0.583984375],[0.40719112273185487,0.6142578125]],"objectness_score":0.8381361365318298,"crop_orientation":{"value":0}}]},{"objectness_score":0.8204305171966553,"words":[{"value":"44","geometry":[[0.6835779989919355,0.5830078125],[0.7366116431451613,0.611328125]],"objectness_score":0.8204305171966553,"crop_orientation":{"value":0}}]},{"objectness_score":0.8541489839553833,"words":[{"value":"158","geometry":[[0.8059633316532258,0.58203125],[0.860016853578629,0.611328125]],"objectness_score":0.8541489839553833,"crop_orientation":{"value":0}}]},{"objectness_score":0.8486344814300537,"words":[{"value":"157","geometry":[[0.9293685420866935,0.58203125],[0.981382308467742,0.611328125]],"objectness_score":0.8486344814300537,"crop_orientation":{"value":0}}]},{"objectness_score":0.8574857413768768,"words":[{"value":"ayroll","geometry":[[0.014538180443548376,0.6318359375],[0.10836693548387094,0.6630859375]],"objectness_score":0.8700445294380188,"crop_orientation":{"value":0}},{"value":"tax","geometry":[[0.11856571320564518,0.6328125],[0.16548009072580644,0.658203125]],"objectness_score":0.8449269533157349,"crop_orientation":{"value":0}}]},{"objectness_score":0.8444637656211853,"words":[{"value":"(42)","geometry":[[0.6784786101310484,0.630859375],[0.7366116431451613,0.6650390625]],"objectness_score":0.8444637656211853,"crop_orientation":{"value":0}}]},{"objectness_score":0.8009033203125,"words":[{"value":"(52)","geometry":[[0.8008639427923387,0.630859375],[0.860016853578629,0.666015625]],"objectness_score":0.8009033203125,"crop_orientation":{"value":0}}]},{"objectness_score":0.8413497805595398,"words":[{"value":"(59)","geometry":[[0.9252890309979839,0.630859375],[0.9824021862399194,0.6650390625]],"objectness_score":0.8413497805595398,"crop_orientation":{"value":0}}]},{"objectness_score":0.8214812576770782,"words":[{"value":"Telehealth","geometry":[[0.013518302671370941,0.6796875],[0.16140057963709675,0.7060546875]],"objectness_score":0.8244268298149109,"crop_orientation":{"value":0}},{"value":"normalisabon","geometry":[[0.17465899067540325,0.6806640625],[0.3633363785282258,0.7060546875]],"objectness_score":0.8185356855392456,"crop_orientation":{"value":0}}]},{"objectness_score":0.8155419230461121,"words":[{"value":"55","geometry":[[0.6998960433467742,0.6787109375],[0.7376315209173387,0.7080078125]],"objectness_score":0.8155419230461121,"crop_orientation":{"value":0}}]},{"objectness_score":0.8641831874847412,"words":[{"value":"25","geometry":[[0.8222813760080645,0.6787109375],[0.860016853578629,0.7080078125]],"objectness_score":0.8641831874847412,"crop_orientation":{"value":0}}]},{"objectness_score":0.8606162411825997,"words":[{"value":"Admin","geometry":[[0.012498424899193561,0.728515625],[0.10530730216733869,0.75390625]],"objectness_score":0.9010951519012451,"crop_orientation":{"value":0}},{"value":"statt","geometry":[[0.11142656880040325,0.728515625],[0.17873850176411288,0.7548828125]],"objectness_score":0.8885868787765503,"crop_orientation":{"value":0}},{"value":"cost","geometry":[[0.18281801285282256,0.728515625],[0.24809019027217744,0.755859375]],"objectness_score":0.8234288096427917,"crop_orientation":{"value":0}},{"value":"apiitt","geometry":[[0.2521697013608871,0.7294921875],[0.3215213898689516,0.7587890625]],"objectness_score":0.8631957173347473,"crop_orientation":{"value":0}},{"value":"o","geometry":[[0.32866053427419356,0.7294921875],[0.35721711189516125,0.7548828125]],"objectness_score":0.8153596520423889,"crop_orientation":{"value":0}},{"value":"Su","geometry":[[0.36639601184475806,0.728515625],[0.4163700226814516,0.75390625]],"objectness_score":0.8918842673301697,"crop_orientation":{"value":0}},{"value":"es","geometry":[[0.4224892893145161,0.73046875],[0.49694036668346775,0.75390625]],"objectness_score":0.8407632112503052,"crop_orientation":{"value":0}}]},{"objectness_score":0.8280504941940308,"words":[{"value":"1q.","geometry":[[0.6794984879032258,0.7333984375],[0.7355917653729839,0.76171875]],"objectness_score":0.8280504941940308,"crop_orientation":{"value":0}}]},{"objectness_score":0.8441044688224792,"words":[{"value":"n.4","geometry":[[0.803923576108871,0.732421875],[0.8610367313508065,0.7607421875]],"objectness_score":0.8441044688224792,"crop_orientation":{"value":0}}]},{"objectness_score":0.8645005822181702,"words":[{"value":"n.4-","geometry":[[0.9252890309979839,0.7333984375],[0.9824021862399194,0.7607421875]],"objectness_score":0.8645005822181702,"crop_orientation":{"value":0}}]},{"objectness_score":0.8627819418907166,"words":[{"value":"Total","geometry":[[0.011478547127016125,0.7783203125],[0.09000913558467744,0.8056640625]],"objectness_score":0.8443922996520996,"crop_orientation":{"value":0}},{"value":"pro","geometry":[[0.09816815776209675,0.7841796875],[0.1522216796875,0.8125]],"objectness_score":0.8807688355445862,"crop_orientation":{"value":0}},{"value":"orma","geometry":[[0.15936082409274194,0.779296875],[0.25012994581653225,0.8046875]],"objectness_score":0.8947840929031372,"crop_orientation":{"value":0}},{"value":"adjustments","geometry":[[0.259308845766129,0.779296875],[0.44594647807459675,0.80859375]],"objectness_score":0.8311825394630432,"crop_orientation":{"value":0}}]},{"objectness_score":0.8656510710716248,"words":[{"value":"(036)","geometry":[[0.6621605657762097,0.7783203125],[0.7355917653729839,0.8115234375]],"objectness_score":0.8656510710716248,"crop_orientation":{"value":0}}]},{"objectness_score":0.8704141974449158,"words":[{"value":"956)","geometry":[[0.7845458984375,0.7783203125],[0.8589969758064516,0.8115234375]],"objectness_score":0.8704141974449158,"crop_orientation":{"value":0}}]},{"objectness_score":0.8814365267753601,"words":[{"value":"938)","geometry":[[0.9059113533266129,0.7783203125],[0.981382308467742,0.810546875]],"objectness_score":0.8814365267753601,"crop_orientation":{"value":0}}]},{"objectness_score":0.8538105189800262,"words":[{"value":"Pro","geometry":[[0.012498424899193561,0.826171875],[0.06757182459677419,0.85546875]],"objectness_score":0.8339617848396301,"crop_orientation":{"value":0}},{"value":"orma","geometry":[[0.07573084677419356,0.826171875],[0.16446021295362906,0.853515625]],"objectness_score":0.8371180891990662,"crop_orientation":{"value":0}},{"value":"adjusted","geometry":[[0.17465899067540325,0.8271484375],[0.3052033455141129,0.8564453125]],"objectness_score":0.8919616341590881,"crop_orientation":{"value":0}},{"value":"BVDA","geometry":[[0.3184617565524194,0.8271484375],[0.43574770035282256,0.853515625]],"objectness_score":0.8522005677223206,"crop_orientation":{"value":0}}]},{"objectness_score":0.8556296229362488,"words":[{"value":"567","geometry":[[0.682558121219758,0.826171875],[0.7366116431451613,0.85546875]],"objectness_score":0.8556296229362488,"crop_orientation":{"value":0}}]},{"objectness_score":0.824989914894104,"words":[{"value":"414","geometry":[[0.803923576108871,0.8251953125],[0.860016853578629,0.8544921875]],"objectness_score":0.824989914894104,"crop_orientation":{"value":0}}]},{"objectness_score":0.8284939527511597,"words":[{"value":"397","geometry":[[0.9273287865423387,0.8251953125],[0.981382308467742,0.8544921875]],"objectness_score":0.8284939527511597,"crop_orientation":{"value":0}}]},{"objectness_score":0.8015340864658356,"words":[{"value":"other","geometry":[[0.014538180443548376,0.875],[0.10224766885080644,0.9013671875]],"objectness_score":0.8081116080284119,"crop_orientation":{"value":0}},{"value":"onsiderations","geometry":[[0.11142656880040325,0.8759765625],[0.3378394342237903,0.900390625]],"objectness_score":0.7949565649032593,"crop_orientation":{"value":0}}]},{"objectness_score":0.8049264947573344,"words":[{"value":"Net","geometry":[[0.012498424899193561,0.921875],[0.06655194682459675,0.951171875]],"objectness_score":0.8502324819564819,"crop_orientation":{"value":0}},{"value":"MDMi","geometry":[[0.07063145791330644,0.921875],[0.16649996849798387,0.9521484375]],"objectness_score":0.8159777522087097,"crop_orientation":{"value":0}},{"value":"ncomelexpenses","geometry":[[0.1563011907762097,0.923828125],[0.4051513671875,0.953125]],"objectness_score":0.7485692501068115,"crop_orientation":{"value":0}}]},{"objectness_score":0.8449603915214539,"words":[{"value":"(13)","geometry":[[0.8008639427923387,0.921875],[0.8610367313508065,0.9560546875]],"objectness_score":0.8449603915214539,"crop_orientation":{"value":0}}]},{"objectness_score":0.8557474613189697,"words":[{"value":"(38)","geometry":[[0.9252890309979839,0.921875],[0.9824021862399194,0.9560546875]],"objectness_score":0.8557474613189697,"crop_orientation":{"value":0}}]},{"objectness_score":0.7972505688667297,"words":[{"value":"open","geometry":[[0.012498424899193561,0.9716796875],[0.09102901335685482,1.0]],"objectness_score":0.8461841940879822,"crop_orientation":{"value":0}},{"value":"posbion","geometry":[[0.10122779107862906,0.97265625],[0.21341434601814518,0.9990234375]],"objectness_score":0.8544489145278931,"crop_orientation":{"value":0}},{"value":"-","geometry":[[0.21851373487903225,0.98046875],[0.23789141255040325,0.99609375]],"objectness_score":0.745957612991333,"crop_orientation":{"value":0}},{"value":"receptionist","geometry":[[0.24197092363911288,0.970703125],[0.40821100050403225,1.0]],"objectness_score":0.7424115538597107,"crop_orientation":{"value":0}}]},{"objectness_score":0.8015744090080261,"words":[{"value":"Va","geometry":[[0.6917370211693549,0.9716796875],[0.7366116431451613,0.998046875]],"objectness_score":0.8015744090080261,"crop_orientation":{"value":0}}]},{"objectness_score":0.877534031867981,"words":[{"value":"nva","geometry":[[0.8131024760584677,0.9716796875],[0.860016853578629,0.9970703125]],"objectness_score":0.877534031867981,"crop_orientation":{"value":0}}]},{"objectness_score":0.8362718224525452,"words":[{"value":"n.q.","geometry":[[0.9263089087701613,0.9765625],[0.981382308467742,1.0]],"objectness_score":0.8362718224525452,"crop_orientation":{"value":0}}]}]}]}]}', metadata={'source': 'OCR_Result.txt'})]
# MAGIC
# MAGIC Reply to the user in JSON format, incorporating the key-value pairs
