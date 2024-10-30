# Databricks notebook source
dbutils.widgets.text("file_path", "", "Enter text here")

# COMMAND ----------

file_path = dbutils.widgets.get("file_path")

# COMMAND ----------

# MAGIC %pip install presidio_analyzer
# MAGIC %pip install presidio_anonymizer
# MAGIC %pip install pdfminer.six
# MAGIC %pip install pikepdf
# MAGIC %pip install spacy
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %sh python -m spacy download en_core_web_sm

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook demonstrates the use of various libraries for analyzing and anonymizing text within PDF documents.
# MAGIC
# MAGIC <h2>Libraries Used:</h2>
# MAGIC
# MAGIC 1. presidio_analyzer: A library for detecting sensitive information (PII) in text.
# MAGIC 2. presidio_anonymizer: A library for anonymizing detected sensitive information.
# MAGIC 3. pdfminer.six: A library for extracting text from PDF files.
# MAGIC 4. pikepdf: A library for manipulating PDF files.
# MAGIC
# MAGIC <h2>Steps:</h2>
# MAGIC
# MAGIC 1. Extract text from PDF using pdfminer.six.
# MAGIC 2. Analyze the extracted text for sensitive information using presidio_analyzer.
# MAGIC 3. Anonymize the detected sensitive information using presidio_anonymizer.
# MAGIC 4. Update the PDF with anonymized text using pikepdf.
# MAGIC
# MAGIC <h2>Details:</h2>
# MAGIC
# MAGIC - presidio_analyzer: Provides the AnalyzerEngine class to detect PII entities in text.
# MAGIC - presidio_anonymizer: Provides the AnonymizerEngine class to anonymize detected PII entities.
# MAGIC - pdfminer.six: Provides functions like extract_text and extract_pages to extract text and layout information from PDF files.
# MAGIC - pikepdf: Provides classes like Pdf to read and manipulate PDF files.
# MAGIC
# MAGIC <h2>Usage:</h2>
# MAGIC
# MAGIC - Ensure the required libraries are installed using %pip install.
# MAGIC - Set the file_path widget to the path of the PDF file to be processed.
# MAGIC - Run the cells in sequence to extract, analyze, anonymize, and update the PDF.
# MAGIC """

# COMMAND ----------

# For Presidio
from presidio_analyzer import AnalyzerEngine, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# For console output
from pprint import pprint

# For extracting text
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTTextLine

# For updating the PDF
from pikepdf import Pdf, AttachedFileSpec, Name, Dictionary, Array

# COMMAND ----------

# MAGIC %md
# MAGIC **AnalyzerEngine**: Presidio's AnalyzerEngine is used to detect sensitive information (PII) in text. It uses predefined recognizers and can be extended with custom recognizers.
# MAGIC
# MAGIC Documentation: https://microsoft.github.io/presidio/analyzer/
# MAGIC
# MAGIC **LTTextContainer**: Part of pdfminer.six, LTTextContainer represents a container of text in a PDF.
# MAGIC It can contain multiple lines of text and is used to extract and analyze text from PDF documents.
# MAGIC
# MAGIC Documentation: https://pdfminersix.readthedocs.io/en/latest/api/layout.html#pdfminer.layout.LTTextContainer
# MAGIC
# MAGIC <h3>Steps</h3>
# MAGIC
# MAGIC 1. Initialize the AnalyzerEngine.
# MAGIC 2. Iterate through each page of the PDF and extract text containers.
# MAGIC 3. For each text container, extract the text and analyze it using the AnalyzerEngine.
# MAGIC 4. Print the text and analysis results if the text is not just whitespace.
# MAGIC 5. Collect individual characters from the text container.
# MAGIC 6. Slice out characters that match the analysis results and store them in a list.

# COMMAND ----------

analyzer = AnalyzerEngine()

analyzed_character_sets = []

for page_layout in extract_pages(file_path):
    for text_container in page_layout:
        if isinstance(text_container, LTTextContainer):

            # The element is a LTTextContainer, containing a paragraph of text.
            text_to_anonymize = text_container.get_text()

            # Analyze the text using the analyzer engine
            analyzer_results = analyzer.analyze(text=text_to_anonymize, language='en')
 
            if text_to_anonymize.isspace() == False:
                print(text_to_anonymize)
                print(analyzer_results)

            characters = list([])

            # Grab the characters from the PDF
            for text_line in filter(lambda t: isinstance(t, LTTextLine), text_container):
                    for character in filter(lambda t: isinstance(t, LTChar), text_line):
                            characters.append(character)


            # Slice out the characters that match the analyzer results.
            for result in analyzer_results:
                start = result.start
                end = result.end
                analyzed_character_sets.append({"characters": characters[start:end], "result": result})

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Next, we define a function to combine bounding boxes and processes the analyzed character sets to create bounding boxes for each set.
# MAGIC
# MAGIC ### Function: combine_rect
# MAGIC The `combine_rect` function takes two bounding boxes (rectA and rectB) and combines them into a single bounding box that encompasses both.
# MAGIC - **Parameters:**
# MAGIC   - `rectA`: A tuple representing the first bounding box (startX, startY, endX, endY).
# MAGIC   - `rectB`: A tuple representing the second bounding box (startX, startY, endX, endY).
# MAGIC - **Returns:**
# MAGIC   - A tuple representing the combined bounding box.
# MAGIC
# MAGIC ### Process:
# MAGIC 1. **Initialization:**
# MAGIC    - An empty list `analyzed_bounding_boxes` is initialized to store the final bounding boxes and their corresponding analysis results.
# MAGIC
# MAGIC 2. **Combining Bounding Boxes:**
# MAGIC    - For each character set in `analyzed_character_sets`, the bounding boxes of individual characters are combined into a single bounding box.
# MAGIC    - The first character's bounding box is used as the initial `completeBoundingBox`.
# MAGIC    - The `combine_rect` function is called iteratively to expand the `completeBoundingBox` to include each character's bounding box.
# MAGIC
# MAGIC 3. **Storing Results:**
# MAGIC    - The combined bounding box and its corresponding analysis result are appended to the `analyzed_bounding_boxes` list.
# MAGIC
# MAGIC ### Example Usage:
# MAGIC This cell is typically used after analyzing text from a PDF and extracting character-level bounding boxes. The combined bounding boxes can then be used for further processing, such as creating annotations in the PDF.

# COMMAND ----------

# Combine the bounding boxes into a single bounding box.
def combine_rect(rectA, rectB):
    a, b = rectA, rectB
    startX = min( a[0], b[0] )
    startY = min( a[1], b[1] )
    endX = max( a[2], b[2] )
    endY = max( a[3], b[3] )
    return (startX, startY, endX, endY)

analyzed_bounding_boxes = []

# For each character set, combine the bounding boxes into a single bounding box.
for analyzed_character_set in analyzed_character_sets:
    completeBoundingBox = analyzed_character_set["characters"][0].bbox
    
    for character in analyzed_character_set["characters"]:
        completeBoundingBox = combine_rect(completeBoundingBox, character.bbox)
    
    analyzed_bounding_boxes.append({"boundingBox": completeBoundingBox, "result": analyzed_character_set["result"]})


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Next steps
# MAGIC
# MAGIC 1. **Open the PDF File**:
# MAGIC    - The `Pdf.open(file_path)` function is used to open the PDF file specified by `file_path`.
# MAGIC
# MAGIC 2. **Initialize Annotations List**:
# MAGIC    - An empty list `annotations` is created to store the highlight annotations.
# MAGIC
# MAGIC 3. **Create Highlight Annotations**:
# MAGIC    - For each bounding box in `analyzed_bounding_boxes`, a highlight annotation is created.
# MAGIC    - The bounding box coordinates are extracted from `analyzed_bounding_box["boundingBox"]`.
# MAGIC    - A `Dictionary` object is created to define the highlight annotation with the following properties:
# MAGIC      - `Type`: Annotation type.
# MAGIC      - `Subtype`: Highlight subtype.
# MAGIC      - `QuadPoints`: Coordinates for the quadrilateral points of the highlight.
# MAGIC      - `Rect`: Rectangle coordinates for the highlight.
# MAGIC      - `C`: Color of the highlight (red in this case).
# MAGIC      - `CA`: Opacity of the highlight.
# MAGIC      - `T`: Entity type from the analysis result.
# MAGIC
# MAGIC 4. **Add Annotations to the PDF**:
# MAGIC    - The annotations are added to the first page of the PDF using `pdf.pages[0].Annots = pdf.make_indirect(annotations)`.
# MAGIC
# MAGIC 5. **Save the Annotated PDF**:
# MAGIC    - The annotated PDF is saved to the specified path `./sample_data/annotated.pdf` using `pdf.save("./sample_data/annotated.pdf")`.

# COMMAND ----------

pdf = Pdf.open(file_path)

annotations = []

# Create a highlight annotation for each bounding box.
for analyzed_bounding_box in analyzed_bounding_boxes:

    boundingBox = analyzed_bounding_box["boundingBox"]

    # Create the annotation. 
    # We could also create a redaction annotation if the ongoing workflows supports them.
    highlight = Dictionary(
        Type=Name.Annot,
        Subtype=Name.Highlight,
        QuadPoints=[boundingBox[0], boundingBox[3],
                    boundingBox[2], boundingBox[3],
                    boundingBox[0], boundingBox[1],
                    boundingBox[2], boundingBox[1]],
        Rect=[boundingBox[0], boundingBox[1], boundingBox[2], boundingBox[3]],
        C=[1, 0, 0],
        CA=0.5,
        T=analyzed_bounding_box["result"].entity_type,
    )
    
    annotations.append(highlight)

# Add the annotations to the PDF.
pdf.pages[0].Annots = pdf.make_indirect(annotations)

# And save.
pdf.save("./sample_data/annotated.pdf")
