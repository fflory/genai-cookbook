# Databricks notebook source
# MAGIC %pip install openpyxl
# MAGIC # %pip install openai
# MAGIC %restart_python

# COMMAND ----------

import openai
import pandas as pd
import json
import math

# COMMAND ----------

_xlsx_file = "/Volumes/felixflory/ey_dbs_workshop_2024_10/raw_data/project_ad_astra/1.6.3.1 Ad Astra - Net assets details - 02.11.2020.xlsx"

# COMMAND ----------

df = pd.read_excel(_xlsx_file, header=None, dtype=str, nrows=5, sheet_name="Trial balance Net assets")

# COMMAND ----------

# Convert DataFrame to JSON
table_json = {
    "header": df.columns.tolist(),
    "rows": df.values.tolist()
}

# Save to a JSON file
# with open('table.json', 'w') as f:
#     json.dump(table_json, f)


# COMMAND ----------

table_json

# COMMAND ----------

all(isinstance(x, str) or x is None or (isinstance(x, float) and math.isnan(x)) for x in lst)
isinstance(val, str) or val is None or (isinstance(val, float) and math.isnan(val))

# COMMAND ----------

import pandas as pd

def read_excel_with_header_detection(file_path):
    # Read the first two rows to check if both rows contain column headers
    first_two_rows = pd.read_excel(file_path, header=None, nrows=2)
    
    # Function to check if a row looks like column headers
    def is_header_row(row):
        # Checking if all values in the row are strings (simplified heuristic)
        return all(isinstance(val, str) or val is None or (isinstance(val, float) and math.isnan(val)) for val in row)
    
    # Check if both the first and second rows could be headers
    first_row_header = is_header_row(first_two_rows.iloc[0])
    second_row_header = is_header_row(first_two_rows.iloc[1])
    
    # Decide which row to use as the header
    if first_row_header and second_row_header:
        # If both rows look like headers, use the second row as header
        header_row = 1
    elif first_row_header:
        # Only the first row looks like headers
        header_row = 0
    else:
        # Default to the first row if no clear headers are found
        header_row = 0
    
    # Read the file with the determined header row
    df = pd.read_excel(file_path, header=header_row)
    
    return df

# Example usage
file_path = _xlsx_file  # Replace with your file path
data = read_excel_with_header_detection(file_path)
print(data.head())


# COMMAND ----------

import pandas as pd

# Read Excel file, ignoring the first row
df = pd.read_excel(_xlsx_file, sheet_name='Trial balance Net assets', skiprows=1)

# COMMAND ----------

df

# COMMAND ----------

import json

# Convert DataFrame to JSON
table_json = {
    "header": df.columns.tolist(),
    "rows": df.values.tolist()
}

# Save to a JSON file
with open('table.json', 'w') as f:
    json.dump(table_json, f)


# COMMAND ----------

from transformers import TapasTokenizer, TapasForQuestionAnswering
import torch

# Load pre-trained TAPAS model and tokenizer
tokenizer = TapasTokenizer.from_pretrained('google/tapas-base-finetuned-wtq')
model = TapasForQuestionAnswering.from_pretrained('google/tapas-base-finetuned-wtq')

# Define questions
questions = [
    "What is the total revenue?",
    "Which product has the highest sales?"
]

# Tokenize inputs
inputs = tokenizer(table=df, queries=questions, padding='max_length', return_tensors="pt")

# Get model outputs
outputs = model(**inputs)

# Process model outputs to get answers
answers = tokenizer.convert_logits_to_predictions(inputs, outputs.logits)


# COMMAND ----------

from transformers import AutoModelForTableQuestionAnswering, AutoTokenizer

# COMMAND ----------

import pandas as pd

from transformers import AutoModelForTableQuestionAnswering, AutoTokenizer



# Load Excel data

df = pd.read_excel(_xlsx_file)

table_data = df.values.tolist()  # Convert to list of lists



# Load TAPAS model

model_name = "google/tapas-base"

model = AutoModelForTableQuestionAnswering.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)



# Ask a question

question = "What is the total sales in the East region?"



# Encode the question and table

encoded_inputs = tokenizer(text=question, table=df, return_tensors="pt")



# Get the answer

output = model(**encoded_inputs)

answer = tokenizer.decode(output.logits.argmax(-1), skip_special_tokens=True)



print(answer)  # Will print the answer based on the table data


# COMMAND ----------

import os

os.environ['OPENAI_API_KEY'] = dbutils.secrets.get(scope='felix-flory', key='oai')

# COMMAND ----------

df

# COMMAND ----------

import pandas as pd

# Read Excel file
df = pd.read_excel(_xlsx_file, sheet_name='Trial balance Net assets', nrows=6) # , skiprows=1, 

# Convert DataFrame to JSON format (or string format suitable for input to GPT)
table_json = df.to_dict(orient='records')


# COMMAND ----------

# get Databricks credentials

from mlflow.utils.databricks_utils import get_databricks_host_creds

creds = get_databricks_host_creds()

creds.token, creds.host

# use openai's client to make calls to Databricks Foundation Model API
import openai
client = openai.OpenAI(
    api_key=creds.token,
    base_url=creds.host + '/serving-endpoints',
)

# COMMAND ----------

from openai import OpenAI
import os

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
# DATABRICKS_TOKEN = creds.host
# Alternatively in a Databricks notebook you can use this:
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url="https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints"
)

# COMMAND ----------

prompt = "Generate a summary from the following table:\n\nColumns: Date, Sales, Product\nData:\n" + str(table_json)
prompt = "what are the column names in the following table. answer by giving the column names as a json formatted list. Don't return anything else Table:\n" + str(table_json)
prompt= "convert the following table into a table with the following columns 'Net assets', 'Level 2', 'Level 3', 'Level 4', 'Account', 'FY17', 'FY18', 'FY19'. please keep the output format in json. Only return the json string an nothing else. Table: "  + str(table_json)
prompt = "what is the schema of the following table Table: " + str(table_json)
prompt = f"Identify the schema of the following table.  write a python program that reads in the table as a pandas dataframe assuming the same table structure exists in the excel file {_xlsx_file}. Table: " + str(table_json)
# prompt = f"Identify the schema of the following table. Table: " + str(table_json)
chat_completion = client.chat.completions.create(
  messages=[
  {
    "role": "system",
    "content": "You are an AI assistant"
  },
  {
    "role": "user",
    "content": prompt
  }
  ],
  model="Yash_GPT_4o",
  max_tokens=756
)

# print(chat_completion.choices[0].message.content)

# COMMAND ----------

import pandas as pd

# Path to the Excel file
file_path = '/Volumes/felixflory/ey_dbs_workshop_2024_10/raw_data/project_ad_astra/1.6.3.1 Ad Astra - Net assets details - 02.11.2020.xlsx'

# Read the Excel file, specifying the header rows
df = pd.read_excel(file_path, header=[1])  # Assuming the second row is the header

# Display the dataframe
print(df)

# Change the column names if yed have blank headers from original
df.columns = ['Net assets', 'Level 2', 'Level 3', 'Level 4', 'Account', 'FY17', 'FY18', 'FY19']

# Convert relevant columns to numeric, handling any errors if needed
df['FY17'] = pd.to_numeric(df['FY17'], errors='coerce')
df['FY18'] = pd.to_numeric(df['FY18'], errors='coerce')
df['FY19'] = pd.to_numeric(df['FY19'], errors='coerce')

# Show the final cleaned dataframe
print(df)

# COMMAND ----------

df

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

import pandas as pd

# Specify the path to the Excel file
file_path = '/Volumes/felixflory/ey_dbs_workshop_2024_10/raw_data/project_ad_astra/1.6.3.1 Ad Astra - Net assets details - 02.11.2020.xlsx'

# Load the Excel file
xls = pd.ExcelFile(file_path)

# If the data is on the first sheet, you can read it directly like this
# Adjust the sheet name/index as necessary
pdf = pd.read_excel(xls, sheet_name=0, header=[0, 1])

# Rename the columns to handle multi-level columns
# The first level is merged from the two header rows
pdf.columns = [' '.join(col).strip() for col in pdf.columns.values]

# Display the DataFrame
print(pdf)

# COMMAND ----------

pdf.head()

# COMMAND ----------

# import json
# import re
# # Extract the json array from the text, removing potential noise
# def extract_json_array(text):
#     # Use regex to find a JSON array within the text
#     match = re.search(r'(\[.*?\])', text)
#     if match:
#         try:
#             parsed = json.loads(match.group(0))
#             if isinstance(parsed, list):
#                 return parsed
#         except json.JSONDecodeError:
#             pass
#     return []

# COMMAND ----------

res = chat_completion.choices[0].message.content

# COMMAND ----------

print(res)

# COMMAND ----------

res = res.strip('```json').strip('```')

# COMMAND ----------

print(res)

# COMMAND ----------

import json
import pandas as pd

# Assuming 'res' contains the JSON data as a string
json_data = json.loads(res)

# Convert JSON data to a pandas DataFrame
df_res = pd.DataFrame(json_data)

# Display the DataFrame
display(df_res)

# COMMAND ----------

import openai
import pandas as pd
# Replace with your OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']
# Load the Excel file
file_path = _xlsx_file
df = pd.read_excel(file_path, skiprows=1)
# Define a function to call the OpenAI API
def generate_summary(text):
 response = openai.Completion.create(
 engine="text-davinci-003",
 prompt=f"Summarize the following data:\n{text}",
 max_tokens=100
 )
 return response.choices[0].text.strip()
# Apply the function to a specific column (e.g., 'Description')
df['Summary'] = df['FY19'].apply(generate_summary)
# Save the updated Excel file
df.to_excel(_xlsx_file, index=False)
print("Summary added to the Excel file successfully!")

# COMMAND ----------

response = openai.Completion.create(
 engine="text-davinci-003",
 prompt=f"Summarize the following data:\n FY19",
 max_tokens=100
 )

# COMMAND ----------

response = openai.ChatCompletion.create(
  model="text-davinci-003",
  messages=[
    {"role": "system", "content": "Summarize the following data:"},
    {"role": "user", "content": "FY19"}
  ],
  max_tokens=100
)

# COMMAND ----------

from transformers import TapasTokenizer
import pandas as pd

model_name = "google/tapas-base"
tokenizer = TapasTokenizer.from_pretrained(model_name)

data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
queries = [
    "What is the name of the first actor?",
    "How many movies has George Clooney played in?",
    "What is the total number of movies?",
]
answer_coordinates = [[(0, 0)], [(2, 1)], [(0, 1), (1, 1), (2, 1)]]
answer_text = [["Brad Pitt"], ["69"], ["209"]]
table = pd.DataFrame.from_dict(data)
inputs = tokenizer(
    table=table,
    queries=queries,
    answer_coordinates=answer_coordinates,
    answer_text=answer_text,
    padding="max_length",
    return_tensors="pt",
)
inputs

# COMMAND ----------

from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd

model_name = "google/tapas-base-finetuned-wtq"
model = TapasForQuestionAnswering.from_pretrained(model_name)
tokenizer = TapasTokenizer.from_pretrained(model_name)

data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
queries = [
    "What is the name of the first actor?",
    "How many movies has George Clooney played in?",
    "What is the total number of movies?",
]
table = pd.DataFrame.from_dict(data)
inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
outputs = model(**inputs)
predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
    inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
)

# let's print out the results:
id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

answers = []
for coordinates in predicted_answer_coordinates:
    if len(coordinates) == 1:
        # only a single cell:
        answers.append(table.iat[coordinates[0]])
    else:
        # multiple cells
        cell_values = []
        for coordinate in coordinates:
            cell_values.append(table.iat[coordinate])
        answers.append(", ".join(cell_values))

display(table)
print("")
for query, answer, predicted_agg in zip(queries, answers, aggregation_predictions_string):
    print(query)
    if predicted_agg == "NONE":
        print("Predicted answer: " + answer)
    else:
        print("Predicted answer: " + predicted_agg + " > " + answer)

# COMMAND ----------

table = table.astype(str)


# COMMAND ----------

print(table.dtypes)

# COMMAND ----------

table = pd.read_excel(_xlsx_file, skiprows=1).head(5)

# COMMAND ----------

from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd

model_name = "google/tapas-base-finetuned-wtq"
model = TapasForQuestionAnswering.from_pretrained(model_name)
tokenizer = TapasTokenizer.from_pretrained(model_name)

# COMMAND ----------

# data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
queries = [
    "What is the aggregated sum of the FY19 values?",
    "What are the column names of the table? ",
    # "What is the total number of movies?",
]
table = pd.read_excel(_xlsx_file, skiprows=1, )
table = table.astype(str)
inputs = tokenizer(table=table, queries=queries, return_tensors="pt", padding=True, truncation=True) # padding="max_length", 
outputs = model(**inputs)


# COMMAND ----------

predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
    inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
)

# COMMAND ----------

predicted_answer_coordinates

# COMMAND ----------

predicted_answer_coordinates[1]

# COMMAND ----------

predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
    inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
)

# let's print out the results:
id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

answers = []
for coordinates in predicted_answer_coordinates:
    if len(coordinates) == 1:
        # only a single cell:
        answers.append(table.iat[coordinates[0]])
    else:
        # multiple cells
        cell_values = []
        for coordinate in coordinates:
            cell_values.append(table.iat[coordinate])
        answers.append(", ".join(cell_values))

display(table)
print("")
for query, answer, predicted_agg in zip(queries, answers, aggregation_predictions_string):
    print(query)
    if predicted_agg == "NONE":
        print("Predicted answer: " + answer)
    else:
        print("Predicted answer: " + predicted_agg + " > " + answer)

# COMMAND ----------

answers
