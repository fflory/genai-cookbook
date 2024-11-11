# Databricks notebook source
# MAGIC %pip install pandasai openpyxl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import getpass
import openpyxl

# COMMAND ----------

"""Example of using PandasAI with am Excel file."""

import os

from pandasai import Agent

# By default, unless you choose a different LLM, it will use BambooLLM.
# You can get your free API key signing up at https://pandabi.ai (you can also configure it in your .env file)
os.environ["PANDASAI_API_KEY"] = getpass.getpass()

# COMMAND ----------



# COMMAND ----------

agent = Agent(_xlsx_file)

# COMMAND ----------

response = agent.chat("what is the schema in the first spreadsheet")
print(response)

# COMMAND ----------

response = agent.chat("what are the values for the  years provided in the first spreadsheet for the first observation")
print(response)

# COMMAND ----------

import pandas as pd

# Load the Excel file
excel_data = pd.read_excel(_xlsx_file)

# Convert the data to JSON
json_data = excel_data.to_json(orient="records", )

# Print or save the JSON data
print(json_data)


# COMMAND ----------

import pandas as pd

# Load only the first 10 rows of each sheet
excel_data = pd.read_excel(_xlsx_file, sheet_name=None, nrows=10)

# Convert each sheet to JSON format
json_data = {sheet: data.head(10).to_json(orient="records") for sheet, data in excel_data.items()}

# Print or save the JSON data for each sheet
for sheet_name, json_content in json_data.items():
    print(f"Sheet: {sheet_name}")
    print(json_content)
    print("\n")  # Separate output for each sheet

# Optionally, save each sheet's JSON data to separate files
for sheet_name, json_content in json_data.items():
    with open(f"{sheet_name}_output.json", "w") as json_file:
        json_file.write(json_content)


# COMMAND ----------

import json
json.loads(json_content)

# COMMAND ----------

_json = [{'Ad Astra Net assets IFRS 16 effects': 'Net assets',
  'Unnamed: 1': 'Level 2',
  'Unnamed: 2': 'Level 3',
  'Unnamed: 3': 'Level 4',
  'Unnamed: 4': 'Account',
  'in €m': 'FY19'},
 {'Ad Astra Net assets IFRS 16 effects': 'Cash-/ Debt-like items',
  'Unnamed: 1': 'Deferred tax assets',
  'Unnamed: 2': 'Deferred tax assets',
  'Unnamed: 3': 'Deferred tax assets (through profit and loss)',
  'Unnamed: 4': '11422000_DTA - Deferred taxes arising from temporary differences (through profit and loss)',
  'in €m': 0.5582476158},
 {'Ad Astra Net assets IFRS 16 effects': 'Other',
  'Unnamed: 1': 'Lease Liability IFRS 16',
  'Unnamed: 2': 'Lease Liability IFRS 16 (long-term)',
  'Unnamed: 3': 'Lease Liability IFRS 16 (long-term)',
  'Unnamed: 4': '22223000 - Lease Liability IFRS 16 (long-term)',
  'in €m': -76.63518931},
 {'Ad Astra Net assets IFRS 16 effects': 'Other',
  'Unnamed: 1': 'Lease Liability IFRS 16',
  'Unnamed: 2': 'Lease Liability IFRS 16 (short-term)',
  'Unnamed: 3': 'Lease Liability IFRS 16 (short-term)',
  'Unnamed: 4': '23233000 - Lease Liability IFRS 16 (short-term)',
  'in €m': -24.21308719},
 {'Ad Astra Net assets IFRS 16 effects': 'Other',
  'Unnamed: 1': 'Right-of-Use Asset IFRS 16',
  'Unnamed: 2': 'Right-of-Use Asset IFRS 16 - Car Leases',
  'Unnamed: 3': 'Right-of-Use Asset IFRS 16 - Car Leases',
  'Unnamed: 4': '11161000 - Right-of-Use Asset IFRS 16 - Car Leases',
  'in €m': 10.31785323},
 {'Ad Astra Net assets IFRS 16 effects': 'Other',
  'Unnamed: 1': 'Right-of-Use Asset IFRS 16',
  'Unnamed: 2': 'Right-of-Use Asset IFRS 16 - Other Leases',
  'Unnamed: 3': 'Right-of-Use Asset IFRS 16 - Other Leases',
  'Unnamed: 4': '11162000 - Right-of-Use Asset IFRS 16 - Other Leases',
  'in €m': 88.73750246},
 {'Ad Astra Net assets IFRS 16 effects': 'Equity',
  'Unnamed: 1': 'Equity',
  'Unnamed: 2': 'Equity',
  'Unnamed: 3': 'Equity',
  'Unnamed: 4': '21152000_DT - Net profit / net loss',
  'in €m': 1.78296907},
 {'Ad Astra Net assets IFRS 16 effects': 'Equity',
  'Unnamed: 1': 'Equity',
  'Unnamed: 2': 'Equity',
  'Unnamed: 3': 'Equity',
  'Unnamed: 4': '21152000_DTA - Net profit / net loss',
  'in €m': -0.5582476158},
 {'Ad Astra Net assets IFRS 16 effects': 'Equity',
  'Unnamed: 1': 'Equity',
  'Unnamed: 2': 'Equity',
  'Unnamed: 3': 'Equity',
  'Unnamed: 4': '21152000_OCI - Net profit / net loss',
  'in €m': 0.0099514}]


# COMMAND ----------

import pandas as pd

df = pd.DataFrame(_json)
display(df)
