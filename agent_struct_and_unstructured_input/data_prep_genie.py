# Databricks notebook source
# MAGIC %pip install openpyxl
# MAGIC
# MAGIC import pandas as pd
# MAGIC import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %run ../rag_app_sample_code/A_POC_app/pdf_uc_volume/00_config

# COMMAND ----------

_xlsx_path = [
  "/Volumes/felixflory/ey_dbs_workshop_2024_10/raw_data/project_ad_astra/1.6.3.1 Ad Astra - Net assets details - 02.11.2020.xlsx",
  "/Volumes/felixflory/ey_dbs_workshop_2024_10/raw_data/project_ad_astra/1.6.3.2 Ad Astra - PL details - 30.10.2020.xlsx"]

# COMMAND ----------

ad_astra_xlsx = {}
for p in _xlsx_path:
  ad_astra_xlsx[p.split("/")[-1]] = pd.read_excel(p, header=1, sheet_name=None)

# COMMAND ----------

ad_astra_xlsx.items()

# COMMAND ----------

def prep_colum_name(col_name):
  return (col_name
          .replace(" ", "_").replace(".", "_").replace("-", "_").replace("/", "_")
          .replace("(", "_").replace(")", "_").replace(":", "_").replace("'", "_")
          .replace(",", "_").replace(";", "_").replace("!", "_").replace("&", "_"))

# COMMAND ----------

for f,x in ad_astra_xlsx.items():
  for s,df in x.items():
    df_name = prep_colum_name(f"ad_astra_{f}_{s}")
    print("\n", df_name)
    print("\t", df.columns)
    df = spark.createDataFrame(ad_astra_xlsx[f][s])
    # print(f"{UC_CATALOG}.{UC_SCHEMA}.{df_name}")
    (df.toDF(*[c.lower().replace(" ", "_") for c in df.columns])
     .write.mode("overwrite").format("delta")
     .saveAsTable(f"{UC_CATALOG}.{UC_SCHEMA}.{df_name}"))
