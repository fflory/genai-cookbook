# Setup

configs similar to the databricks_connect repo

# Update the current conda env

```bash
conda env update --prefix /Users/Felix.Flory/git/genai-cookbook/.conda --file environment.yml --prune
```

# databricks-connect

validate the authentication
```bash
databricks auth profiles
```

# Get a spark session

```python
# COMMAND ----------

try:
    print(spark.version)
except NameError:
    print("No SparkSession found, creating a new one...")
    from databricks.connect import DatabricksSession
    
    # uses DATABRICKS_CONFIG_PROFILE environment
    spark = DatabricksSession.builder.validateSession(True).getOrCreate()

# COMMAND ----------
```

# start with [agent_app_sample_code/00_global_config.py](agent_app_sample_code/00_global_config.py)