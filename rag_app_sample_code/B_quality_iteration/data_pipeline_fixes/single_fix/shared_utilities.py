# Databricks notebook source
from typing import Dict, Any

def _flatten_nested_params(
    d: Dict[str, Any], parent_key: str = "", sep: str = "/"
) -> Dict[str, str]:
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_nested_params(v, new_key, sep=sep))
        else:
          items[new_key] = v
    return items

def tag_delta_table(table_fqn, config):
    flat_config = _flatten_nested_params(config)
    sqls = []
    for key, item in flat_config.items():
        
        sqls.append(f"""
        ALTER TABLE {table_fqn}
        SET TAGS ("{key.replace("/", "__")}" = "{item}")
        """)
    sqls.append(f"""
        ALTER TABLE {table_fqn}
        SET TAGS ("table_source" = "rag_data_prep")
        """)
    for sql in sqls:
        # print(sql)
        spark.sql(sql)

# COMMAND ----------

def compare_dicts(dict1, dict2):
    differences = []
    for key in dict1.keys():
        if key not in dict2:
            differences.append((key, dict1[key], None))
        elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            nested_differences = compare_dicts(dict1[key], dict2[key])
            differences.extend([(f"{key}.{nested_key}", nested_value1, nested_value2) for nested_key, nested_value1, nested_value2 in nested_differences])
        elif dict1[key] != dict2[key]:
            differences.append((key, dict1[key], dict2[key]))
    for key in dict2.keys():
        if key not in dict1:
            differences.append((key, None, dict2[key]))
    return differences


# COMMAND ----------

import mlflow
def get_or_start_mlflow_run(experiment_name, run_name):
    # Get the POC's data pipeline configuration
    runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string=f"run_name = '{run_name}'", output_format="list")

    if len(runs) == 1:
        return mlflow.start_run(run_id=runs[0].info.run_id)
    elif len(runs) >1:
        raise ValueError("There are multiple runs named {run_name} in the experiment {experiment_name}.  Remove the additional runs or choose a different run name.")
    else:
        return mlflow.start_run(run_name=run_name)

# COMMAND ----------

def append_to_fq_tablename(tablename: str, val: str) -> str:
    """
    Appends a specified value to the table name, handling fully qualified table names.
    Use this function if the fully qualified name has tickmarks around each part. 

    Args:
    tablename (str): The original fully qualified table name.
    val (str): The value to append to the table name.

    Returns:
    str: The modified fully qualified table name with the value appended.

    Example:
    append_to_fq_tablename(destination_tables_config.get("parsed_docs_table_name"), "_quarantine")
    """
    
    _s = tablename.split('.')
    firsts = ".".join(_s[:-1])
    last = _s[-1]
    return ".".join([firsts, f'`{last.replace("`", "")}{val}`'])
