# Databricks notebook source
from typing import List
def install_apt_get_packages(package_list: List[str]):
    """
    Installs apt-get packages required by the parser.

    Parameters:
        package_list (str): A space-separated list of apt-get packages.
    """
    import subprocess

    num_workers = max(
        1, int(spark.conf.get("spark.databricks.clusterUsageTags.clusterWorkers"))
    )
    print(f"number of works: {num_workers}")

    packages_str = " ".join(package_list)
    command = f"sudo rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* && sudo apt-get clean && sudo apt-get update && sudo apt-get install {packages_str} -y"
    subprocess.check_output(command, shell=True)

    def run_command(iterator):
        for x in iterator:
            yield subprocess.check_output(command, shell=True)

    data = spark.sparkContext.parallelize(range(num_workers), num_workers)
    # Use mapPartitions to run command in each partition (worker)
    output = data.mapPartitions(run_command)
    try:
        output.collect()
        print(f"{package_list} libraries installed")
    except Exception as e:
        print(f"Couldn't install {package_list} on all nodes: {e}")
        raise e

# COMMAND ----------

def install_ocr_on_nodes():
    """
    install poppler on the cluster nodes
    """
    # from pyspark.sql import SparkSession
    import subprocess
    num_workers = max(1,int(spark.conf.get("spark.databricks.clusterUsageTags.clusterWorkers")))
    print(f"number of works: {num_workers}")

    command = "sudo rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* && sudo apt-get purge && sudo apt-get clean && sudo apt-get update && sudo apt-get install poppler-utils tesseract-ocr -y" 
    def run_subprocess(command):
        try:
            output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
            return output.decode()
        except subprocess.CalledProcessError as e:
            raise Exception("An error occurred installing OCR libs:"+ e.output.decode())
    #install on the driver
    run_subprocess(command)
    def run_command(iterator):
        for x in iterator:
            yield run_subprocess(command)
    # spark = SparkSession.builder.getOrCreate()
    data = spark.sparkContext.parallelize(range(num_workers), num_workers) 
    # Use mapPartitions to run command in each partition (worker)
    output = data.mapPartitions(run_command)
    try:
        output.collect();
        print("OCR libraries installed")
    except Exception as e:
        print(f"Couldn't install on all node: {e}")
        raise e

# COMMAND ----------

def truncate_text_by_tokens(text, max_tokens=5000, encoding_name="cl100k_base"):
    """truncate text by the token count"""
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text, disallowed_special=())
    truncated_tokens = tokens[:max_tokens]
    truncated_text = encoding.decode(truncated_tokens)
    return truncated_text

def chunk_text_by_size(text, chunk_size=8000, model="cl100k_base", overlap=100, prompt_token_size=100):
    """chunk the text by the token count"""
    encoding = tiktoken.get_encoding(model)
    tokens = encoding.encode(text)
    
    # Text is already within the acceptable range
    if chunk_size > len(tokens) + prompt_token_size:
        return text  

    chunks = []
    start = 0
    
    while start < len(tokens) - prompt_token_size:
        end = start + chunk_size

        if start > 0:
            start = start - overlap
        
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end
    
    return chunks

def index_exists(vsc, endpoint_name, index_full_name):
    """check if a vector search index exists"""
    try:
        vsc.get_index(endpoint_name, index_full_name).describe()
        return True
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print(f'Unexpected error describing the index. This could be a permission issue.')
            raise e
    return False


def display_vs_search(docs, k=3):
    """display the returns from vector search for tables, this assumes the vector search will return page number and table"""
    for d in docs[:k]:
        print(f"Page number: {d[0]} | Table Title: {d[1]}")
        displayHTML(d[2])