# Databricks notebook source
source_table = "felixflory.agent_demos.customer_orders"
target_table = "felixflory.cookbook_local_test.customer_orders"

df = spark.table(source_table)
df.write.mode("overwrite").saveAsTable(target_table)

# COMMAND ----------

source_table = "felixflory.agent_demos.product_docs"
target_table = "felixflory.cookbook_local_test.product_docs"

df = spark.table(source_table)
df.write.mode("overwrite").saveAsTable(target_table)
