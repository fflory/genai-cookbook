# Databricks notebook source
# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION
# MAGIC felixflory.ey_dbs_workshop_2024_10.python_exec (
# MAGIC  code STRING COMMENT 'Python code to execute. Remember to print the final result to stdout.'
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC DETERMINISTIC
# MAGIC COMMENT 'Executes Python code in the sandboxed environment and returns its stdout. The runtime is stateless and you can not read output of the previous tool executions. i.e. No such variables "rows", "observation" defined. Calling another tool inside a Python code is NOT allowed. Use standard python libraries only.'
# MAGIC AS $$
# MAGIC  import sys
# MAGIC  from io import StringIO
# MAGIC  sys_stdout = sys.stdout
# MAGIC  redirected_output = StringIO()
# MAGIC  sys.stdout = redirected_output
# MAGIC  exec(code)
# MAGIC  sys.stdout = sys_stdout
# MAGIC  return redirected_output.getvalue()
# MAGIC $$
# MAGIC

# COMMAND ----------


