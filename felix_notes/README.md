search for deployements (agents)
```console
from databricks import agents
# UC_MODEL_NAME = "tjcycyota_catalog.rag_tjcycyota.tjcycyota_agent_quick_start"
_d = agents.get_deployments(UC_MODEL_NAME)
_d[0].review_app_url, _d[0].endpoint_url
_l = agents.list_deployments()
[d.model_name for d in _l if "felix" in d.model_name]
# agents.delete_deployment(UC_MODEL_NAME)
```

I deleted the deployment and the inference tables on e2-demo 