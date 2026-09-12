# FastAPI host example

This is an application-shaped document-search endpoint. Authentication is a
FastAPI dependency that produces trusted caller context before search. Query
parameters cannot select tenant or resource scope.

```bash
uv sync
uv run uvicorn app:app --reload
curl -H 'x-principal: ana' 'http://127.0.0.1:8000/search?query=rollout%20approval'
curl -X POST -H 'content-type: application/json' -H 'x-principal: ana' \
  http://127.0.0.1:8000/search/follow-up \
  -d '{"query":"release-manager approval"}'
```
