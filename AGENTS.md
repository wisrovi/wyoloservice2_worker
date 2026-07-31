# wyoloservice2_worker — codegraph + graphify

## Sync automático
Antes de analizar o editar código, ejecuta **codegraph-sync**:
- codegraph: índice de símbolos (rápido, MCP siempre disponible)
- graphify: grafo de conocimiento (usa `/graphify` para consultas)

## Estado
- codegraph: ✅ (.codegraph/codegraph.db)
- graphify: ✅ (graphify-out/graph.json + graph.html)

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## 🐳 Docker Deployment & Test Verification Workflow
Every time you modify the source code of this worker (e.g., inside `executor_v2.0` or `app/`):
1. **Build and push the updated image:** You MUST compile the Docker image and push it to Docker Hub under the tag `wisrovi/train_service:worker_executor_v1.0.0` (or the corresponding tag).
2. **Explain the Diagnosis:** Before proposing any pull or test, you MUST explicitly explain to the user what failed, what was fixed, and the technical rationale behind the changes.
3. **Request Explicit Authorization:** Do NOT perform docker pulls, pushes, broadcasts, or test runs automatically. You must ask the user for explicit authorization to execute these commands (e.g. asking "Would you like me to trigger the broadcast update on all nodes?"). Only execute them upon receiving direct confirmation.
