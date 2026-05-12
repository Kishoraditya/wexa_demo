flowchart TD

subgraph group_backend["Backend"]
node_backend_main["App entry<br/>FastAPI startup<br/>[main.py]"]
node_backend_config["Config<br/>settings<br/>[config.py]"]
node_backend_dependencies["Dependencies<br/>DI wiring<br/>[dependencies.py]"]
node_backend_cache[("Cache<br/>[cache.py]")]
node_backend_logging["Logging<br/>observability<br/>[logging.py]"]
node_backend_metrics["Metrics<br/>prometheus<br/>[metrics.py]"]
node_backend_models["Schemas<br/>API models<br/>[schemas.py]"]
node_route_generate["Generate API<br/>route<br/>[generate.py]"]
node_route_ingest["Ingest API<br/>route<br/>[ingest.py]"]
node_route_health["Health API<br/>route<br/>[health.py]"]
node_svc_ingestion["Ingestion<br/>document pipeline<br/>[ingestion.py]"]
node_svc_vector[("Vector store<br/>faiss index<br/>[vector_store.py]")]
node_svc_rag["RAG pipeline<br/>orchestration<br/>[rag_pipeline.py]"]
node_svc_prompts["Prompts<br/>templates<br/>[prompts.py]"]
node_svc_llm["LLM router<br/>model routing<br/>[llm_manager.py]"]
node_svc_guardrails["Guardrails<br/>safety checks<br/>[guardrails.py]"]
end

subgraph group_client["Client"]
node_frontend_app["Streamlit app<br/>ui client<br/>[app.py]"]
end

subgraph group_offline["Offline workflows"]
node_eval_suite["Eval suite<br/>offline QA<br/>[run_ragas.py]"]
node_fine_tune["Fine-tuning<br/>notebook"]
end

subgraph group_docs["Docs"]
node_arch_docs["Architecture docs<br/>design notes<br/>[architecture.mmd]"]
end

node_backend_main-- >| "loads" | node_backend_config
node_backend_main-- >| "wires" | node_backend_dependencies
node_backend_main-- >| "mounts" | node_route_generate
node_backend_main-- >| "mounts" | node_route_ingest
node_backend_main-- >| "mounts" | node_route_health
node_route_generate-- >| "validates" | node_backend_models
node_route_ingest-- >| "validates" | node_backend_models
node_route_generate-- >| "calls" | node_svc_rag
node_route_ingest-- >| "calls" | node_svc_ingestion
node_route_health-- >| "reports" | node_backend_metrics
node_svc_ingestion-- >| "indexes" | node_svc_vector
node_svc_rag-- >| "retrieves" | node_svc_vector
node_svc_rag-- >| "assembles" | node_svc_prompts
node_svc_rag-- >| "generates" | node_svc_llm
node_svc_rag-- >| "checks" | node_svc_guardrails
node_svc_rag-- >| "uses" | node_backend_cache
node_svc_vector-- >| "shares" | node_backend_cache
node_svc_llm-- >| "reads" | node_backend_config
node_svc_guardrails-- >| "reads" | node_backend_config
node_frontend_app-- >| "queries" | node_route_generate
node_frontend_app-- >| "uploads" | node_route_ingest
node_frontend_app-- >| "checks" | node_route_health
node_eval_suite-- >| "benchmarks" | node_svc_rag
node_fine_tune -.->| "produces" | node_svc_llm
node_arch_docs -.->| "describes" | node_backend_main

click node_backend_main "https://github.com/kishoraditya/wexa_demo/blob/main/backend/main.py"
click node_backend_config "https://github.com/kishoraditya/wexa_demo/blob/main/backend/core/config.py"
click node_backend_dependencies "https://github.com/kishoraditya/wexa_demo/blob/main/backend/core/dependencies.py"
click node_backend_cache "https://github.com/kishoraditya/wexa_demo/blob/main/backend/core/cache.py"
click node_backend_logging "https://github.com/kishoraditya/wexa_demo/blob/main/backend/core/logging.py"
click node_backend_metrics "https://github.com/kishoraditya/wexa_demo/blob/main/backend/core/metrics.py"
click node_backend_models "https://github.com/kishoraditya/wexa_demo/blob/main/backend/models/schemas.py"
click node_route_generate "https://github.com/kishoraditya/wexa_demo/blob/main/backend/routes/generate.py"
click node_route_ingest "https://github.com/kishoraditya/wexa_demo/blob/main/backend/routes/ingest.py"
click node_route_health "https://github.com/kishoraditya/wexa_demo/blob/main/backend/routes/health.py"
click node_svc_ingestion "https://github.com/kishoraditya/wexa_demo/blob/main/backend/services/ingestion.py"
click node_svc_vector "https://github.com/kishoraditya/wexa_demo/blob/main/backend/services/vector_store.py"
click node_svc_rag "https://github.com/kishoraditya/wexa_demo/blob/main/backend/services/rag_pipeline.py"
click node_svc_prompts "https://github.com/kishoraditya/wexa_demo/blob/main/backend/services/prompts.py"
click node_svc_llm "https://github.com/kishoraditya/wexa_demo/blob/main/backend/services/llm_manager.py"
click node_svc_guardrails "https://github.com/kishoraditya/wexa_demo/blob/main/backend/services/guardrails.py"
click node_frontend_app "https://github.com/kishoraditya/wexa_demo/blob/main/frontend/app.py"
click node_eval_suite "https://github.com/kishoraditya/wexa_demo/blob/main/eval/run_ragas.py"
click node_fine_tune "https://github.com/kishoraditya/wexa_demo/blob/main/notebooks/fine_tuning_phi3_qlora.ipynb"
click node_arch_docs "https://github.com/kishoraditya/wexa_demo/blob/main/docs/architecture.mmd"

classDef toneNeutral fill: #f8fafc, stroke:#334155, stroke - width: 1.5px, color:#0f172a
classDef toneBlue fill: #dbeafe, stroke:#2563eb, stroke - width: 1.5px, color:#172554
classDef toneAmber fill: #fef3c7, stroke: #d97706, stroke - width: 1.5px, color:#78350f
classDef toneMint fill: #dcfce7, stroke:#16a34a, stroke - width: 1.5px, color:#14532d
classDef toneRose fill: #ffe4e6, stroke: #e11d48, stroke - width: 1.5px, color:#881337
classDef toneIndigo fill: #e0e7ff, stroke:#4f46e5, stroke - width: 1.5px, color:#312e81
classDef toneTeal fill: #ccfbf1, stroke:#0f766e, stroke - width: 1.5px, color:#134e4a
class node_backend_main, node_backend_config, node_backend_dependencies, node_backend_cache, node_backend_logging, node_backend_metrics, node_backend_models, node_route_generate, node_route_ingest, node_route_health, node_svc_ingestion, node_svc_vector, node_svc_rag, node_svc_prompts, node_svc_llm, node_svc_guardrails toneBlue
class node_frontend_app toneAmber
class node_eval_suite,node_fine_tune toneMint
class node_arch_docs toneRose