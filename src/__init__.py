"""TechGear Solutions LLM Application Package.

Core modules:
- retriever: Hybrid document retrieval (BM25 + Dense Vector + RRF) for unstructured policies (Phase 2).
- sql_tool: Structured SQL querying and validation against techgear.db (Phase 3).
- agent: ReAct / Function-calling router agent combining SQL and RAG capabilities (Phase 3).
- app: FastAPI REST API service for interacting with the assistant (Phase 4).
"""

__version__ = "0.1.0"

