"""
backend/core/config.py

Configuration Loader
====================
Reads config.yaml and exposes a typed config object.
All modules import get_config() — no module reads config.yaml directly.

Why a typed config object rather than a raw dict?
  - IDE autocomplete works on config.ingestion.chunking.chunk_size
  - Typos in config keys raise AttributeError at startup, not at runtime
    during a user request
  - The config is validated once at startup, not on every access
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ChunkingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 50
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", " "])


@dataclass
class IngestionConfig:
    data_dir: str = "data/pdfs"
    supported_formats: list[str] = field(default_factory=lambda: ["pdf", "md", "txt"])
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    deduplication_enabled: bool = True
    hash_algorithm: str = "sha256"


@dataclass
class EmbeddingConfig:
    model: str = "BAAI/bge-small-en-v1.5"
    dimensions: int = 384
    batch_size: int = 100
    cache_enabled: bool = True
    cache_dir: str = ".cache/embeddings"


@dataclass
class RetrievalConfig:
    top_k: int = 5
    score_threshold: float = 0.70
    search_type: str = "similarity"


@dataclass
class VectorStoreConfig:
    provider: str = "faiss"
    index_name: str = "aws-well-architected"
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


@dataclass
class RerankerConfig:
    enabled: bool = True
    model: str = "BAAI/bge-reranker-base"
    top_n: int = 5


@dataclass
class GenerationConfig:
    temperature: float = 0.1
    max_new_tokens: int = 512
    timeout_seconds: int = 30


@dataclass
class PrimaryModelConfig:
    type: str = "huggingface_peft"
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    adapter_repo: str = "kishoraditya/enterprise-rag-adapter"
    load_in_4bit: bool = True
    device: str = "auto"


@dataclass
class FallbackModelConfig:
    type: str = "openai"
    model: str = "gpt-4o-mini"
    max_tokens: int = 512
    temperature: float = 0.1


@dataclass
class ModelsConfig:
    primary: PrimaryModelConfig = field(default_factory=PrimaryModelConfig)
    fallback: FallbackModelConfig = field(default_factory=FallbackModelConfig)


@dataclass
class GuardrailsInputConfig:
    max_query_length: int = 1000
    injection_detection: bool = True


@dataclass
class GuardrailsOutputConfig:
    grounding_check: bool = True
    grounding_threshold: float = 0.50
    pii_redaction: bool = True


@dataclass
class GuardrailsConfig:
    input: GuardrailsInputConfig = field(default_factory=GuardrailsInputConfig)
    output: GuardrailsOutputConfig = field(default_factory=GuardrailsOutputConfig)


@dataclass
class CacheConfig:
    enabled: bool = True
    backend: str = "diskcache"
    cache_dir: str = ".cache/responses"
    ttl_seconds: int = 3600

@dataclass  
class AppMetaConfig:  
    name: str = "Enterprise RAG Assistant"  
    version: str = "1.0.0"  
    environment: str = "development"  
    log_level: str = "INFO"  
  
@dataclass  
class APIConfig:  
    host: str = "0.0.0.0"  
    port: int = 8000  
    workers: int = 1  
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:8501"])  
 

@dataclass
class AppConfig:
    """Root configuration object. All modules depend on this."""
    app: AppMetaConfig = field(default_factory=AppMetaConfig)
    api: APIConfig = field(default_factory=APIConfig)   
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    guardrails: GuardrailsConfig = field(default_factory=GuardrailsConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)


def _dict_to_config(data: dict[str, Any], config_class: type) -> Any:
    """
    Recursively convert a nested dict (from YAML) into typed dataclass instances.
    """
    import dataclasses

    if not dataclasses.is_dataclass(config_class):
        return data

    field_types = {f.name: f.type for f in dataclasses.fields(config_class)}
    kwargs = {}

    for f in dataclasses.fields(config_class):
        if f.name in data:
            value = data[f.name]
            # If the field type is also a dataclass, recurse
            field_class = f.default_factory.__class__ if callable(
                getattr(f, 'default_factory', None)
            ) else None

            # Resolve the actual type annotation
            type_hint = f.type
            if isinstance(type_hint, str):
                # String annotation — skip nested conversion for simplicity
                kwargs[f.name] = value
            elif isinstance(value, dict) and hasattr(type_hint, '__dataclass_fields__'):
                kwargs[f.name] = _dict_to_config(value, type_hint)
            else:
                kwargs[f.name] = value

    return config_class(**kwargs)


@lru_cache(maxsize=1)
def get_config(config_path: str = "config.yaml") -> AppConfig:
    """
    Load and return the application configuration.

    Cached with lru_cache — config.yaml is read once at first call,
    then the same AppConfig instance is returned on all subsequent calls.
    This is safe because config is read-only after startup.

    Args:
        config_path: Path to the config YAML file.

    Returns:
        Populated AppConfig instance.
    """
    path = Path(config_path)
    if not path.exists():
        # Return defaults if config.yaml is missing.
        # This supports unit tests that don't need a config file.
        return AppConfig()

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # Manual mapping — straightforward and explicit
    cfg = AppConfig()

    if "ingestion" in raw:
        ing = raw["ingestion"]
        chunking_data = ing.get("chunking", {})
        cfg.ingestion = IngestionConfig(
            data_dir=ing.get("data_dir", "data/pdfs"),
            chunking=ChunkingConfig(
                chunk_size=chunking_data.get("chunk_size", 512),
                chunk_overlap=chunking_data.get("chunk_overlap", 50),
                separators=chunking_data.get("separators", ["\n\n", "\n", " "]),
            ),
            deduplication_enabled=ing.get("deduplication", {}).get("enabled", True),
        )

    if "embedding" in raw:
        emb = raw["embedding"]
        cfg.embedding = EmbeddingConfig(
            model=emb.get("model", "BAAI/bge-small-en-v1.5"),
            dimensions=emb.get("dimensions", 384),
            batch_size=emb.get("batch_size", 100),
            cache_enabled=emb.get("cache_enabled", True),
            cache_dir=emb.get("cache_dir", ".cache/embeddings"),
        )

    if "vector_store" in raw:
        vs = raw["vector_store"]
        ret = vs.get("retrieval", {})
        cfg.vector_store = VectorStoreConfig(
            provider=vs.get("provider", "faiss"),
            index_name=vs.get("index_name", "aws-well-architected"),
            retrieval=RetrievalConfig(
                top_k=ret.get("top_k", 5),
                score_threshold=ret.get("score_threshold", 0.70),
                search_type=ret.get("search_type", "similarity"),
            ),
        )

    if "reranker" in raw:
        rer = raw["reranker"]
        cfg.reranker = RerankerConfig(
            enabled=rer.get("enabled", True),
            model=rer.get("model", "BAAI/bge-reranker-base"),
            top_n=rer.get("top_n", 5),
        )

    if "generation" in raw:
        gen = raw["generation"]
        cfg.generation = GenerationConfig(
            temperature=gen.get("temperature", 0.1),
            max_new_tokens=gen.get("max_new_tokens", 512),
            timeout_seconds=gen.get("timeout_seconds", 30),
        )

    if "models" in raw:
        mod = raw["models"]
        pri = mod.get("primary", {})
        fal = mod.get("fallback", {})
        cfg.models = ModelsConfig(
            primary=PrimaryModelConfig(
                base_model=pri.get("base_model", "microsoft/Phi-3-mini-4k-instruct"),
                adapter_repo=pri.get("adapter_repo", "kishoraditya/enterprise-rag-adapter"),
                load_in_4bit=pri.get("load_in_4bit", True),
                device=pri.get("device", "auto"),
            ),
            fallback=FallbackModelConfig(
                model=fal.get("model", "gpt-4o-mini"),
                max_tokens=fal.get("max_tokens", 512),
                temperature=fal.get("temperature", 0.1),
            ),
        )

    if "guardrails" in raw:
        grd = raw["guardrails"]
        inp = grd.get("input", {})
        out = grd.get("output", {})
        cfg.guardrails = GuardrailsConfig(
            input=GuardrailsInputConfig(
                max_query_length=inp.get("max_query_length", 1000),
                injection_detection=inp.get("injection_detection", True),
            ),
            output=GuardrailsOutputConfig(
                grounding_check=out.get("grounding_check", True),
                grounding_threshold=out.get("grounding_threshold", 0.50),
                pii_redaction=out.get("pii_redaction", True),
            ),
        )

    if "cache" in raw:
        cch = raw["cache"]
        cfg.cache = CacheConfig(
            enabled=cch.get("enabled", True),
            backend=cch.get("backend", "diskcache"),
            cache_dir=cch.get("cache_dir", ".cache/responses"),
            ttl_seconds=cch.get("ttl_seconds", 3600),
        )

    return cfg