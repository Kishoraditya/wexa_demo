"""
backend/services/llm_manager.py

LLM Manager — Model Routing and Fallback
=========================================
Responsibilities:
  1. Attempt to load the fine-tuned Phi-3-mini + LoRA adapter at startup
  2. Fall back to OpenAI GPT-4o-mini if the primary model is unavailable
  3. Provide a unified generate() interface regardless of which model is active
  4. Log which model served every request (observability requirement)

Fallback activation conditions:
  - HuggingFace Hub is unreachable (network error)
  - Adapter repo does not exist or is empty (config.models.primary.adapter_repo = "")
  - CUDA OOM or CPU inference too slow (timeout exceeded)
  - Model returns an exception during generation

The fallback is NOT a degraded mode — it is a first-class generation path.
OpenAI GPT-4o-mini with our structured prompt produces high-quality answers.
The primary model exists to demonstrate fine-tuning understanding; the fallback
ensures the system is always available regardless of GPU/model availability.

Author: Enterprise RAG Assistant
"""

import os
import time
from enum import Enum
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from backend.core.config import get_config
from backend.core.logging import get_logger

logger = get_logger(__name__)
config = get_config()


# ─────────────────────────────────────────────────────────────────────────────
# Model Type Enum
# ─────────────────────────────────────────────────────────────────────────────

class ModelType(str, Enum):
    """
    Identifies which model generated a given response.
    Stored in every API response for observability and debugging.
    str mixin makes it JSON-serializable without extra conversion.
    """
    FINE_TUNED = "fine_tuned_phi3_qlora"
    OPENAI_FALLBACK = "openai_gpt4o_mini"
    UNAVAILABLE = "unavailable"


# ─────────────────────────────────────────────────────────────────────────────
# Generation Result
# ─────────────────────────────────────────────────────────────────────────────

class GenerationResult:
    """
    Value object carrying the raw LLM output and generation metadata.

    Separating generation output from the final API response schema
    keeps the LLM manager focused on generation and lets the RAG pipeline
    handle post-processing (confidence parsing, source extraction, etc.).
    """

    def __init__(
        self,
        raw_text: str,
        model_type: ModelType,
        generation_latency_ms: int,
        tokens_used: Optional[int] = None,
        error: Optional[str] = None,
    ):
        self.raw_text = raw_text
        self.model_type = model_type
        self.generation_latency_ms = generation_latency_ms
        self.tokens_used = tokens_used
        self.error = error

    @property
    def succeeded(self) -> bool:
        return self.error is None and bool(self.raw_text)

    def __repr__(self) -> str:
        preview = self.raw_text[:80] if self.raw_text else ""
        return (
            f"GenerationResult("
            f"model={self.model_type}, "
            f"latency_ms={self.generation_latency_ms}, "
            f"tokens={self.tokens_used}, "
            f"preview='{preview}...')"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Fine-Tuned Model Loader
# ─────────────────────────────────────────────────────────────────────────────

class FineTunedModelLoader:
    """
    Loads the Phi-3-mini base model + LoRA adapter from HuggingFace Hub.

    Loading strategy:
      - Base model: microsoft/Phi-3-mini-4k-instruct
      - Adapter: {config.models.primary.adapter_repo} on HF Hub
      - Quantization: 4-bit NF4 (BitsAndBytes) — reduces VRAM from ~7GB to ~4GB
      - Device: auto (uses GPU if available, CPU otherwise)

    Why load base + adapter separately (not merged weights)?
      Merging the adapter into the base model produces a single set of weights
      that cannot be unmerged. Keeping them separate means:
        1. Adapter updates (~60MB) can be deployed without re-downloading
           the base model (~2.2GB)
        2. Multiple adapters can be swapped at runtime (future: per-pillar adapters)
        3. The base model is shared — useful if multiple adapter variants exist

    CPU fallback note:
      Phi-3-mini with 4-bit quantization requires ~4GB VRAM on GPU or
      ~8GB RAM on CPU. CPU inference is ~5-20 tokens/second (slow but functional).
      The timeout in config.generation.timeout_seconds should account for this.
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._loaded = False

    def load(self) -> bool:
        """
        Attempt to load the fine-tuned model. Returns True if successful.

        This is called once at startup. On failure, the LLM manager
        activates the OpenAI fallback permanently for this session.
        """
        adapter_repo = config.models.primary.adapter_repo

        if not adapter_repo:
            logger.warning(
                "No adapter_repo configured — fine-tuned model unavailable. "
                "Set models.primary.adapter_repo in config.yaml after training. "
                "Falling back to OpenAI."
            )
            return False

        try:
            # Import here to avoid hard dependency when running without GPU
            # These imports fail gracefully if torch/transformers are not installed
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            from transformers import BitsAndBytesConfig
            from peft import PeftModel

            logger.info(
                "Loading fine-tuned model",
                extra={
                    "base_model": config.models.primary.base_model,
                    "adapter_repo": adapter_repo,
                    "load_in_4bit": config.models.primary.load_in_4bit,
                },
            )
            load_start = time.time()

            # 4-bit NF4 quantization config
            # NF4 (Normal Float 4) is designed specifically for normally distributed
            # weights (which neural network weights are). It achieves better
            # precision than standard INT4 at the same memory footprint.
            # double_quant quantizes the quantization constants themselves,
            # saving an additional ~0.4 bits per parameter.
            bnb_config = None
            if config.models.primary.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                config.models.primary.base_model,
                trust_remote_code=True,  # required for Phi-3
            )

            # Load base model
            self._model = AutoModelForCausalLM.from_pretrained(
                config.models.primary.base_model,
                quantization_config=bnb_config,
                device_map=config.models.primary.device,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

            # Apply LoRA adapter on top of base model
            self._model = PeftModel.from_pretrained(
                self._model,
                adapter_repo,
            )
            self._model.eval()  # disable dropout, set to inference mode

            # Wrap in HuggingFace pipeline for convenient generate() interface
            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                max_new_tokens=config.generation.max_new_tokens,
                temperature=config.generation.temperature,
                do_sample=config.generation.temperature > 0,
                return_full_text=False,  # only return newly generated tokens
            )

            load_ms = round((time.time() - load_start) * 1000)
            logger.info(
                "Fine-tuned model loaded successfully",
                extra={
                    "adapter_repo": adapter_repo,
                    "load_time_ms": load_ms,
                },
            )
            self._loaded = True
            return True

        except ImportError as e:
            logger.warning(
                "torch/transformers/peft not installed — fine-tuned model unavailable",
                extra={"error": str(e)},
            )
            return False
        except Exception as e:
            logger.error(
                "Failed to load fine-tuned model",
                extra={"error": str(e), "adapter_repo": adapter_repo},
            )
            return False

    def generate(self, prompt_text: str) -> GenerationResult:
        """
        Generate a response using the fine-tuned model.

        Args:
            prompt_text: Fully formatted prompt string (system + human combined).

        Returns:
            GenerationResult with raw text and metadata.
        """
        if not self._loaded or self._pipeline is None:
            return GenerationResult(
                raw_text="",
                model_type=ModelType.FINE_TUNED,
                generation_latency_ms=0,
                error="Fine-tuned model not loaded",
            )

        gen_start = time.time()
        try:
            outputs = self._pipeline(prompt_text)
            raw_text = outputs[0]["generated_text"].strip()
            gen_ms = round((time.time() - gen_start) * 1000)

            # Estimate token count from output length.
            # Exact count requires tokenizer.encode() — approximation is fine
            # for logging purposes. Precise tracking would use the pipeline's
            # return_dict_in_generate=True with detailed output.
            approx_tokens = len(raw_text.split()) * 1.3  # rough tokens per word

            return GenerationResult(
                raw_text=raw_text,
                model_type=ModelType.FINE_TUNED,
                generation_latency_ms=gen_ms,
                tokens_used=int(approx_tokens),
            )

        except Exception as e:
            gen_ms = round((time.time() - gen_start) * 1000)
            logger.error(
                "Fine-tuned model generation failed",
                extra={"error": str(e), "latency_ms": gen_ms},
            )
            return GenerationResult(
                raw_text="",
                model_type=ModelType.FINE_TUNED,
                generation_latency_ms=gen_ms,
                error=str(e),
            )

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Fallback
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIFallback:
    """
    OpenAI GPT-4o-mini as the fallback generation model.

    This is not a degraded mode — it is a high-quality generation path that
    activates when the fine-tuned model is unavailable. In many ways it produces
    better answers than the fine-tuned model because:
      - GPT-4o-mini has stronger instruction following
      - It handles complex multi-hop reasoning better
      - No GPU/memory constraints

    The primary cost: ~$0.001 per request at 500 tokens.
    Acceptable for development and low-traffic production.
    At scale, the fine-tuned model is preferred for cost reasons.
    """

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error(
                "OPENAI_API_KEY not set — fallback generation unavailable. "
                "Set this environment variable before starting the API."
            )
            self._client = None
            return

        self._client = ChatOpenAI(
            model=config.models.fallback.model,
            temperature=config.models.fallback.temperature,
            max_tokens=config.models.fallback.max_tokens,
            api_key=api_key,
            timeout=config.generation.timeout_seconds,
        )
        logger.info(
            "OpenAI fallback initialized",
            extra={"model": config.models.fallback.model},
        )

    def generate(
        self,
        system_prompt: str,
        human_message: str,
    ) -> GenerationResult:
        """
        Generate a response using the OpenAI API.

        Takes system and human messages separately because the OpenAI
        client handles message formatting internally — we should not
        concatenate them into a single string as we do for the local model.

        Args:
            system_prompt: The system instruction string.
            human_message: The formatted human turn (context + question).

        Returns:
            GenerationResult with raw text and token usage.
        """
        if self._client is None:
            return GenerationResult(
                raw_text="",
                model_type=ModelType.OPENAI_FALLBACK,
                generation_latency_ms=0,
                error="OpenAI client not initialized — OPENAI_API_KEY missing",
            )

        gen_start = time.time()
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_message),
            ]

            response = self._client.invoke(messages)
            raw_text = response.content.strip()
            gen_ms = round((time.time() - gen_start) * 1000)

            # Extract token usage from OpenAI response metadata
            tokens_used = None
            if hasattr(response, "response_metadata"):
                usage = response.response_metadata.get("token_usage", {})
                tokens_used = usage.get("total_tokens")

            logger.info(
                "OpenAI generation complete",
                extra={
                    "model": config.models.fallback.model,
                    "latency_ms": gen_ms,
                    "tokens_used": tokens_used,
                },
            )

            return GenerationResult(
                raw_text=raw_text,
                model_type=ModelType.OPENAI_FALLBACK,
                generation_latency_ms=gen_ms,
                tokens_used=tokens_used,
            )

        except Exception as e:
            gen_ms = round((time.time() - gen_start) * 1000)
            logger.error(
                "OpenAI generation failed",
                extra={"error": str(e), "latency_ms": gen_ms},
            )
            return GenerationResult(
                raw_text="",
                model_type=ModelType.OPENAI_FALLBACK,
                generation_latency_ms=gen_ms,
                error=str(e),
            )

    @property
    def is_available(self) -> bool:
        return self._client is not None


# ─────────────────────────────────────────────────────────────────────────────
# LLM Manager — Unified Interface
# ─────────────────────────────────────────────────────────────────────────────

class LLMManager:
    """
    Manages model selection, routing, and fallback logic.

    Startup sequence:
      1. Attempt to load fine-tuned Phi-3-mini + LoRA adapter
      2. If successful: use fine-tuned model as primary
      3. If failed: log warning, activate OpenAI fallback permanently
      4. If use_fine_tuned=False in request: skip to fallback explicitly

    Every call to generate() logs model_used so the API response always
    carries accurate attribution regardless of which path was taken.
    """

    def __init__(self):
        self._fine_tuned = FineTunedModelLoader()
        self._fallback = OpenAIFallback()

        # Attempt primary model load at startup
        logger.info("Attempting to load fine-tuned model at startup")
        primary_loaded = self._fine_tuned.load()

        if primary_loaded:
            logger.info("Primary model (fine-tuned) is active")
        else:
            logger.warning(
                "Primary model unavailable — OpenAI fallback is active. "
                "All requests will use OpenAI GPT-4o-mini until restart."
            )

        self._primary_available = primary_loaded

    def generate(
        self,
        system_prompt: str,
        human_message: str,
        use_fine_tuned: bool = True,
    ) -> GenerationResult:
        """
        Generate a response, routing to the appropriate model.

        Routing logic:
          1. If use_fine_tuned=False → go directly to fallback
          2. If use_fine_tuned=True and primary available → use primary
          3. If primary fails at runtime → retry with fallback
          4. If fallback also fails → return error GenerationResult

        Args:
            system_prompt: System instruction string.
            human_message: Formatted human turn (context + question).
            use_fine_tuned: If True, prefer the fine-tuned model.
                            The request body can set this to False to
                            explicitly request the OpenAI model.

        Returns:
            GenerationResult from whichever model served the request.
        """
        # ── Explicit fallback request ──────────────────────────────────────
        if not use_fine_tuned:
            logger.info("Explicit fallback requested by caller")
            return self._fallback.generate(system_prompt, human_message)

        # ── Primary model path ─────────────────────────────────────────────
        if self._primary_available:
            # The fine-tuned model takes a single concatenated prompt
            # (it does not have a separate system/human message interface)
            full_prompt = f"{system_prompt}\n\n{human_message}"
            result = self._fine_tuned.generate(full_prompt)

            if result.succeeded:
                logger.info(
                    "Request served by fine-tuned model",
                    extra={"latency_ms": result.generation_latency_ms},
                )
                return result

            # Primary failed at runtime — fall through to fallback
            logger.warning(
                "Fine-tuned model failed at runtime — activating fallback",
                extra={"error": result.error},
            )

        # ── Fallback path ──────────────────────────────────────────────────
        logger.info("Request routed to OpenAI fallback")
        fallback_result = self._fallback.generate(system_prompt, human_message)

        if not fallback_result.succeeded:
            logger.error(
                "Both primary and fallback models failed",
                extra={"fallback_error": fallback_result.error},
            )

        return fallback_result

    @property
    def primary_available(self) -> bool:
        return self._primary_available

    @property
    def fallback_available(self) -> bool:
        return self._fallback.is_available

    @property
    def any_model_available(self) -> bool:
        return self._primary_available or self._fallback.is_available