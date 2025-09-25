"""Retrieval augmented generation pipeline for KUx."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ..config import MODEL_OPTIONS, ModelOption, RAGConfig, TrainConfig

LOGGER = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = dedent(
    """
    You are KUx, an omniscient assistant for Kasetsart University Computer Science students.
    Answer with verified facts from Kasetsart University sources. If unsure, state that you do not know.
    """
).strip()

MODEL_OPTION_MAP = {option.key: option for option in MODEL_OPTIONS}


@dataclass(slots=True)
class MediaInput:
    """Container for user-provided multimodal attachments."""

    images: List[str] = field(default_factory=list)
    audio: List[str] = field(default_factory=list)
    video: List[str] = field(default_factory=list)

    @classmethod
    def from_payload(
        cls,
        images: Optional[Sequence[str]] = None,
        audio: Optional[Sequence[str]] = None,
        video: Optional[Sequence[str]] = None,
    ) -> "MediaInput":
        def _clean(items: Optional[Sequence[str]]) -> List[str]:
            if not items:
                return []
            return [item for item in items if item]

        return cls(images=_clean(images), audio=_clean(audio), video=_clean(video))

    def is_empty(self) -> bool:
        return not (self.images or self.audio or self.video)


class LocalHFGenerator:
    """Wrapper around a local Hugging Face causal LM for inference."""

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        adapter_path: Optional[str] = None,
    ) -> None:
        LOGGER.info("Loading generator from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model = self._maybe_apply_adapter(base_model, adapter_path)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

    @staticmethod
    def _maybe_apply_adapter(model: AutoModelForCausalLM, adapter_path: Optional[str]):
        if not adapter_path:
            return model
        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists():
            LOGGER.warning("Adapter path %s does not exist; continuing with the base model.", adapter_dir)
            return model
        try:  # pragma: no cover - requires peft at runtime
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "peft is required to load LoRA adapters. Install with `pip install peft`."
            ) from exc

        LOGGER.info("Applying LoRA adapters from %s", adapter_dir)
        adapted_model = PeftModel.from_pretrained(model, adapter_dir)
        return adapted_model

    def generate(self, prompt: str) -> str:
        outputs = self.pipe(prompt)
        text = outputs[0]["generated_text"]
        return text[len(prompt) :].strip()


class QwenOmniGenerator:
    """Inference helper for Qwen3-Omni multimodal dialogue."""

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.2,
        use_audio_in_video: bool = True,
        adapter_path: Optional[str] = None,
    ) -> None:
        try:  # pragma: no cover - heavy dependency only available at runtime
            from transformers import (
                Qwen3OmniMoeForConditionalGeneration,
                Qwen3OmniMoeProcessor,
            )
        except ImportError as exc:  # pragma: no cover - import validated in runtime environment
            raise ImportError(
                "Qwen3-Omni dependencies are missing. Install the latest transformers from "
                "source (pip install git+https://github.com/huggingface/transformers)."
            ) from exc

        LOGGER.info("Loading Qwen3-Omni model from %s", model_path)
        base_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        if hasattr(base_model, "disable_talker"):
            base_model.disable_talker()
        self.model = self._maybe_apply_adapter(base_model, adapter_path)
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_audio_in_video = use_audio_in_video

    @staticmethod
    def _maybe_apply_adapter(model, adapter_path: Optional[str]):
        if not adapter_path:
            return model
        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists():
            LOGGER.warning("Adapter path %s does not exist; continuing with the base model.", adapter_dir)
            return model
        try:  # pragma: no cover - requires peft during runtime
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "peft is required to load Qwen3-Omni LoRA adapters. Install with `pip install peft`."
            ) from exc

        LOGGER.info("Applying Qwen3-Omni LoRA adapters from %s", adapter_dir)
        adapted_model = PeftModel.from_pretrained(model, adapter_dir)
        return adapted_model

    @staticmethod
    def _collect_media(messages: Sequence[dict]) -> tuple[Optional[List[str]], Optional[List[str]], Optional[List[str]]]:
        images: List[str] = []
        audio: List[str] = []
        videos: List[str] = []
        for message in messages:
            content = message.get("content", [])
            if isinstance(content, dict):
                content = [content]
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "image" and item.get("image") is not None:
                    images.append(item["image"])
                elif item_type == "audio" and item.get("audio") is not None:
                    audio.append(item["audio"])
                elif item_type == "video" and item.get("video") is not None:
                    videos.append(item["video"])
        return (audio or None, images or None, videos or None)

    def generate(self, messages: Sequence[dict]) -> str:
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        audios, images, videos = self._collect_media(messages)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        )
        inputs = inputs.to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            return_audio=False,
            thinker_return_dict_in_generate=True,
            use_audio_in_video=self.use_audio_in_video,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )
        sequences = getattr(outputs, "sequences", outputs)
        offset = inputs["input_ids"].shape[1]
        text = self.processor.batch_decode(
            sequences[:, offset:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return text.strip()


class RAGPipeline:
    """High level helper for running retrieval augmented generation."""

    def __init__(
        self,
        rag_config: Optional[RAGConfig] = None,
        train_config: Optional[TrainConfig] = None,
        system_prompt: Optional[str] = None,
        model_key: Optional[str] = None,
        use_finetuned: bool = True,
    ) -> None:
        self.rag_config = rag_config or RAGConfig()
        self.train_config = train_config or TrainConfig()
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.model_key = model_key or MODEL_OPTIONS[0].key
        self.model_option = self._resolve_model_option(self.model_key)
        self.use_finetuned = use_finetuned
        self.embeddings = HuggingFaceEmbeddings(model_name=self.rag_config.embedding_model_name)
        self.vector_store = self._load_vector_store(self.rag_config.vector_db_path)
        base_model, adapter_path = self._resolve_model_sources()
        if self.model_option.multimodal:
            self.generator = QwenOmniGenerator(base_model, adapter_path=adapter_path)
        else:
            self.generator = LocalHFGenerator(base_model, adapter_path=adapter_path)

    def _resolve_model_option(self, model_key: str) -> ModelOption:
        try:
            return MODEL_OPTION_MAP[model_key]
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise ValueError(f"Unknown model selection: {model_key}") from exc

    def _resolve_model_sources(self) -> Tuple[str, Optional[str]]:
        base_model = self.model_option.model_name
        if not self.use_finetuned:
            return base_model, None

        if self.model_option.key != "qwen3-omni-30b":
            LOGGER.warning(
                "Fine-tuned adapters are currently only supported for Qwen3-Omni. Continuing with base %s.",
                base_model,
            )
            return base_model, None

        adapter_dir = Path(self.train_config.output_dir)
        if not adapter_dir.exists():
            LOGGER.warning(
                "Requested fine-tuned model but no adapters found at %s. Continuing with base model.",
                adapter_dir,
            )
            return base_model, None

        if not self._contains_adapter_weights(adapter_dir):
            LOGGER.warning(
                "Adapter directory %s does not contain LoRA weights. Continuing with base model.",
                adapter_dir,
            )
            return base_model, None

        return base_model, str(adapter_dir)

    @staticmethod
    def _contains_adapter_weights(adapter_dir: Path) -> bool:
        expected_config = adapter_dir / "adapter_config.json"
        if expected_config.exists():
            return True
        # Accept common safetensors/bin outputs
        for pattern in ("adapter_model.bin", "adapter_model.safetensors"):
            if (adapter_dir / pattern).exists():
                return True
        return False

    def _load_vector_store(self, path: Path | str) -> Optional[FAISS]:
        path = Path(path)
        if not path.exists():
            LOGGER.warning(
                "Vector store not found at %s. Responses will be generated without retrieval until you run the ingestion pipeline.",
                path,
            )
            return None
        return FAISS.load_local(
            str(path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def _format_context(self, documents: Iterable[Document]) -> str:
        context_blocks: List[str] = []
        for idx, doc in enumerate(documents, start=1):
            metadata = doc.metadata
            source = metadata.get("source", "unknown")
            block = dedent(
                f"""
                [Document {idx} | Source: {source}]
                {doc.page_content.strip()}
                """
            ).strip()
            context_blocks.append(block)
        return "\n\n".join(context_blocks)

    def _build_text_history(self, history: Optional[Sequence[Tuple[str, str]]]) -> str:
        if not history:
            return ""
        turns: List[str] = []
        for user_text, assistant_text in history:
            if user_text:
                turns.append(
                    dedent(
                        f"""
                        <|im_start|>user
                        {user_text}
                        <|im_end|>
                        """
                    ).strip()
                )
            if assistant_text:
                turns.append(
                    dedent(
                        f"""
                        <|im_start|>assistant
                        {assistant_text}
                        <|im_end|>
                        """
                    ).strip()
                )
        return "\n".join(turns)

    def build_prompt(
        self,
        question: str,
        documents: List[Document],
        history: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> str:
        context = self._format_context(documents)
        history_block = self._build_text_history(history)
        prompt_blocks = [
            dedent(
                f"""
                <|im_start|>system
                {self.system_prompt}
                <|im_end|>
                """
            ).strip()
        ]
        if history_block:
            prompt_blocks.append(history_block)
        prompt_blocks.append(
            dedent(
                f"""
                <|im_start|>user
                Question: {question}

                Use the following context to ground your answer:
                {context}
                <|im_end|>
                <|im_start|>assistant
                """
            ).strip()
        )
        return "\n".join(prompt_blocks)

    def _build_multimodal_messages(
        self,
        question: str,
        documents: List[Document],
        media: MediaInput,
        history: Optional[Sequence[Tuple[str, str]]],
    ) -> List[dict]:
        messages: List[dict] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        ]
        if documents:
            context = self._format_context(documents)
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Kasetsart University supporting material:\n"
                                f"{context}"
                            ),
                        }
                    ],
                }
            )
        if history:
            for user_turn, assistant_turn in history:
                if user_turn:
                    messages.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": user_turn}],
                        }
                    )
                if assistant_turn:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": assistant_turn}],
                        }
                    )

        user_content: List[dict] = []
        for image_path in media.images:
            user_content.append({"type": "image", "image": image_path})
        for audio_path in media.audio:
            user_content.append({"type": "audio", "audio": audio_path})
        for video_path in media.video:
            user_content.append({"type": "video", "video": video_path})

        request_text = question.strip() or (
            "Please analyse the uploaded media and explain how it relates to Kasetsart University's "
            "Computer Science programme."
        )
        user_content.append({"type": "text", "text": request_text})
        messages.append({"role": "user", "content": user_content})
        return messages

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        media: Optional[MediaInput] = None,
        history: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> str:
        media = media or MediaInput()
        question_text = question.strip()
        if not question_text and media.is_empty():
            return "Please provide a question or upload audio, image, or video content."

        documents: List[Document] = []
        if question_text and self.vector_store is not None:
            top_k = top_k or self.rag_config.max_retrieval_docs
            retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
            documents = retriever.get_relevant_documents(question_text)
            if not documents and not self.model_option.multimodal:
                return "I could not find supporting documents for that question."
        elif question_text and self.vector_store is None:
            LOGGER.warning("Vector store unavailable; continuing without retrieved context.")

        if self.model_option.multimodal:
            messages = self._build_multimodal_messages(question_text, documents, media, history)
            return self.generator.generate(messages)

        prompt = self.build_prompt(question_text, documents, history)
        return self.generator.generate(prompt)


__all__ = ["RAGPipeline", "LocalHFGenerator", "QwenOmniGenerator", "MediaInput", "DEFAULT_SYSTEM_PROMPT"]
