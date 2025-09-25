"""Gradio chat interface for the KUx assistant."""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import gradio as gr

from ..config import MODEL_OPTIONS, RAGConfig, TrainConfig
from ..rag.pipeline import DEFAULT_SYSTEM_PROMPT, MediaInput, RAGPipeline

LOGGER = logging.getLogger(__name__)

MODEL_LABEL_TO_KEY = {option.label: option.key for option in MODEL_OPTIONS}
MODEL_KEY_TO_LABEL = {option.key: option.label for option in MODEL_OPTIONS}


@dataclass(frozen=True)
class LaunchState:
    """Runtime configuration supplied when launching the Gradio app."""

    vector_db_path: Optional[str] = None
    adapter_dir: Optional[str] = None
    default_model_key: str = MODEL_OPTIONS[0].key
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT


_LAUNCH_STATE = LaunchState()


@lru_cache(maxsize=8)
def _load_pipeline_cached(
    model_key: str,
    use_finetuned: bool,
    system_prompt: str,
    vector_db_path: Optional[str],
    adapter_dir: Optional[str],
) -> RAGPipeline:
    LOGGER.info(
        "Initialising RAG pipeline (model=%s, finetuned=%s)",
        model_key,
        use_finetuned,
    )
    default_rag = RAGConfig()
    rag_config = RAGConfig(
        vector_db_path=Path(vector_db_path) if vector_db_path else default_rag.vector_db_path
    )
    default_train = TrainConfig()
    train_config = TrainConfig(output_dir=adapter_dir if adapter_dir else default_train.output_dir)
    return RAGPipeline(
        rag_config=rag_config,
        train_config=train_config,
        system_prompt=system_prompt,
        model_key=model_key,
        use_finetuned=use_finetuned,
    )


def load_pipeline(model_key: str, use_finetuned: bool, system_prompt: str) -> RAGPipeline:
    base_prompt = _LAUNCH_STATE.default_system_prompt or DEFAULT_SYSTEM_PROMPT
    prompt = system_prompt.strip() if system_prompt and system_prompt.strip() else base_prompt
    return _load_pipeline_cached(
        model_key,
        use_finetuned,
        prompt,
        _LAUNCH_STATE.vector_db_path,
        _LAUNCH_STATE.adapter_dir,
    )


def _extract_paths(payload: Any) -> List[str]:
    if not payload:
        return []
    if isinstance(payload, (str, Path)):
        return [str(payload)]
    items: Sequence[Any]
    if isinstance(payload, Sequence):
        items = payload
    else:  # single temp file or dict
        items = [payload]
    paths: List[str] = []
    for item in items:
        if not item:
            continue
        if isinstance(item, (str, Path)):
            paths.append(str(item))
        elif isinstance(item, dict):
            for key in ("path", "name"):
                value = item.get(key)
                if value:
                    paths.append(str(value))
                    break
        else:
            path = getattr(item, "name", None)
            if path:
                paths.append(str(path))
    return paths


def _format_user_display(message: str, media: MediaInput) -> str:
    parts: List[str] = []
    text = message.strip()
    if text:
        parts.append(text)
    for label, files in (
        ("Image", media.images),
        ("Audio", media.audio),
        ("Video", media.video),
    ):
        for file_path in files:
            parts.append(f"[{label}] {Path(file_path).name}")
    return "\n".join(parts) if parts else "[No user content]"


def respond(
    message: str,
    history: List[Tuple[str, str]],
    model_label: str,
    use_finetuned: bool,
    system_prompt: str,
    image_payload: Any,
    audio_payload: Any,
    video_payload: Any,
) -> Tuple[List[Tuple[str, str]], gr.Textbox, gr.File, gr.File, gr.File]:
    model_key = MODEL_LABEL_TO_KEY.get(model_label, MODEL_OPTIONS[0].key)
    pipeline = load_pipeline(model_key, use_finetuned, system_prompt)
    media = MediaInput.from_payload(
        images=_extract_paths(image_payload),
        audio=_extract_paths(audio_payload),
        video=_extract_paths(video_payload),
    )
    display_text = _format_user_display(message, media)
    prompt_history = [(user, bot) for user, bot in history if user or bot]
    answer = pipeline.answer(message, media=media, history=prompt_history)
    updated_history = history + [(display_text, answer)]
    return (
        updated_history,
        gr.Textbox.update(value=""),
        gr.File.update(value=None),
        gr.File.update(value=None),
        gr.File.update(value=None),
    )


def launch(
    *,
    vector_db_path: Optional[str] = None,
    adapter_dir: Optional[str] = None,
    default_model_key: Optional[str] = None,
    default_system_prompt: Optional[str] = None,
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
) -> None:
    global _LAUNCH_STATE
    desired_model_key = default_model_key or MODEL_OPTIONS[0].key
    _LAUNCH_STATE = replace(
        _LAUNCH_STATE,
        vector_db_path=str(vector_db_path) if vector_db_path else None,
        adapter_dir=str(adapter_dir) if adapter_dir else None,
        default_model_key=(
            desired_model_key if desired_model_key in MODEL_KEY_TO_LABEL else MODEL_OPTIONS[0].key
        ),
        default_system_prompt=(
            default_system_prompt.strip()
            if default_system_prompt and default_system_prompt.strip()
            else DEFAULT_SYSTEM_PROMPT
        ),
    )
    _load_pipeline_cached.cache_clear()
    description = (
        "KUx is a retrieval-augmented assistant for Kasetsart University Computer Science students."
    )
    theme = gr.themes.Default(primary_hue=gr.themes.colors.green)
    default_model_label = MODEL_KEY_TO_LABEL.get(_LAUNCH_STATE.default_model_key, MODEL_OPTIONS[0].label)
    system_prompt_default = _LAUNCH_STATE.default_system_prompt or DEFAULT_SYSTEM_PROMPT
    with gr.Blocks(theme=theme) as demo:
        gr.Markdown(
            "# KUx – Kasetsart CS Assistant\n"
            f"<span style='color: #0f5132'>{description}</span>",
            elem_id="kux-header",
        )
        chatbot = gr.Chatbot(label="Conversation", height=520)
        with gr.Row():
            with gr.Column(scale=3):
                message_box = gr.Textbox(
                    label="Ask KUx",
                    placeholder="Ask a question or leave blank to analyse uploaded media",
                    lines=4,
                )
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear conversation", variant="secondary")
            with gr.Column(scale=2):
                image_files = gr.File(
                    label="Upload images (OCR, object grounding, image math)",
                    file_types=["image"],
                    file_count="multiple",
                )
                audio_files = gr.File(
                    label="Upload audio (speech recognition, translation, captioning)",
                    file_types=["audio"],
                    file_count="multiple",
                )
                video_files = gr.File(
                    label="Upload videos (audio-visual QA/interactions)",
                    file_types=["video"],
                    file_count="multiple",
                )
        with gr.Accordion("Assistant settings", open=False):
            model_dropdown = gr.Dropdown(
                label="Base model",
                choices=list(MODEL_LABEL_TO_KEY.keys()),
                value=default_model_label,
                interactive=True,
            )
            finetune_checkbox = gr.Checkbox(
                label="Use fine-tuned adapters (LoRA)",
                value=True,
                interactive=True,
            )
            system_prompt_box = gr.Textbox(
                label="System prompt",
                value=system_prompt_default,
                lines=4,
                interactive=True,
            )

        send_btn.click(
            fn=respond,
            inputs=[
                message_box,
                chatbot,
                model_dropdown,
                finetune_checkbox,
                system_prompt_box,
                image_files,
                audio_files,
                video_files,
            ],
            outputs=[chatbot, message_box, image_files, audio_files, video_files],
        )

        clear_btn.click(
            fn=lambda: (
                [],
                gr.Textbox.update(value=""),
                gr.File.update(value=None),
                gr.File.update(value=None),
                gr.File.update(value=None),
            ),
            inputs=None,
            outputs=[chatbot, message_box, image_files, audio_files, video_files],
        )

    demo.queue().launch(server_name=server_name, server_port=server_port, share=share)


__all__ = ["launch"]
