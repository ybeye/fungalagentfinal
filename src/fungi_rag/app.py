from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from fungi_rag.config import get_settings
from fungi_rag.evaluate import evaluate_single_answer, run_evaluation
from fungi_rag.ingest import DocumentIngestor
from fungi_rag.models import ResearchBrief
from fungi_rag.sources import SourceDownloader, load_manifest
from fungi_rag.workflow import FungiWorkflow


class MissingGradioApp:
    def launch(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("gradio is not installed. Run `python -m pip install -e .` first.")


def build_app():
    try:
        import gradio as gr
    except ImportError:
        return MissingGradioApp()

    settings = get_settings()
    brief_path = Path("examples/fungi_research_brief.yaml")
    brief_value = ""
    if brief_path.exists():
        brief_value = brief_path.read_text(encoding="utf-8")

    with gr.Blocks(title="Fungi RAG Learning System") as demo:
        gr.Markdown("# Fungi RAG Learning System")
        last_ask_result = gr.State({})
        with gr.Tab("Brief"):
            brief_text = gr.Code(
                label="YAML research brief",
                language="yaml",
                value=brief_value,
            )
            validate_button = gr.Button("Validate Brief")
            brief_status = gr.JSON(label="Validation")
            validate_button.click(validate_brief_ui, inputs=brief_text, outputs=brief_status)

        with gr.Tab("Corpus"):
            manifest_path = gr.Textbox(label="Source manifest", value="examples/source_manifest.yaml")
            download_button = gr.Button("Download Seed Academic Corpus")
            download_status = gr.JSON(label="Download status")
            gr.Markdown("### Project folders")
            background_path = gr.Textbox(
                label="Background information folder",
                value=str(settings.background_dir),
            )
            references_path = gr.Textbox(
                label="Bibliographic references folder",
                value=str(settings.references_dir),
            )
            ingest_background_button = gr.Button("Ingest Background Folder")
            ingest_references_button = gr.Button("Ingest References Folder")
            ingest_project_button = gr.Button("Ingest Background + References")
            project_ingest_status = gr.JSON(label="Project folder ingestion status")
            gr.Markdown("### Seed corpus")
            ingest_path = gr.Textbox(label="Seed corpus path to ingest", value=str(settings.source_raw_dir))
            ingest_button = gr.Button("Ingest Seed Corpus")
            ingest_status = gr.JSON(label="Seed ingestion status")
            download_button.click(download_sources_ui, inputs=manifest_path, outputs=download_status)
            ingest_button.click(ingest_ui, inputs=ingest_path, outputs=ingest_status)
            ingest_background_button.click(
                ingest_background_ui,
                inputs=background_path,
                outputs=project_ingest_status,
            )
            ingest_references_button.click(
                ingest_references_ui,
                inputs=references_path,
                outputs=project_ingest_status,
            )
            ingest_project_button.click(
                ingest_project_folders_ui,
                inputs=[background_path, references_path],
                outputs=project_ingest_status,
            )

        with gr.Tab("Research Run"):
            run_button = gr.Button("Create Codex Outline/Draft Packets")
            run_status = gr.JSON(label="Run status")
            run_button.click(run_research_ui, inputs=brief_text, outputs=run_status)

        with gr.Tab("Ask"):
            question = gr.Textbox(label="Learner question")
            ask_button = gr.Button("Retrieve Evidence and Generate Answer")
            answer = gr.JSON(label="Result")
            ask_button.click(ask_ui, inputs=question, outputs=[answer, last_ask_result])

        with gr.Tab("Evaluation"):
            evaluate_current_button = gr.Button("Evaluate Last Answer")
            current_eval_status = gr.JSON(label="Current Answer Evaluation")

            evaluate_button = gr.Button("Run Benchmark Evaluation")
            eval_status = gr.JSON(label="Benchmark Evaluation")

            evaluate_current_button.click(
                evaluate_current_answer_ui,
                inputs=last_ask_result,
                outputs=current_eval_status,
            )
            evaluate_button.click(evaluate_ui, outputs=eval_status)

    return demo


def validate_brief_ui(text: str) -> dict[str, Any]:
    try:
        brief = ResearchBrief.model_validate(yaml.safe_load(text))
        return {"ok": True, "brief": brief.model_dump(mode="json")}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def download_sources_ui(manifest_path: str) -> dict[str, Any]:
    manifest = load_manifest(Path(manifest_path))
    rows = SourceDownloader().download_manifest(manifest)
    return {"count": len(rows), "rows": rows}


def ingest_ui(path: str) -> dict[str, Any]:
    chunks = DocumentIngestor().ingest_path(Path(path))
    return {"chunks": len(chunks), "path": path}


def ingest_background_ui(path: str) -> dict[str, Any]:
    chunks = DocumentIngestor().ingest_path(Path(path), corpus_role="background")
    return {"background_chunks": len(chunks), "background_path": path}


def ingest_references_ui(path: str) -> dict[str, Any]:
    chunks = DocumentIngestor().ingest_path(Path(path), corpus_role="reference")
    return {"reference_chunks": len(chunks), "references_path": path}


def ingest_project_folders_ui(background_path: str, references_path: str) -> dict[str, Any]:
    ingestor = DocumentIngestor()
    background_chunks = ingestor.ingest_path(Path(background_path), corpus_role="background")
    reference_chunks = ingestor.ingest_path(Path(references_path), corpus_role="reference")
    return {
        "background_chunks": len(background_chunks),
        "reference_chunks": len(reference_chunks),
        "background_path": background_path,
        "references_path": references_path,
    }


def run_research_ui(text: str) -> dict[str, Any]:
    try:
        brief = ResearchBrief.model_validate(yaml.safe_load(text))
        state, paths = FungiWorkflow().run_research(brief)
        return {"state": state.model_dump(mode="json"), "paths": paths}
    except Exception as exc:
        return {"error": str(exc)}


def ask_ui(question: str) -> tuple[dict[str, Any], dict[str, Any]]:
    result = FungiWorkflow().ask(question)
    return result, result


def evaluate_current_answer_ui(last_result: dict[str, Any]) -> dict[str, Any]:
    if not last_result:
        return {"error": "No Ask result found yet. Ask a question first."}

    query = last_result.get("evidence", {}).get("query", "")
    answer = last_result.get("answer", "")
    evidence_items = last_result.get("evidence", {}).get("items", [])

    return evaluate_single_answer(query, answer, evidence_items)


def evaluate_ui() -> dict[str, Any]:
    return run_evaluation()


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch the Fungi RAG Gradio app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(list(argv) if argv is not None else None)
    app = build_app()
    app.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
