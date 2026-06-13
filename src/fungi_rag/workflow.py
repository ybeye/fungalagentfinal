from __future__ import annotations

from pathlib import Path

from fungi_rag.config import Settings, get_settings
from fungi_rag.export import Exporter
from fungi_rag.generator import Generator, build_generator
from fungi_rag.models import AgentState, EvidenceItem, EvidencePacket, GenerationRequest, ResearchBrief
from fungi_rag.retrieval import HybridRetriever
from fungi_rag.safety import REFUSAL, SafetyPolicy
from fungi_rag.utils import ensure_dir, slugify, stable_id, utc_now_iso


APPROVAL_STAGES = ["outline_approval", "draft_approval"]


class FungiWorkflow:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        retriever: HybridRetriever | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever
        self.generator = generator
        self.safety = SafetyPolicy("strict")

    def get_generator(self) -> Generator:
        if self.generator is None:
            self.generator = build_generator(
                self.settings.generator_backend,
                self.settings.enable_codex_cli,
            )
        return self.generator

    def ask(self, question: str, run_id: str | None = None) -> dict[str, object]:
        decision = self.safety.check(question)
        run_id = run_id or make_run_id("ask")
        output_dir = ensure_dir(self.settings.output_dir / run_id)
        if not decision.allowed:
            return {
                "run_id": run_id,
                "status": "refused",
                "answer": REFUSAL,
                "reason": decision.reason,
                "output_dir": str(output_dir),
            }
        background_retriever = HybridRetriever.from_settings(
            self.settings,
            corpus_role="background",
        )
        reference_retriever = HybridRetriever.from_settings(
            self.settings,
            corpus_role="reference",
        )
        background_packet, background_trace = background_retriever.retrieve(
            question,
            top_k=3,
            run_id=f"{run_id}-background",
        )
        reference_packet, reference_trace = reference_retriever.retrieve(
            question,
            top_k=3,
            run_id=f"{run_id}-reference",
        )
        packet = merge_packets(question, [background_packet, reference_packet], max_items=6)
        request = GenerationRequest(
            run_id=run_id,
            step="ask",
            task=f"Answer this learner question using the retrieved evidence: {question}",
            evidence=packet,
            output_dir=output_dir,
        )
        result = self.get_generator().generate(request)
        background_trace.generator_packet_path = result.prompt_path
        reference_trace.generator_packet_path = result.prompt_path
        return {
            "run_id": run_id,
            "status": result.status,
            "answer": result.text,
            "generation": result.model_dump(mode="json"),
            "evidence": packet.model_dump(mode="json"),
            "background_evidence": background_packet.model_dump(mode="json"),
            "reference_evidence": reference_packet.model_dump(mode="json"),
            "trace": {
                "background": background_trace.model_dump(mode="json"),
                "reference": reference_trace.model_dump(mode="json"),
            },
            "output_dir": str(output_dir),
        }

    def run_research(self, brief: ResearchBrief) -> tuple[AgentState, dict[str, str]]:
        run_id = make_run_id(brief.title)
        output_dir = ensure_dir(self.settings.output_dir / run_id)
        queries = plan_queries(brief)
        evidence_packets: list[EvidencePacket] = []
        traces = []
        retriever = self.retriever or HybridRetriever.from_settings(self.settings)
        for query in queries:
            packet, trace = retriever.retrieve(query, run_id=run_id)
            evidence_packets.append(packet)
            traces.append(trace)
        combined_evidence = merge_packets(brief.topic, evidence_packets)
        generator = self.get_generator()
        outline_result = generator.generate(
            GenerationRequest(
                run_id=run_id,
                step="outline",
                task=outline_task(brief),
                evidence=combined_evidence,
                output_dir=output_dir,
                safety_mode=brief.safety_mode,
            )
        )
        draft_result = generator.generate(
            GenerationRequest(
                run_id=run_id,
                step="draft",
                task=draft_task(brief, outline_result.text),
                evidence=combined_evidence,
                output_dir=output_dir,
                safety_mode=brief.safety_mode,
            )
        )
        for trace in traces:
            trace.generator_packet_path = outline_result.prompt_path
        status = "complete"
        if outline_result.status == "pending" or draft_result.status == "pending":
            status = "pending_codex"
        state = AgentState(
            run_id=run_id,
            brief=brief,
            queries=queries,
            evidence_packets=evidence_packets,
            outline=outline_result.text,
            draft=draft_result.text,
            citations=[item.citation() for item in combined_evidence.items],
            output_dir=str(output_dir),
            status=status,
        )
        paths = Exporter(output_dir).export_run(
            state=state,
            evidence=combined_evidence,
            generation_results=[outline_result, draft_result],
            traces=traces,
        )
        return state, paths


def make_run_id(label: str) -> str:
    return f"{slugify(label, 40)}-{stable_id(label, utc_now_iso(), length=8)}"


def plan_queries(brief: ResearchBrief) -> list[str]:
    seeds = [
        brief.topic,
        *brief.learning_objectives,
        *[f"{brief.topic} {section}" for section in brief.section_names],
        *brief.seed_references,
    ]
    normalized: list[str] = []
    seen: set[str] = set()
    for seed in seeds:
        clean = seed.strip()
        key = clean.lower()
        if clean and key not in seen:
            normalized.append(clean)
            seen.add(key)
        if len(normalized) >= brief.number_of_queries:
            break
    return normalized


def merge_packets(topic: str, packets: list[EvidencePacket], max_items: int = 12) -> EvidencePacket:
    seen: set[str] = set()
    items: list[EvidenceItem] = []
    for packet in packets:
        for item in packet.items:
            if item.chunk_id in seen:
                continue
            seen.add(item.chunk_id)
            items.append(item.model_copy(update={"citation_id": len(items) + 1}))
            if len(items) >= max_items:
                return EvidencePacket(query=topic, normalized_query=topic.lower(), items=items)
    return EvidencePacket(query=topic, normalized_query=topic.lower(), items=items)


def outline_task(brief: ResearchBrief) -> str:
    sections = "\n".join(f"- {section}: {count} paragraphs" for section, count in brief.paragraph_plan().items())
    objectives = "\n".join(f"- {objective}" for objective in brief.learning_objectives)
    return f"""Create a research-style learning outline.

Title: {brief.title}
Topic: {brief.topic}
Audience: {brief.audience}

Learning objectives:
{objectives}

Required sections:
{sections}
"""


def draft_task(brief: ResearchBrief, outline: str) -> str:
    outline_text = outline or "No accepted outline is available yet. Draft from the requested sections."
    return f"""Draft the learning module for "{brief.title}".

Use this outline if available:

{outline_text}

Write concise academic prose, include source citations, and preserve the safety boundary.
"""


def workflow_stage_names() -> list[str]:
    return [
        "validate_brief",
        "download_ingest_corpus",
        "plan_retrieval_queries",
        "retrieve_rank_evidence",
        "create_outline_codex_packet",
        *APPROVAL_STAGES,
        "create_draft_codex_packet",
        "safety_review",
        "reflection_review",
        "citation_audit",
        "export_artifacts",
    ]


def build_langgraph_workflow():
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return None

    graph = StateGraph(dict)

    def validate(state: dict) -> dict:
        return {**state, "stage": "validate_brief"}

    def retrieve(state: dict) -> dict:
        return {**state, "stage": "retrieve_rank_evidence"}

    def outline(state: dict) -> dict:
        return {**state, "stage": "outline_approval"}

    def draft(state: dict) -> dict:
        return {**state, "stage": "draft_approval"}

    def export(state: dict) -> dict:
        return {**state, "stage": "export_artifacts"}

    graph.add_node("validate_brief", validate)
    graph.add_node("retrieve_rank_evidence", retrieve)
    graph.add_node("outline_approval", outline)
    graph.add_node("draft_approval", draft)
    graph.add_node("export_artifacts", export)
    graph.set_entry_point("validate_brief")
    graph.add_edge("validate_brief", "retrieve_rank_evidence")
    graph.add_edge("retrieve_rank_evidence", "outline_approval")
    graph.add_edge("outline_approval", "draft_approval")
    graph.add_edge("draft_approval", "export_artifacts")
    graph.add_edge("export_artifacts", END)
    return graph.compile()
