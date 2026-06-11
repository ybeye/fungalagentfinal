from __future__ import annotations

import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from fungi_rag.citation import CitationAuditor, normalize_numeric_citations
from fungi_rag.models import GenerationRequest, GenerationResult
from fungi_rag.safety import SafetyPolicy
from fungi_rag.utils import atomic_write_text, write_json


class Generator(ABC):
    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError


class CodexBridgeGenerator(Generator):
    def __init__(self) -> None:
        self.auditor = CitationAuditor()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        task_dir = request.output_dir / "codex_tasks"
        task_dir.mkdir(parents=True, exist_ok=True)
        base = task_dir / request.step
        prompt_path = base.with_suffix(".prompt.md")
        evidence_path = base.with_suffix(".evidence.json")
        schema_path = base.with_suffix(".schema.json")
        response_path = base.with_suffix(".response.md")

        atomic_write_text(prompt_path, self._render_prompt(request))
        write_json(evidence_path, request.evidence.model_dump(mode="json"))
        write_json(schema_path, request.response_schema or default_response_schema(request.step))

        if not response_path.exists():
            return GenerationResult(
                status="pending",
                step=request.step,
                prompt_path=str(prompt_path),
                evidence_path=str(evidence_path),
                schema_path=str(schema_path),
                response_path=str(response_path),
                validation_errors=[
                    "Codex response is pending. Complete the response file, then rerun validation."
                ],
            )

        text = response_path.read_text(encoding="utf-8")
        errors = self.validate_text(text, request)
        return GenerationResult(
            status="accepted" if not errors else "invalid",
            step=request.step,
            prompt_path=str(prompt_path),
            evidence_path=str(evidence_path),
            schema_path=str(schema_path),
            response_path=str(response_path),
            text=text,
            validation_errors=errors,
        )

    def validate_text(self, text: str, request: GenerationRequest) -> list[str]:
        available = {item.citation_id for item in request.evidence.items}
        text = normalize_numeric_citations(text, available)
        errors = SafetyPolicy(request.safety_mode).validate_response(text)
        audit = self.auditor.audit(text, request.evidence, require_all=False)
        if audit.unknown_ids:
            errors.append(f"Unknown citation ids: {audit.unknown_ids}")
        if not audit.cited_ids and request.evidence.items:
            errors.append("Response must include at least one citation from the evidence packet.")
        if audit.unsupported_sentences:
            errors.append(
                "Long unsupported sentences need citations: "
                + "; ".join(audit.unsupported_sentences[:3])
            )
        return errors

    def add_fallback_citation(self, text: str, request: GenerationRequest) -> str:
        if not text.strip() or not request.evidence.items:
            return text
        text = self.fix_leading_citations(text, request)
        audit = self.auditor.audit(text, request.evidence, require_all=False)
        if audit.cited_ids and not audit.unsupported_sentences:
            return text
        clean = text.rstrip()
        citation = f"[{request.evidence.items[0].citation_id}]"
        return self.cite_uncited_sentences(clean, citation)

    def fix_leading_citations(self, text: str, request: GenerationRequest) -> str:
        available = {item.citation_id for item in request.evidence.items}
        fixed_lines = []
        for line in text.splitlines():
            match = re.match(r"^\s*\[(\d+)\]\s+(.+)$", line.strip())
            if not match:
                fixed_lines.append(line)
                continue
            citation_id = int(match.group(1))
            if citation_id not in available:
                fixed_lines.append(line)
                continue
            fixed_lines.append(self.cite_uncited_sentences(match.group(2), f"[{citation_id}]"))
        return "\n".join(fixed_lines)

    def cite_uncited_sentences(self, text: str, citation: str) -> str:
        pieces = re.split(r"(?<=[.!?])\s+", text)
        cited = []
        for piece in pieces:
            clean = piece.strip()
            if not clean:
                continue
            if self.auditor.citation_pattern.search(clean):
                cited.append(clean)
            elif clean[-1:] in {".", "!", "?"}:
                cited.append(f"{clean[:-1].rstrip()} {citation}{clean[-1]}")
            else:
                cited.append(f"{clean} {citation}")
        return " ".join(cited)

    def _render_prompt(self, request: GenerationRequest) -> str:
        evidence_lines = []
        for item in request.evidence.items:
            locator = item.url or item.path or item.source_id
            evidence_lines.append(
                "\n".join(
                    [
                        f"[{item.citation_id}] {item.title}",
                        f"Source: {locator}",
                        f"Chunk: {item.chunk_id}",
                        f"Support: {item.confidence_note}",
                        f"Snippet: {item.snippet}",
                    ]
                )
            )
        evidence_block = "\n\n".join(evidence_lines) or "No evidence was retrieved."
        return f"""# Codex RAG Task: {request.step}

You are generating academic learning content about fungi. Use only the evidence
packet below. Do not browse, invent sources, or cite anything outside the numbered
evidence items. Cite claims with bracketed source IDs such as [1].

Safety boundary: refuse edibility, dosage, medical decisions, field identification,
and "safe to eat" decisions. Academic discussion of traits, ecology, toxicology,
and risks is allowed when source-backed.

## User Task

{request.task}

## Retrieved Evidence

{evidence_block}

## Output Requirements

- Write Markdown unless the schema requests JSON.
- Include citations from the numbered evidence packet.
- If the evidence is insufficient, say what is missing instead of guessing.
- Keep the answer clear and natural, like a student explaining the result in a class demo.
- Use short paragraphs instead of a polished essay style.
"""

    def _render_model_prompt(self, request: GenerationRequest) -> str:
        evidence_lines = []
        for item in request.evidence.items:
            evidence_lines.append(f"[{item.citation_id}] {item.snippet}")
        evidence_block = "\n".join(evidence_lines) or "No evidence was retrieved."
        return f"""You are helping with a student NLP project about fungi.
Answer only from the evidence below. Do not add outside facts.
Use a normal student explanation, not a fancy essay.
Keep it to one short paragraph.
Put citation IDs like [1] after claims.
If the question asks for eating advice, dosage, medical decisions, or field identification, refuse briefly.

Question: {request.task}

Evidence:
{evidence_block}

Answer:
"""


class CodexCliGenerator(CodexBridgeGenerator):
    def __init__(self, enabled: bool = False) -> None:
        super().__init__()
        self.enabled = enabled

    def generate(self, request: GenerationRequest) -> GenerationResult:
        result = super().generate(request)
        if not self.enabled or result.status != "pending":
            return result
        codex_path = shutil.which("codex")
        if not codex_path:
            result.validation_errors.append("codex executable was not found on PATH.")
            return result
        try:
            completed = subprocess.run(
                [codex_path, "exec", Path(result.prompt_path).read_text(encoding="utf-8")],
                cwd=str(request.output_dir),
                text=True,
                capture_output=True,
                timeout=180,
                check=False,
            )
            if completed.returncode != 0:
                result.status = "failed"
                result.validation_errors.append(completed.stderr or "codex CLI failed.")
                return result
            Path(result.response_path).write_text(completed.stdout, encoding="utf-8")
            return super().generate(request)
        except Exception as exc:  # noqa: BLE001 - CLI adapter must fail closed.
            result.status = "failed"
            result.validation_errors.append(str(exc))
            return result


class TransformersGenerator(CodexBridgeGenerator):
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
        adapter_path: str | Path | None = None,
        device: str = "auto",
        max_new_tokens: int = 220,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None
        self._torch = None

    def generate(self, request: GenerationRequest) -> GenerationResult:
        task_dir = request.output_dir / "codex_tasks"
        task_dir.mkdir(parents=True, exist_ok=True)
        base = task_dir / request.step
        prompt_path = base.with_suffix(".prompt.md")
        evidence_path = base.with_suffix(".evidence.json")
        schema_path = base.with_suffix(".schema.json")
        response_path = base.with_suffix(".response.md")

        prompt = self._render_model_prompt(request)
        atomic_write_text(prompt_path, prompt)
        write_json(evidence_path, request.evidence.model_dump(mode="json"))
        write_json(schema_path, request.response_schema or default_response_schema(request.step))

        try:
            text = self._generate_text(prompt)
        except Exception as exc:  # noqa: BLE001 - local model adapter should fail closed.
            return GenerationResult(
                status="failed",
                step=request.step,
                prompt_path=str(prompt_path),
                evidence_path=str(evidence_path),
                schema_path=str(schema_path),
                response_path=str(response_path),
                validation_errors=[f"Transformers generation failed: {exc}"],
            )

        text = self._clean_answer(text)
        available = {item.citation_id for item in request.evidence.items}
        text = normalize_numeric_citations(text, available)
        atomic_write_text(response_path, text)
        errors = SafetyPolicy(request.safety_mode).validate_response(text)
        return GenerationResult(
            status="accepted" if not errors else "invalid",
            step=request.step,
            prompt_path=str(prompt_path),
            evidence_path=str(evidence_path),
            schema_path=str(schema_path),
            response_path=str(response_path),
            text=text,
            validation_errors=errors,
        )

    def _generate_text(self, prompt: str) -> str:
        self._load_model()
        tokenizer = self._tokenizer
        model = self._model
        torch = self._torch
        messages = [{"role": "user", "content": prompt}]
        try:
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        encoded = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                temperature=0.2,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0][encoded["input_ids"].shape[-1] :]
        return self._clean_model_text(tokenizer.decode(new_tokens, skip_special_tokens=True))

    def _render_model_prompt(self, request: GenerationRequest) -> str:
        evidence_lines = []
        ordered_items = sorted(
            request.evidence.items,
            key=lambda item: 0 if item.metadata.get("corpus_role") == "reference" else 1,
        )
        for index, item in enumerate(ordered_items, start=1):
            role = item.metadata.get("corpus_role", "source")
            evidence_lines.append(f"Evidence item {index} ({role}): {item.snippet}")
        evidence_block = "\n".join(evidence_lines) or "No evidence was retrieved."
        return f"""You are helping with a student NLP project about fungi.
Answer only from the evidence below. Do not add outside facts.
Use a normal student explanation, not a fancy essay.
Keep it to two or three complete sentences.
Use reference evidence first when it directly answers the question.
Do not copy the evidence word-for-word.
Do not start with evidence item numbers, bracket labels, or phrases like "Fungi help fungi."
Do not include bracket citations in the answer. The app shows the evidence separately.
If the question asks for eating advice, dosage, medical decisions, or field identification, refuse briefly.

Question: {request.task}

Evidence:
{evidence_block}

Answer:
"""

    def _load_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install local model dependencies with `python -m pip install -e '.[local-llm]'`."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if self.adapter_path is not None:
            if not self.adapter_path.exists():
                raise RuntimeError(f"LoRA adapter path does not exist: {self.adapter_path}")
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "Install PEFT to load the fine-tuned LoRA adapter: `python -m pip install peft`."
                ) from exc
            model = PeftModel.from_pretrained(model, str(self.adapter_path))
        target_device = self._pick_device(torch)
        model = model.to(target_device)
        model.eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model

    def _pick_device(self, torch) -> str:  # noqa: ANN001
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _clean_model_text(text: str) -> str:
        cleaned = text.strip()
        for prefix in ["assistant\n", "assistant:", "assistant"]:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
        return cleaned

    def _clean_answer(self, text: str) -> str:
        cleaned_lines = []
        for line in text.strip().splitlines():
            clean = re.sub(r"^\s*(?:\[\d+\]|Evidence item \d+[:.)-]?)\s*", "", line).strip()
            if clean:
                cleaned_lines.append(clean)
        cleaned = " ".join(cleaned_lines).strip()
        if not cleaned:
            return cleaned
        sentence_parts = re.findall(r".*?[.!?](?=\s|$)", cleaned)
        if sentence_parts:
            return " ".join(sentence.strip() for sentence in sentence_parts[:3])
        return cleaned

def default_response_schema(step: str) -> dict[str, object]:
    return {
        "type": "object",
        "description": f"Optional structured response for {step}. Markdown response is accepted.",
        "properties": {
            "content": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["content"],
    }


def build_generator(backend: str = "codex_bridge", enable_codex_cli: bool = False) -> Generator:
    if backend == "codex_cli":
        return CodexCliGenerator(enabled=enable_codex_cli)
    if backend == "transformers":
        from fungi_rag.config import get_settings

        settings = get_settings()
        return TransformersGenerator(
            model_name=settings.hf_model,
            adapter_path=settings.hf_adapter_path,
            device=settings.hf_device,
            max_new_tokens=settings.hf_max_new_tokens,
        )
    return CodexBridgeGenerator()
