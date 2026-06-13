# Fungi RAG Learning System

This project is a local RAG system for learning about fungi. It focuses on the
retrieval steps we covered in class:

- source download and provenance tracking
- extraction, normalization, chunking, and deduplication
- local dense retrieval plus BM25 keyword retrieval
- reciprocal-rank fusion and diversity filtering
- evidence packets with source IDs and snippets
- file based prompt/response packets for generation
- citation auditing, safety checks, exports, and evaluation

The default path does not require `OPENAI_API_KEY`. Generation writes file-based
task packets under `outputs/<run_id>/codex_tasks/`.

## Clone And Setup

Use Python 3.12 or newer. The commands below work on macOS/Linux shells and
PowerShell unless otherwise noted:

```bash
git clone https://github.com/ybeye/fungalagentfinal.git
cd fungalagentfinal
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,eval,local-llm]"
python -m pip install peft
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

The app stores runtime data under relative project paths by default:
`data/`, `outputs/`, and `data/chroma/`. Override these with `.env` settings
when needed.


## Commands and Local LLM Setup

Create a local .env file from the example configuration:

```bash
cp .env.example .env
```

The `.env` file should contain:
```text
FUNGI_GENERATOR_BACKEND=transformers
FUNGI_HF_MODEL=HuggingFaceTB/SmolLM2-1.7B-Instruct
FUNGI_HF_ADAPTER_PATH=models/smollm2-fungi-checkpoint-800
```

## Add the Fine-Tuned Adapter
The fine-tuned LoRA adapter is not included in the GitHub repository, but we will include it in the final submission. 
Create a folder with the name models in the main directory, and place smollm2-fungi-checkpoint-800 within it. 
Place the adapter folder at:
```text
models/
└── smollm2-fungi-checkpoint-800/
```
If the adapter folder is missing, the application will most likely fail.

```bash
python -m fungi_rag.sources download
python -m fungi_rag.ingest data/sources/raw
python -m fungi_rag.app --host 127.0.0.1 --port 7860
```

Then open `http://127.0.0.1:7860`.

The first run may download the configured local embedding model. No API key is
required for the default file based generation path and local embedding path.
Open the Gradio interface in your browser and test the system using questions like: What role do fungi play in decomposition?

## Safety Boundary

The system refuses edibility, dosage, medical decisions, field identification, and
"is this safe to eat" questions. It allows academic discussion of fungal traits,
ecology, toxicity, and risk with expert-verification guidance.
