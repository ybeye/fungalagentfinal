# SmolLM2 Fungi SFT Colab Bundle

This directory is a portable Colab handoff for continuing the SmolLM2 fungi RAG/tool-use fine-tuning run on a stronger GPU.

The notebook downloads the Hugging Face base model `HuggingFaceTB/SmolLM2-1.7B-Instruct` in Colab. It does not fine-tune the local GGUF file. The included checkpoint is the local LoRA checkpoint from the interrupted full run and can be used to resume from step 100.

Large runtime artifacts under `artifacts/` and the generated upload zip are intentionally gitignored. The local project directory and generated zip contain those files; a plain GitHub clone contains the notebook/config/packaging code and manifest only.

## Contents

- `smollm2_fungi_sft_colab.ipynb` - self-contained Colab notebook for QLoRA SFT, validation metric comparison, and lightweight behavior eval.
- `requirements-colab.txt` - pinned Python packages tested against the local trainer stack.
- `config/smollm2_colab_config.yaml` - conservative training defaults.
- `artifacts/prepared_data/` - TRL-ready JSONL train/val/test files.
- `artifacts/source_dataset/` - copied local dataset tree for provenance and optional reprocessing.
- `artifacts/evals/behavior_eval.jsonl` - fixed 80-case behavior eval set.
- `artifacts/reports/` - local inspection/preparation/smoke reports.
- `artifacts/checkpoints/checkpoint-100/` - resumable LoRA checkpoint from the interrupted full run.

## Current Local Run State

The local full run stopped before completion.

- Last saved checkpoint: `artifacts/checkpoints/checkpoint-100`
- Last checkpoint global step: `100 / 846`
- Last observed progress before stop: `109 / 846`
- Step-100 validation metrics: `eval_loss=0.6623`, `eval_mean_token_accuracy=0.8355`
- Smoke behavior eval did not pass safety/citation/uncertainty thresholds, so improvement is not yet proven.

## Upload Workflow

1. Build the upload zip from PowerShell:

   ```powershell
   .\colab_smollm2_fungi_sft\scripts\make_upload_zip.ps1
   ```

2. Upload `colab_smollm2_fungi_sft\smollm2_fungi_colab_bundle.zip` to Google Drive.

3. In Colab, create a GPU runtime, preferably L4, A100, or T4.

4. Unzip the bundle in Colab:

   ```python
   !unzip -q /content/drive/MyDrive/smollm2_fungi_colab_bundle.zip -d /content/
   ```

5. Open or upload `smollm2_fungi_sft_colab.ipynb`, set `PROJECT_DIR`, and run the cells.

## Notes

- The prepared JSONL files are the recommended training input in Colab.
- The raw source dataset is included for auditability and optional reprocessing, but the notebook does not need it for SFT.
- The original local GGUF is inference-only and is intentionally not included as a training input.
- Save Colab outputs to Google Drive if you want them to persist after the runtime shuts down.
