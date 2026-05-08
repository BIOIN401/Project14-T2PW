# Pipeline

The app and CLI paths use the same package modules.

## Extraction And Inference

Core extraction and inference helpers live in `t2pw.pipeline.pipeline`. The main stages are:

1. Stage 1 extraction with JSON repair retries.
2. Optional chunking for long source text.
3. Stage 2 inference and enrichment with graph QA feedback.
4. Merge of supported additions into the final pathway payload.
5. Draft graph and QA report generation.

## Post Processing

Normalization and hard gates live in `t2pw.pipeline.process_normalizer`. The important checks are:

- composite token cleanup before mapping and export,
- process references must resolve to declared entities,
- transport events attach directly to entity state nodes,
- scaffold proteins cannot leak into reaction modifiers,
- duplicate processes are collapsed before export.

## PWML Export

PWML is the primary export path:

```powershell
python scripts/run_pwml.py --in final.mapped.json --out-dir outputs --non-strict-db
```

The script delegates to `t2pw.pwml.writer`, which builds PWML IR, validates it, writes deterministic PWML XML, and runs PWML QA.

## Legacy SBML Export

SBML is retained as a legacy export path:

```powershell
python scripts/run.py --in final.json --out-dir outputs --no-llm-audit --no-sbml-overwatch
```

The script delegates to `t2pw.pipeline.pipeline`, which runs audit, patch application, ID mapping, SBML build, validation, and optional semantic overwatch.

## Verification

Before and after cleanup batches, run:

```powershell
pytest -q
ruff check src tests scripts
python -m py_compile src/t2pw/pipeline/pipeline.py src/t2pw/pwml/writer.py scripts/run.py scripts/run_pwml.py
```

Smoke-test the app manually:

```powershell
streamlit run src/t2pw/app/streamlit_app.py
```
