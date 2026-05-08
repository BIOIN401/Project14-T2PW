# Architecture

The package code lives under `src/t2pw`. The older flat `src/*.py` files remain during the migration, but new code should import from the package paths.

## Main Areas

- `t2pw.app`: Streamlit UI. `streamlit_app.py` should remain the top-level app orchestration point until the later UI decomposition pass.
- `t2pw.pipeline`: extraction, inference, graph QA, normalization, and legacy SBML orchestration.
- `t2pw.pwml`: PWML IR construction, deterministic writing, validation, rendering, and QA.
- `t2pw.sbml`: legacy SBML build, layout, rendering, and cleanup helpers.
- `t2pw.mapping`: PathBank/API/database/cache/routing and enrichment behavior.
- `t2pw.curation`: audit, patch application, pathway curation, and gap resolution behavior.
- `t2pw.llm`: OpenAI-compatible LLM client and prompt files.
- `t2pw.tools`: application-specific tool modules, including the PathWhiz converter.

## Entry Points

The root scripts are intentionally thin:

- `scripts/run.py` delegates to `t2pw.pipeline.pipeline.legacy_sbml_cli_main`.
- `scripts/run_pwml.py` delegates to `t2pw.pwml.writer.pwml_pipeline_cli_main`.

Real packaging entry points can be added later after the package move is complete.

## Utility Tools

Ruby parser utilities are kept under `tools/ruby`:

- `tools/ruby/pwml_parser.rb`
- `tools/ruby/sbml_parser.rb`
