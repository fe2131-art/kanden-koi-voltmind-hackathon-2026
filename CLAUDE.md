# Safety View Agent - Coding Agent Guide

This file is for coding agents working in this repository.
Human-facing project documentation lives in [README.md](./README.md) and [docs/README.md](./docs/README.md).
Do not treat this file as the main user documentation.

## License

The original source code, documentation, and configuration files in this repository
are licensed under Apache License 2.0. See `LICENSE` and `NOTICE` for the exact terms.

Third-party models, checkpoints, datasets, and external repositories are not relicensed
by this repository. Use them under their original licenses and terms.

## Repository Summary

Safety View Agent is a LangGraph-based multimodal safety monitoring pipeline.
It combines:

- RGB frames
- audio
- depth estimation
- infrared frames
- temporal change analysis
- SAM3 segmentation

to produce:

- per-frame safety assessments
- grounded critical regions
- a rolling `BeliefState`
- serialized results for the Demo UI

## What Matters Most Before You Change Anything

### 1. `uv sync` depends on external local repos

`pyproject.toml` expects these editable local dependencies to exist under `external/`:

- `external/Depth-Anything-3`
- `external/sam3`
- `external/vllm-omni`

If they are missing, `uv sync --extra dev` will fail.

### 2. `configs/default.yaml` is team-environment-biased

The checked-in defaults assume:

- `data.mode: "inspesafe"`
- `llm.provider / vlm.provider / alm.provider: "vllm"`
- `agent.enable_sam3: true`
- `sam3.checkpoint_path` points to a team-local path

On a fresh machine, expect config edits before successful execution.

### 3. `agent.enable_sam3` is the real SAM3 execution switch

The `sam3:` section holds analyzer configuration.
Actual enable/disable behavior is controlled by `agent.enable_sam3`.

### 4. The Demo UI depends on serialized outputs, not direct inference hooks

The browser UI reads streamed frame results derived from:

- `data/perception_results/manifest.json`
- `data/perception_results/frames/*.json`

`src/apps/server.py` polls those files and serves a WebSocket stream on port `8010`.

## Key Files

- `src/run.py`
  - main entry point
  - config and prompt loading
  - input preparation for `manual` and `inspesafe`
  - analyzer initialization
  - result persistence
  - TTS integration
- `src/safety_agent/agent.py`
  - LangGraph definition
  - multimodal fan-out / fan-in
  - belief update and final action selection
- `src/safety_agent/modality_nodes.py`
  - vision, audio, depth, infrared, temporal, and SAM3 analyzers
- `src/safety_agent/schema.py`
  - Pydantic schemas for IR, belief state, and assessment outputs
- `configs/default.yaml`
  - runtime behavior, providers, modalities, data mode
- `configs/prompt.yaml`
  - prompts for modality analysis, belief update, and final safety assessment
- `src/apps/server.py`
  - WebSocket bridge for the Demo UI
- `src/apps/App.jsx`
  - React frontend for visualization
- `docs/`
  - human-facing project documentation

## Input Modes

### `manual`

Uses either:

- video files under `data/videos/`, or
- existing frames under `data/frames/`

If a video exists, `run.py` uses lazy frame extraction to overlap extraction and inference.

### `inspesafe`

Uses:

- `data.inspesafe.dataset_path`
- `data.inspesafe.session`

to resolve an InspecSafe-V1 session and automatically prepare RGB, infrared, and audio inputs.

## Main Outputs

Primary runtime outputs:

```text
data/
├── perception_results/
│   ├── manifest.json
│   └── frames/*.json
├── results_archive/
├── frames/
├── depth/
├── infrared_frames/
├── sam3_masks/
├── voice/
└── flow.md
```

Notes:

- each new run archives the previous `data/perception_results/` into `data/results_archive/`
- `assessment.safety_status` may be synthesized into `data/voice/*.wav`
- the Demo UI consumes newly written frame results incrementally

## Common Commands

### Core setup

```bash
uv sync --extra dev
```

### Video generation only

```bash
uv sync --extra video_generation
```

### Demo UI dependencies

```bash
uv sync --extra dev --extra demo
cd src/apps
npm install
```

### Tests

```bash
uv run pytest tests/ -v
```

### Main pipeline

```bash
uv run python src/run.py
```

### Demo backend

```bash
uv run python src/apps/server.py
```

### Demo frontend

```bash
cd src/apps
npm run build
npm run preview
```

## Implementation Notes For Agents

### Assessments are not just plain summaries

`determine_next_action_llm()` returns `ActionWithGrounding`, not only a flat assessment.
Keep both of these aligned:

- `assessment`
- `grounded_critical_points`

### UI overlays use vision critical points

The Demo UI highlights bounding boxes from:

- `vision_analysis.critical_points[*].normalized_bbox`

not from `grounded_critical_points`.
If you change grounding behavior, verify that UI-visible region IDs still line up with `assessment.target_region`.

### Output schema changes propagate widely

If you change output structure, check at minimum:

- `src/safety_agent/schema.py`
- `src/safety_agent/agent.py`
- `src/run.py`
- `src/apps/server.py`
- `src/apps/App.jsx`
- relevant tests under `tests/`
- human-facing docs in `README.md` and `docs/`

### `run.py` aggressively manages output directories

On a new run, previous perception results are archived and several `data/` subdirectories are cleaned.
Do not assume old intermediate files remain in place during execution.

## Frequent Failure Modes

- `uv sync` fails
  - usually missing `external/` repositories
- session not found
  - often still using `data.mode: "inspesafe"` with invalid local dataset settings
- `Connection refused`
  - expected vLLM servers are not running
- `Sam3Analyzer: model load failed`
  - patch, checkpoint path, or dependency issue
- `ffprobe not available` / `ffmpeg not available`
  - system package missing

For human-facing troubleshooting details, use [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md).

## When Updating Documentation

- Keep human-facing setup and usage details in `README.md` and `docs/`
- Keep this file concise and agent-oriented
- If you change core behavior, update the relevant human docs as part of the same change

## Human-Facing Docs

- [README.md](./README.md)
- [docs/README.md](./docs/README.md)
- [docs/SETUP.md](./docs/SETUP.md)
- [docs/QUICK_START.md](./docs/QUICK_START.md)
- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)
- [docs/DEMO_APP.md](./docs/DEMO_APP.md)
- [docs/INSPESAFE_INTEGRATION.md](./docs/INSPESAFE_INTEGRATION.md)
- [docs/EXTENDING.md](./docs/EXTENDING.md)
- [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md)
- [docs/UV_VENV.md](./docs/UV_VENV.md)
