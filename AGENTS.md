# AGENTS.md — bio-connectome-cl

Research project: does Drosophila visual-pathway connectome topology (FlyWire v783) reduce catastrophic forgetting in continual RL navigation? Source of truth: `PROJECT_SPEC.md`; agent constraints: `docs/agent_rules.md`; science background: `docs/SCIENTIFIC_OVERVIEW.md`, `docs/hypothesis.md`, `docs/experiment_plan.md`.

## Status: skeleton — implement before running
All `src/**/*.py` and `scripts/*.py` are stubs raising `NotImplementedError` with Russian TODO docstrings. Implement them; don't assume any pipeline step works until its `TODO` is resolved.

## Setup (Windows-first)
```powershell
python -m venv .venv; .\.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
- Python 3.11+, `torch==2.4.0`, `torch-geometric==2.5.3`, `stable-baselines3==2.3.2`, `gymnasium==0.29.1`. CUDA install order matters (torch from cu121 index first).
- No tests, no CI, no formatter config. Only linter: `ruff check --fix` (version in `requirements.txt`).

## Актуальное окружение (2026-09) — ВАЖНО, расходится с разделом Setup выше

> Раздел Setup выше описывает исходный замысел (torch 2.4.0 / cu121 / .venv) и **УСТАРЕЛ для этой машины**. Ниже — фактическое окружение, проверенное 2026-09-16. Не переустанавливать по старой инструкции!

- GPU: **NVIDIA RTX 5060 Ti 16GB** (Blackwell, sm_120), драйвер 595.79, CUDA 13.2. `nvidia-smi` работает.
- Python: **3.11.9, глобальная установка `c:\Python311\python.exe`** — `.venv` НЕ используется, установка глобальная.
- PyTorch: **`torch 2.11.0+cu128`**, `torchvision 0.26.0+cu128`. **Критично: cu121-колеса НЕ поддерживают Blackwell sm_120** — нужен cu128 и новее. Установка: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`.
- Прочее: torch-geometric 2.8.0.post1, stable-baselines3 2.3.2, gymnasium 0.29.1, numpy 2.4.6, networkx 3.6.1, pandas, scipy, matplotlib, seaborn.
- `requirements.txt` уже обновлён: `torch==2.11.0`, `torch-geometric==2.8.0.post1` + комментарий про cu128.
- Проверка CUDA: `torch.cuda.is_available()` → True, device `NVIDIA GeForce RTX 5060 Ti`, capability (12,0).

## Pipeline order (do not skip)
1. `python scripts/01_build_graphs.py [--rebuild]` → `data/processed/*.pt` + `results/metrics/graph_stats.json`. ВНИМАНИЕ: без `--rebuild` при наличии `.pt` пересборка ПРОПУСКАЕТСЯ; при пустом `data/raw/` скрипт ВИСИТ на download — не запускать без данных.
2. `python scripts/02_run_2k.py --all` (or `--arch bio,rewired,random_er,unsigned,mlp --seed N`) → `results/metrics/2k_*.json` (15 runs: 5 arch × 3 seeds). ВНИМАНИЕ: текущие `2k_*.json` — **реальные** результаты (15/15 на фикшенном коде, config_hash 7d6f4e48); повторный запуск их перезапишет.
3. `python scripts/04_analyze.py --scale 2k` → `2k_summary.json`
4. `python scripts/05_make_figures.py --scale 2k` → `fig1-3.png`
5. `python scripts/03_run_8k.py --arch bio,rewired [--dry-run] [--seed 1]` → `results/metrics/8k_*.json` (2 runs; always `--dry-run` on 16GB box the day before a 96GB session)
6. `04_analyze --scale 8k`, then `05_make_figures --scale all`
- Machine mapping: debug/mini-graph (300 nodes) on 8GB; full 2K on 16GB (~2–3h/run); 8K bio+rewired only, 1 seed, ~6h/run on 96GB server. 8K uses `hidden_dim 128`, 2K uses `64` (`config/models.yaml`).
- Raw CSVs (`data/raw/`, ~290MB) are gitignored — данные скачаны вручную (Codex), состав см. в `data/data_registry.md`; автоматический download не использовался.

## Running from a fresh session
- Активация окружения не нужна — Python глобальный (`C:\Python311\python.exe`). Нагруженный прогон из другой сессии:
  1. `python -u scripts/02_run_2k.py --all` (15 прогонов: 5 arch × 3 seed, ~2-3 ч/прогон) или выборочно `python -u scripts/02_run_2k.py --arch bio,rewired,random_er,unsigned,mlp --seed 1`
  2. `python scripts/04_analyze.py --scale 2k`
  3. `python scripts/05_make_figures.py --scale 2k`
- `--smoke` = 2048 шагов на задачу, `--dry-run` = проверка без обучения, env-var `SMOKE_TIMESTEPS=<N>` переопределяет шаги без правки config. Для 8K: `python -u scripts/03_run_8k.py --arch bio,rewired --dry-run` (всегда `--dry-run` на 16GB за день до 96GB сессии).
- Текущие `results/metrics/2k_*.json` — **реальные результаты** (15/15 на фикшенном коде, config_hash 7d6f4e48); `8k_*.json` — dry-run-заглушки, их перезапишет полный прогон. Реальные прогоны перезапишутся при повторном запуске; не удалять `results/` руками.
- После установки/переустановки зависимостей: `pip install -r requirements.txt` (torch ставится с cu128 индекса отдельно — см. комментарий в requirements).

## Conventions (enforced, differ from defaults)
- All hyperparams live in `config/*.yaml` (`graphs/env/models/training/viz.yaml`) — never hardcode; load via `yaml.safe_load`.
- Code: PEP 8, type hints, Google-style docstrings, `snake_case` funcs/files, `PascalCase` classes, **absolute imports** (`from src.data.build_graph import ...`), **`pathlib.Path` only** (no raw slash strings), **comments/docstrings in Russian**.
- Reproducibility: seed everything (`numpy`, `torch`, `random`, `gymnasium`); seeds `[1,2,3]` in `config/training.yaml`; every run writes `results/metrics/{run_id}.json` in the schema from `docs/agent_rules.md` (run_id e.g. `bio_2k_seed1`, includes `config_hash`, `timestamp`).
- Commits: prefixes `feat: fix: exp: viz: docs:`. Never delete/overwrite `results/` without confirmation; never change hypothesis or arch/task set (`bio/rewired/random_er/unsigned/mlp`; `task1_goto → task2_avoid`) without confirmation.
- Graph validation after build (from `data/data_registry.md`): no isolated nodes, no NaN weights, `sensory_mask ≥50`, `motor_mask ≥20`, bio modularity >0.3.
- Domain facts agents get wrong: edge weights are **signed** (`+syn_count` for acetylcholine/glutamate, `-syn_count` for GABA; `unsigned` variant = `|w|`); GNN topology (`edge_index`) is **frozen**, only message weights + proj/readout train; message-passing steps per scale (`n_steps` = 7 для 2K, 5 для 8K, `config/models.yaml`), веса рёбер нормируются до mean|w| ≈ 0.5; obs = 8 lidar rays + goal angle (dim 9), 3 discrete actions; PPO 200K steps/task, no EWC/replay; forgetting = `R(T1|after T1) − R(T1|after T2)`; stats = Wilcoxon rank-sum on 3 seeds (2K), qualitative direction check on 8K.
- Viz (`config/viz.yaml`): PNG 300dpi, `DejaVu Sans` 12/14/10, wide `[10,4]` / square `[6,6]`, fixed arch colors (`bio #1f77b4`, `rewired #ff7f0e`, `random_er #2ca02c`, `unsigned #d62728`, `mlp #9467bd`), 95% CI error bars, gray grid `alpha=0.3`.

## OpenCode wiring (from `C:\Github\ACCOS`)
- Config: `opencode.json` (MCP: `filesystem` scoped to repo root, `sequential-thinking`, `memory` → `.opencode/memory.jsonl`, `git`, `context7`, `fetch`, `arxiv` (`arxiv-mcp-search` from shared `C:\Github\mcpservers\.venv` — handshake verified); `wordex` intentionally omitted — no office docs in this project). LSP: `pyright` (same venv). Plugin: `@mohak34/opencode-notifier` (Windows OK; only `.wav` sounds). Behavior: `.opencode/instructions.md`.
- Temp files/scripts go only to `temp/` (gitignored) — never system temp.
- Skill `ponytail` (`skills/ponytail/SKILL.md`, commands `/ponytail`, `/ponytail-review`) is on-demand only. Its "no comments" rule conflicts with this repo's Russian-comments requirement — repo rule wins unless user overrides.

## Access (all agents incl. orchestrator)
- Full access strictly within `C:\Github\bio-connectome-cl\`. Never read, write, or execute outside it (except MCP server paths already fixed in `opencode.json`).
- Temp files/scripts go only to `temp/` (gitignored) — never system temp.

## Workflow: multi-agent (mandatory)
- All non-trivial work runs via subagents (`Task` tool); the main thread orchestrates and never implements directly.
- Orchestrator splits work into independent units and runs worker agents in parallel (one message, several `Task` calls). Sequential only when a step depends on a previous result.
- Each worker returns evidence (files changed, key numbers, errors), not prose summaries.
- A dedicated reviewer agent re-checks every worker result (code runs, metrics match schema, docs consistent) before it reaches the main thread. Reviewer verdict: accept / send back with exact defects.
- On defects the orchestrator relaunches the same worker with the reviewer's findings — no silent self-fixes in main thread.
- Exactly one `in_progress` todo at a time per agent; orchestrator tracks the whole list via `TodoWrite`.

## Autonomy (all agents incl. orchestrator)
- Maximum autonomy: never block on user input. If a step needs user decision/confirmation, log it as pending, defer it, and continue all other independent work.
- Batch deferred questions into a single `question` call only when blocked work accumulates; never ask about what config/docs/code already answer.
- Prefer reversible default + log assumption over waiting; record choice in code comment/commit message.

## Data preservation (all agents incl. orchestrator)
- Never delete files directly. Commit first (`git add` + `git commit`), only then delete — so history preserves the data.
- Applies to all tracked/untracked project files; exception: `temp/` (gitignored scratch space) may be cleaned without commit.

## Failure handling (from `docs/agent_rules.md`)
- OOM → `batch_size` 64→32→16, `hidden` 128→64, then `torch.amp.autocast`; never silently ignore.
- NaN loss → `lr` 3e-4→1e-4, keep `max_grad_norm=0.5`.
- Graph too big → step down 8K→5K→2K and log it.
- Model not learning → check weight signs, sensory/motor masks, reduce `n_steps` by 1-2 (2K: 7→5, 8K: 5→4).
- Note: `.gitignore` ignores `data/raw/`, `data/processed/`, `results/logs/`, `results/checkpoints/`, `*.pt`, `.venv/` — only `.gitkeep` placeholders are committed; `*.pt` graphs won't show in git.
