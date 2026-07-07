# Dev notebook VGGT-Omega judge cell

Paste these constants into the top config cell of `main.ipynb`, then paste the run cell below immediately after the `geometry_filter.py` cell.

The judge consumes `ASPAN_OUTPUT_DIR / 'vggt_candidates_manifest.jsonl'`; it does **not** rerun ASpanFormer. It runs VGGT-Omega only on ASpan-passed candidate pairs and writes both an all-judged manifest and a true-match-only manifest.

## Add to the config cell

```python
# VGGT-Omega judging stage. This starts from aspanfilter.py outputs; it does not rerun ASpanFormer.
RUN_VGGT_JUDGE = True
VGGT_JUDGE_SCRIPT = DRIVE_ROOT / 'vggt_signals.py'
VGGT_OMEGA_CHECKPOINT_DRIVE = DRIVE_ROOT / 'weights/VGGT-Omega/vggt_omega_1b_512.pt'
VGGT_OMEGA_CHECKPOINT_LOCAL = LOCAL_ROOT / 'vggt_omega_1b_512.pt'
VGGT_OUTPUT_DIR = LOCAL_ROOT / 'vggt_output'
VGGT_MAX_RES = 384
VGGT_GLOBAL_SIM_THRESHOLD = 0.90
VGGT_POSE_SHIFT_THRESHOLD = 0.10
VGGT_MAX_PAIRS = None  # set to a small integer such as 5 for first smoke tests
# Resume only helps if the previous VGGT_OUTPUT_DIR still exists. If you rerun the
# staging/reset cell with RESET_LOCAL_ROOT=True, local VGGT outputs are deleted.
VGGT_RESUME = True
```

## Paste under the aspanfilter cell

```python
# Run VGGT-Omega judge on ASpan-passed candidate pairs.
# Input:  ASPAN_OUTPUT_DIR / 'vggt_candidates_manifest.jsonl'
# Output: VGGT_OUTPUT_DIR / 'vggt_judged_manifest.jsonl'
#         VGGT_OUTPUT_DIR / 'true_matches_manifest.jsonl'
#         VGGT_OUTPUT_DIR / 'vggt_judge_summary.json'
if RUN_VGGT_JUDGE:
    run_command([sys.executable, '-m', 'pip', 'install', '-q', 'git+https://github.com/facebookresearch/vggt-omega.git'])

    ensure_dir(VGGT_OUTPUT_DIR)
    if not VGGT_JUDGE_SCRIPT.exists():
        raise FileNotFoundError(f'Missing VGGT judge script: {VGGT_JUDGE_SCRIPT}')
    if not VGGT_OMEGA_CHECKPOINT_DRIVE.exists():
        raise FileNotFoundError(f'Missing VGGT-Omega checkpoint: {VGGT_OMEGA_CHECKPOINT_DRIVE}')
    copy_file(VGGT_OMEGA_CHECKPOINT_DRIVE, VGGT_OMEGA_CHECKPOINT_LOCAL)

    vggt_input_manifest = ASPAN_OUTPUT_DIR / 'vggt_candidates_manifest.jsonl'
    if not vggt_input_manifest.exists():
        raise FileNotFoundError(f'Missing aspanfilter output manifest: {vggt_input_manifest}')

    cmd = [
        sys.executable,
        VGGT_JUDGE_SCRIPT,
        '--input-manifest', vggt_input_manifest,
        '--output-dir', VGGT_OUTPUT_DIR,
        '--checkpoint', VGGT_OMEGA_CHECKPOINT_LOCAL,
        '--global-sim-threshold', str(VGGT_GLOBAL_SIM_THRESHOLD),
        '--pose-shift-threshold', str(VGGT_POSE_SHIFT_THRESHOLD),
        '--max-res', str(VGGT_MAX_RES),
        '--progress-every', 1,
    ]
    if VGGT_RESUME:
        cmd.append('--resume')
    if VGGT_MAX_PAIRS is not None:
        cmd += ['--max-pairs', str(VGGT_MAX_PAIRS)]

    run_command(cmd, cwd=LOCAL_ROOT)

    for path in [
        VGGT_OUTPUT_DIR / 'vggt_judged_manifest.jsonl',
        VGGT_OUTPUT_DIR / 'true_matches_manifest.jsonl',
        VGGT_OUTPUT_DIR / 'vggt_judge_summary.json',
    ]:
        print(path, 'exists=', path.exists())
        if path.suffix == '.jsonl' and path.exists():
            print('rows=', count_jsonl(path) if 'count_jsonl' in globals() else sum(1 for line in path.open() if line.strip()))
else:
    print('RUN_VGGT_JUDGE=False; skipping VGGT-Omega judge')
```

## Decision rule

`vggt_signals.py` (Step 3) records all signals into `vggt_judged_manifest.jsonl`
and emits a provisional `true_match` based on its internal default thresholds.

**The paper's actual decision rule is applied by `pose_scoring.py` (Step 4)**:
- `aspan_2d_inlier_ratio >= 0.65` (Filter 2)
- `pose_component_score <= 2.13` (Filter 3)

Run `pose_scoring.py` on `vggt_judged_manifest.jsonl` to get the final
`pose_scored_manifest.jsonl` used for the paper's P/R/F1 numbers.
