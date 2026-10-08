# Proof-Economic Delegation: public reproducibility artifacts

This release accompanies the manuscript **Proof-Economic Delegation: Near-Optimal Acquisition of Reusable Formal Authority for Recurrent AI Agent Workloads** (target: *Big Data and Cognitive Computing*). It reproduces the paper's tables, diagrams, graphs, and in-text quantities from frozen results. The release includes exact acquisition oracles, a second weighted-completion solver, exhaustive density-greedy and commitment controls, checked tool execution, and reproducible runtime fault cases. The empirical near-optimality result concerns the formal-source acquisition objective on the evaluated ProofWriter cohorts.

## Recreate manuscript inputs

Python 3.11 or newer is sufficient for artifact regeneration:

```bash
python scripts/generate_manuscript_inputs.py --output /tmp/proof-economic-generated
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/audit_release.py
```

`generated/` is the checked-in expected output. The unit test regenerates every LaTeX input and compares its bytes. No model weights, raw customer transcripts, or GPU is needed for this step. `results/tau2_telecom_stream_heldout/public_summary.json` contains aggregate results; `public_trace.json` contains only dialogue-free episode metrics. Both identify the local raw result files by SHA-256. The trace reproduces the cumulative-call and lifecycle plots but does not permit replay of individual model conversations or independent semantic audit of them. The original model-call counts can be audited from the shipped episode metrics; conversation-level semantic review requires the original inputs and traces.

## Re-run the exact ProofWriter analysis

The official ProofWriter V2020.12.3 OWA archive is fetched and checked by `delegation_lab.adapters.proofwriter.download_proofwriter`. See [INPUTS.md](INPUTS.md) for URL and hashes. The exact source-order, structural-stress, and workload-skew runners are in `src/delegation_lab/experiments/`; their frozen configurations and protocols are in `configs/` and `docs/`. They refuse to overwrite published results. Download the data in a separate copy of this repository with:

```bash
PYTHONPATH=src python -c "from pathlib import Path; from delegation_lab.adapters.proofwriter import download_proofwriter; download_proofwriter(Path('data/proofwriter'))"
```

Use a separate copy for each optional CPU-intensive rerun, retaining the published artifacts that the other studies use as inputs. In the source-order copy, move only `results/proofwriter_exact_characterization/results.json` aside, then run `PYTHONPATH=src python -m delegation_lab.experiments.proofwriter_exact_characterization`. In the stress copy, move only its `results.json` and `selection_manifest.json` aside, then run `PYTHONPATH=src python -m delegation_lab.experiments.proofwriter_structural_stress`. In the skew copy, move only its `results.json` aside, then run `PYTHONPATH=src python -m delegation_lab.experiments.proofwriter_workload_skew_sensitivity`. Keep the prior training artifact and `results/proofwriter/results.json` overlap-guard record in every copy. The freeze manifests verify source/config/protocol checksums before computation.

The banking artifact is one development catalog. Its packet costs and values are shipped in `results/banking_exact_separable/results.json`; the public test independently recomputes the optimal ratio order and fixed-budget knapsack frontier from those arrays. The document-to-rule mapping was not prospectively validated. The original pinned $\tau^2$ checkout is identified in [INPUTS.md](INPUTS.md).

## Reproduce the scheduling and runtime review

The secondary acquisition audit reconstructs every source catalog in both frozen cohorts, under uniform and training-frequency workloads. It independently solves weighted first-certification cost and compares LPBA with an exhaustive density-greedy control and a matched bundle-commitment variant. It reproduces 400 theory/workload optima. This is a second analysis of the existing cohorts, rather than a new independent dataset.

In a separate copy with the pinned ProofWriter data downloaded, move only `results/acquisition_review/results.json` aside and run:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python scripts/run_acquisition_review.py --workers 8
python scripts/summarize_acquisition_review.py --output /tmp/paired-summary.json
```

`--data-root` accepts a separate benchmark checkout. NumPy and PyYAML are required for source reconstruction; optional `numba` accelerates the independent backward dynamic program. It runs correctly without that accelerator, with a longer runtime. Bootstrap summaries use NumPy's random generator and the seed recorded in `configs/acquisition_review.json`.

The standalone tests execute all 15 constructed runtime cases without a language model or benchmark download. To write a new copy of the case artifact, move `results/runtime_review/results.json` aside in a separate release copy and run `python scripts/run_runtime_review.py`. The cases cover changed formal source meaning, versions, program or argument modification, principal and state guards, failed postconditions, policy and state races, and selective invalidation. They check implementation mechanisms; they do not estimate a production failure probability.

The review reproduced and fixed an initial-snapshot race in the earlier public runtime. An update between checking a proof and capturing its dependency epoch could allow a write. The current kernel captures epochs before checking. `results/runtime_review/regression.json` records the before/after outcome and the historical release commit. Historical telecom metrics predate the repair and remain frozen.

## Layout

- `src/`: model-independent authority, proof, acquisition, and exact ProofWriter analysis code used by the reported studies.
- `configs/`, `docs/`: frozen study designs and methods.
- `results/`: source-order, stress, skew, banking, secondary acquisition review, runtime faults, and aggregate and episode-metric telecom artifacts. The large train-prior artifact is retained because its exact bytes are checked by the frozen runs.
- `generated/`: publication LaTeX quantities, complete tables, and native TikZ/pgfplots figures.
- `scripts/`: generation, verified DOI bibliography construction, checksum and release audit.
- `tests/`: artifact and exact-objective consistency checks.

The publisher's MDPI class is deliberately absent. MDPI's template is used only in the private submission package, under its stated template-use terms. The release code is Apache-2.0 licensed; the upstream benchmark and model have their own terms. Author funding, conflicts, and CRediT fields are submission metadata, not inferred by this repository.
