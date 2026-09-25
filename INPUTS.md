# External inputs and frozen revisions

| Input | Pinned source | Integrity record | Used for |
| --- | --- | --- | --- |
| ProofWriter V2020.12.3 | `https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip` | Archive SHA-256 `bbc5694901e8306d0bd659aa1ad53ccfd02c201864f4b320ffa3777827d1fc26`; OWA depth-5 `meta-test.jsonl` SHA-256 `c09fad796aaf546d6fcbfc77ecf91f935ffed3c936c2b0e96f4aa57211fad842` | Exact source-order and stress cohorts |
| $\tau^2$-Bench | `https://github.com/sierra-research/tau2-bench/tree/2174a603f6d014ef94473ffa95957f6ce27100db` | Git commit `2174a603f6d014ef94473ffa95957f6ce27100db`; banking catalog digest in `results/banking_exact_separable/results.json` | Banking packets and local telecom stream |
| Qwen3.5-2B | `https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc` | Model revision `15852e8c16360a2fea060d615a32b45270f8a8fc`; no model weights redistributed | Telecom agent run |

ProofWriter policies receive the benchmark's original formal source, not gold labels or proof DAGs. The banking source catalog is a reviewed mapping from documents and is used as one development instance. The telecom public artifact is aggregate only; it does not reproduce model inference.
