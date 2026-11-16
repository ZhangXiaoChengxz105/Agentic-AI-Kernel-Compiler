# Lumen models (for the agent), checked 2026-11-15

**Decision (2026-11-15, Xiaocheng): agent model = `glm-5.3`** (zai-org/GLM-5.3, flagship for long-horizon coding/agents,
524k context, reasoning). Needs a one-time acknowledgment on the Lumen web UI before use.
Cost implication: cached input has no discount for GLM (`cached_input_cost_per_million` = null → billed at 1.321),
so the repair loop should send compact context (current code + latest diagnostics), not the full chat history.

Source: `GET https://lumen.ncsa.illinois.edu/v1/models?include_unacknowledged=true` (free: verified from
Lumen source `lumen/blueprints/api/routes.py`, which has no billing in `list_models`; `/v1/usage` showed 10.00 coins and 0 requests before and after).
Lumen API = OpenAI-compatible, base URL `https://lumen.ncsa.illinois.edu/v1`, header `Authorization: Bearer $LUMEN_API_KEY`.
Only `POST /v1/chat/completions` (and audio/embeddings) costs coins.

Budget: 10 coins max, refill +0.10/hr (= 2.4/day). Empty → full takes ~4 days.

## Table (prices = coins per 1M tokens)
| Lumen id | Upstream model | Context | In | Out | Reasoning | Cutoff | Flags |
|---|---|---|---|---|---|---|---|
| qwen3-coder-next | Qwen/Qwen3-Coder-Next-FP8 (80B total / 3B active) | 131k | 0.227 | 1.236 | no | 2025-09 | needs_ack |
| qwen3.8-flash | RadixArk/Qwen3.8-Flash-Next-NVFP4 (~6B active) | 262k | 0.143 | 0.453 | yes | — | needs_ack, early_access |
| deepseek-v4.1-flash-1017 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | 1M | 0.257 | 0.931 | yes | 2025-05 | needs_ack, early_access, **may be removed (DHS directive)** |
| deepseek-v4-flash | deepseek-ai/DeepSeek-V4-Flash-0731 | 1M | 0.162 | 0.371 | yes | 2025-05 | needs_ack, same removal risk |
| glm-5.3 | zai-org/GLM-5.3 | 524k | 1.321 | 4.248 | yes | — | needs_ack |
| nemotron-3-super-120b-a12b | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 | 262k | 0.169 | 0.759 | yes | 2025-06 | — |
| ornith-1.0-35b | deepreinforce-ai/Ornith-1.0-35B-FP8 (MoE, agentic coding, MIT) | 262k | 0.05 | 0.10 | yes | — | — |
| beam-1.0 | reflection/Beam-1.0-501B-A23B-FP8 | 524k | 0.33 | 1.10 | yes | 2026-06 | early_access |
| qwen3.8-27b | RadixArk/Qwen3.8-27B-NVFP4 (vision-language) | 262k | 0.365 | 2.397 | yes | — | needs_ack |
| qwen3.6-35b-a3b | Qwen/Qwen3.6-35B-A3B-FP8 (multimodal) | 262k | 0.185 | 1.138 | yes | 2025-04 | needs_ack |
| gemma-4-31b-it | google/gemma-4-31B-it | 137k | 0.168 | 0.436 | yes | 2025-01 | — |
| muse-glimmer-30b | meta-models/Muse-Glimmer-30B | 131k | 0.323 | 1.345 | yes | 2026-01 | needs_ack, testing only |
| granite-speech-4.1-2b-plus | speech-to-text only | — | 0 | 0 | — | — | not usable for code |

`needs_ack` = must acknowledge a notice once (web UI or `POST /v1/models/<id>/acknowledge`) before use.
Notices: Qwen/DeepSeek/GLM carry the UIUC "Country of Concern" notice. DeepSeek: DHS directive (2026-10-19) ordered federal
agencies to remove DeepSeek; Lumen "could end up" removing it "with little to no advanced notice".

## Cost estimate per 10-iteration repair run (assumption, not measured)
Input = Σ(6k + 4k·i) = 240k tokens (history resent each turn); output = 4k/turn (non-reasoning) or 10k/turn (reasoning).
| Model | coins/run | runs per 10 coins |
|---|---|---|
| ornith-1.0-35b | 0.022 | ~455 |
| deepseek-v4-flash | 0.076 | ~132 |
| qwen3.8-flash | 0.080 | ~126 |
| gemma-4-31b-it | 0.084 | ~119 |
| qwen3-coder-next | 0.104 | ~96 |
| nemotron-3-super | 0.116 | ~86 |
| deepseek-v4.1-flash | 0.155 | ~65 |
| beam-1.0 | 0.189 | ~53 |
| qwen3.8-27b | 0.327 | ~31 |
| glm-5.3 | 0.742 | ~13 |

## Key facts for model choice
- **No model's knowledge cutoff is after TIRx's release (2026-09-29)** → none has seen TIRx; TIRx docs/examples must go in the prompt.
  CUDA is well covered by all of them.
- Public coding benchmarks are mostly vendor-reported; GPU-kernel-specific evidence is thin. arXiv 2609.21058: on KernelBench L1
  the best open-weights model was 30.4% correct vs 91.1% for a frontier model (models not named in abstract).
- Pick by a small measured pilot (same task, same prompt, few calls), not by benchmarks.
