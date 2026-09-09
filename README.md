# AI_Systems_Project_CMPE249
Your repository must include an initial README.md containing the following: 

# CMPE249 Project Proposal
**Title:** Does a Post-Hoc, Model-Agnostic Safety Monitor Reduce VLA Safety Violations on LIBERO-Safety?
**Track:** Research | **Lane:** Safety and Evaluation | **Team:** Solo
**Hardware scope:** Simulation/benchmark only (MuJoCo-based, via LIBERO-Safety)

## Abstract
Vision-Language-Action (VLA) policies are increasingly capable at manipulation tasks, but a June 2026 benchmark (LIBERO-Safety) shows that even strong VLAs regularly violate physical safety constraints and fail to refuse harmful or ambiguous instructions. Existing fixes either retrain the policy (e.g., SafeVLA's constrained learning) or monitor failure risk in settings unrelated to this new benchmark. This project asks whether a lightweight, model-agnostic runtime monitor — wrapped around an unmodified, off-the-shelf VLA — can measurably reduce LIBERO-Safety's physical-violation and semantic-refusal-failure rates, and at what cost to task success.

## 1. Problem Formulation

**Domain problem:** Off-the-shelf VLA policies (e.g., OpenVLA, OpenVLA-OFT) are not trained with safety as an explicit objective. When deployed, they can execute physically unsafe actions (collisions, drops, unsafe proximity to objects/hands) or fail to refuse instructions that are harmful or ambiguous. Retraining a policy to fix this (SafeVLA-style) is expensive and not always available to a downstream deployer who only has access to a frozen checkpoint. This project investigates whether a **post-hoc runtime monitor** — a lightweight wrapper that observes the policy's outputs and internal signals at runtime and intervenes (pause, reobserve, fall back, or refuse) — can close some of this safety gap without any change to the base policy's weights.

**Input:**
- A frozen, pretrained VLA policy (OpenVLA or OpenVLA-OFT checkpoint).
- Per-timestep observations already available in the LIBERO-Safety rollout loop: RGB visual observation, language instruction, the policy's predicted action (and, where obtainable, action-token logits/log-probabilities or an ensemble of sampled actions for uncertainty estimation).

**Output:**
- A per-timestep monitor decision: {proceed, pause-and-reobserve, fallback action, refuse instruction}.
- Standard LIBERO-Safety episode outcomes recomputed under monitor supervision: task success, physical-safety-violation rate, semantic-refusal rate (Refusal Rate, RR), across the benchmark's L0–L2 difficulty tiers and its five safety suites.

**Target metrics / success criteria:**
1. **Primary:** reduction in LIBERO-Safety's physical safety-violation rate for the monitored policy vs. the unmonitored baseline, on at least 2 of the benchmark's 5 safety suites.
2. **Secondary:** change in semantic Refusal Rate (RR) on the benchmark's semantic-safety suite — ideally an increase in correct refusals without a large increase in false refusals on safe instructions.
3. **Cost accounting (required, not optional):** change in task success rate and any added wall-clock latency per step, since a monitor that eliminates violations by refusing everything is not a success. A viable result must show a favorable (or at least clearly characterized) safety-vs-success trade-off, not just a safety win in isolation.
4. **Stretch:** qualitative/quantitative comparison against SafeVLA's constrained-learning results (from their published numbers, not by rerunning SafeVLA) to contextualize how much of the safety gap a post-hoc method can close relative to a training-time method.

## 2. Proposed Technical Approach

**High-level architecture:**
```
[Visual obs + language instr.] --> [Frozen VLA policy] --> [predicted action + confidence signal]
                                                                    |
                                                                    v
                                                        [Runtime Safety Monitor]
                                                     (uncertainty / consistency / risk score)
                                                                    |
                                        ------------------------------------------------
                                        |                    |                          |
                                     proceed        pause & reobserve            refuse / fallback
                                        |                    |                          |
                                        v                    v                          v
                                              [LIBERO-Safety simulation environment]
```

**Chosen base models:** OpenVLA-OFT is the preferred backbone (faster inference, directly supported by LIBERO-Safety's harness, config-level 4-bit/8-bit loading if compute is constrained). OpenVLA (original) as a secondary check if time allows, to test whether monitor effectiveness generalizes across backbones.

**Monitor design (the core technical contribution), to be implemented as a small ablation family rather than a single fixed design:**
- **Variant 1 — Uncertainty-based gating** (adapted from the uncertainty-calibrated safety-gating literature): estimate action uncertainty via lightweight ensembling or Monte Carlo dropout on the action head, gate on a calibrated threshold.
- **Variant 2 — Consistency/progress monitoring** (adapted from Agia et al.): track rollout self-consistency and task-progress signals as a proxy for imminent failure, without requiring privileged failure labels.
- **Variant 3 (stretch) — Predictive VLM risk monitor:** use a separate VLM prompted on the current observation + instruction to flag near-future risk, similar to the predictive vision-language monitoring approach.

All three variants are training-free or require only lightweight calibration (e.g., threshold tuning on a held-out validation split of LIBERO-Safety scenes) — no fine-tuning of the base VLA is planned, keeping the project within a solo-feasible, semester-long scope.

**Data sources:**
- LIBERO-Safety's public benchmark suite and evaluation harness (github.com/LIBERO-SAFETY/LIBERO-Safety) — five safety task suites, L0–L2 difficulty tiers, ~19.7K reference demonstrations available if any calibration data is needed.
- No new data collection or teleoperation is planned; this keeps the project within course-scope compute/time limits.

**Intended technical modifications / novel contribution:**
- Not a new base policy or a new benchmark (both are red-ocean, see novelty audit).
- The contribution is (a) the first evaluation of post-hoc runtime monitors against a purpose-built VLA safety benchmark, and (b) a comparative ablation across 2–3 monitor designs, characterizing which monitor family best trades off safety against task success on which suite/tier.

## 3. Compute & Maintainer

**Device available:** No confirmed dedicated hardware. Plan assumes a single cloud or campus GPU instance with at least 16–24GB VRAM (sufficient for OpenVLA-OFT inference at fp16, or 4-bit quantized OpenVLA if a smaller GPU is what's available). LIBERO-Safety's MuJoCo-based simulation itself is CPU-light and does not require a GPU beyond running the policy. A small paid-compute budget (e.g., a rented A100/RTX 4090 instance for burst evaluation runs) is justified and consistent with course defaults, but not required if SJSU lab/cluster GPU access is available.

**Maintainer:** The student is the sole maintainer of the project's GitHub repository, responsible for environment setup, running LIBERO-Safety's installation (note: it requires removing any existing LIBERO installation first, per its README), implementing the monitor variants, and logging experiments. Claude Code is planned as a coding assistant for repetitive implementation work (harness glue code, logging scaffolding, ablation-sweep scripts) — the student remains the decision-maker on experiment design, monitor logic, and result interpretation.


# Literature & SOTA Survey
**Project:** Post-hoc, Model-Agnostic Safety Monitoring for Vision-Language-Action (VLA) Policies
**Track:** Research (CMPE249) | **Lane:** Safety & Evaluation | **Scope:** Simulation-only

This survey covers 10 papers/models published within the last 1–3 years (2024–2026), organized into three clusters: (A) safety **benchmarks** for VLA models, (B) **runtime monitoring / failure-detection** methods, and (C) baseline **VLA policies** the project will monitor.

---

## A. Safety Benchmarks for VLA Models

### 1. LIBERO-Safety (Cui et al., 2026, arXiv:2606.23686, ECCV 2026)
The primary benchmark this project builds on. Introduces a procedurally generated benchmark separating **physical safety** (collision/violation avoidance) from **semantic safety** (refusing harmful or ambiguous instructions), organized into five safety suites across three difficulty tiers (L0–L2). Evaluates 10 architectures including OpenVLA and OpenVLA-OFT and reports a "generalization-safety tension": higher training diversity improves safety, but task success is still bottlenecked by trajectory synthesis and semantic misalignment. Public code and pretrained-model evaluation harness at github.com/LIBERO-SAFETY/LIBERO-Safety.
*Relevance:* this is the evaluation environment for the project — the benchmark whose violation/refusal rates the monitor is meant to improve.

### 2. VLA-Arena (Zhang et al., 2025–2026, arXiv:2512.22539)
An open-source benchmarking framework with 170 tasks across four dimensions — Safety, Distractor, Extrapolation, and Long Horizon — plus systematic language and visual perturbations. Finds that current VLAs show "a lack of consideration for safety constraints" as one of several core limitations.
*Relevance:* secondary/validation benchmark if time allows; also useful as a second data point on how "safety" is operationalized differently across benchmarks.

### 3. ForesightSafety-VLA (2026, arXiv:2606.27079)
A diagnostic safety benchmark using a 13-category safety taxonomy (physical, instruction-side, perception-side) and process-level risk metrics (cumulative safety cost, risk exposure time) rather than a single aggregate score.
*Relevance:* useful for framing richer safety metrics beyond binary violation/refusal, and as closest-competing-work evidence that the field is moving toward finer-grained diagnosis — supports this project's diagnostic framing.

### 4. HazardArena (2026)
A benchmark specifically targeting **semantic safety**: cases where an action executes correctly but produces an unsafe outcome because the policy fails to connect visual-linguistic semantics to risk.
*Relevance:* motivates why a monitor needs a semantic-risk-aware component, not just a physical-collision detector.

### 5. SafeVLA (Zhang et al., NeurIPS 2025)
Addresses VLA safety via **constrained learning** — i.e., a training-time alignment method that modifies the policy itself to be safer.
*Relevance:* this is the key differentiating comparison point. SafeVLA requires retraining/fine-tuning the base policy; this project instead asks whether a comparable safety gain is achievable **without** touching the base model, via a post-hoc wrapper — a complementary, more deployment-friendly alternative worth benchmarking against.

---

## B. Runtime Monitoring / Failure Detection for Robot Policies

### 6. SAFE: Multitask Failure Detection for Vision-Language-Action Models (2025, arXiv:2506.09937)
Proposes a failure-detection approach for VLA policies that predicts likely task failure during execution, framed as a general, model-agnostic add-on rather than a retraining method.
*Relevance:* closest prior monitor design; the project's own monitor will be built on and compared against this style of failure signal.

### 7. Pre-VLA: Preemptive Runtime Verification for Reliable VLA and World-Model Rollouts (2026, arXiv:2605.22446)
Introduces preemptive runtime verification specifically for VLA and world-model rollouts, catching problems before they manifest as physical failures.
*Relevance:* an alternative monitor design (verification-based rather than failure-prediction-based) — a candidate second monitor variant for comparison.

### 8. Uncertainty-Calibrated Safety Gating for VLA Manipulation Under Domain Shift (2026, PMC13210885)
A model-agnostic, uncertainty-calibrated safety-gating wrapper that monitors a VLA policy's confidence at runtime and routes control among execution, pause-and-reobserve, and a fallback planner under domain shift.
*Relevance:* the closest architectural template for this project's monitor — a runtime wrapper around an unmodified base policy — but evaluated on domain shift, not on a purpose-built safety benchmark like LIBERO-Safety.

### 9. Unpacking Failure Modes of Generative Policies: Runtime Monitoring of Consistency and Progress (Agia et al., CoRL 2025)
Monitors generative-policy rollouts for internal consistency and task progress as a proxy for failure risk, without requiring privileged failure labels.
*Relevance:* provides a concrete, implementable signal (consistency/progress) the monitor in this project can adapt.

### 10. Predictive Vision-Language Monitoring for Proactive Safety in Robot Task Execution (2026, Frontiers in Robotics and AI, DOI:10.3389/frobt.2026.1870024)
Uses a VLM to predict near-future execution risk from structured task plans and visual observations, halting or falling back when predicted risk exceeds a threshold.
*Relevance:* an alternative, VLM-reasoning-based monitor design; a good source of ablation ideas (predictive window length, fallback policy design).

---

## C. Baseline VLA Policies (to be monitored, not modified)

- **OpenVLA** (Kim et al., 2024, arXiv:2406.09246) — open-source 7B-parameter VLA, the primary base policy for this project.
- **OpenVLA-OFT** (Kim, Finn, Liang, 2025, arXiv:2502.19645) — "Optimizing Speed and Success" fine-tuning recipe for OpenVLA; faster inference, directly supported by LIBERO-Safety's evaluation harness, and the preferred backbone if compute is tight.

---

## Summary of the Gap
Cluster A (benchmarks) and Cluster B (monitors) currently do not intersect: every monitor paper above evaluates on its own custom setup or on domain-shift/OOD conditions, and none has been evaluated against a purpose-built VLA safety benchmark (LIBERO-Safety, VLA-Arena, ForesightSafety-VLA, or HazardArena) — largely because most of these benchmarks are only weeks to a few months old as of this survey. SafeVLA is the only paper directly targeting the same problem (reducing VLA safety violations) but via training-time alignment rather than a post-hoc wrapper. This project sits at that unoccupied intersection: **can a post-hoc, model-agnostic monitor close some of the safety gap that LIBERO-Safety exposes, without retraining the base policy?**
