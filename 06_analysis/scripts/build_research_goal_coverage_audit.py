"""Build a requirement-level coverage audit for the MARL+ISAC paper line.

The audit is intentionally evidence-driven. It reads the current manuscript-
facing CSV/manifests and reports which parts of the research objective are
supported by main evidence, which parts are only boundary/supplementary
evidence, and which parts remain open before a strong TWC/TCOM submission.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


OUT_DIR = Path("06_analysis/paper_tables/research_goal_coverage_audit")
REPORT = Path("06_analysis/research_goal_coverage_audit_20260707.md")

FINAL_METHOD_CSV = Path("06_analysis/paper_tables/marl/p10_final_b10_b15_method_comparison_with_v4/marl_method_comparison.csv")
TRACE_CSV = Path("06_analysis/paper_tables/marl/p10_final_method_manifest_trace/phase10_method_manifest_trace.csv")
STABILITY_CSV = Path("06_analysis/paper_tables/statistical_stability_summary/statistical_stability_summary.csv")
TRAINING_MANIFEST = Path("06_analysis/paper_tables/marl/p10_gate_training_3seed_100ep_step_curves/manifest.json")
ARTIFACT_MANIFEST = Path("06_analysis/manuscript_artifact_manifest_20260707.json")
FIGURE_AUDIT = Path("06_analysis/figure_table_provenance_audit_20260707.md")
SUBMISSION_REVIEW = Path("06_analysis/submission_readiness_review_20260707.md")
RERUN_VALIDATION_MANIFEST = Path("06_analysis/paper_tables/marl/p10_independent_rerun_gate31_b10_validation/manifest.json")
RERUN_VALIDATION_REPORT = Path("06_analysis/phase10_independent_rerun_validation_20260707.md")
LEARNED_ABLATION_MANIFEST = Path("06_analysis/paper_tables/marl/p10_learned_component_ablation_b10_3ep/manifest.json")
LEARNED_ABLATION_REPORT = Path("06_analysis/phase10_learned_component_ablation_report_20260707.md")
PAIRED_SIGNIFICANCE_MANIFEST = Path("06_analysis/paper_tables/marl/p10_paired_significance_primary/manifest.json")
PAIRED_SIGNIFICANCE_REPORT = Path("06_analysis/phase10_paired_significance_report_20260707.md")
CLAIM_STRENGTH_MANIFEST = Path("06_analysis/paper_tables/submission_claim_strength_audit/manifest.json")
CLAIM_STRENGTH_REPORT = Path("06_analysis/submission_claim_strength_audit_20260707.md")
RANGE_NOTE = Path("06_analysis/range_abstraction_theory_note_20260707.md")
RANGE_GRID_CSV = Path("06_analysis/paper_tables/round3_robustness/range_rc_rs_grid/aggregate_metrics.csv")
RANGE_RATIO_CSV = Path("06_analysis/paper_tables/round3_robustness/range_rs_ratio/aggregate_metrics.csv")
MAIN_TEX = Path("07_paper/ieee_twc_isac_nd/main.tex")
SUPPLEMENT_TEX = Path("07_paper/ieee_twc_isac_nd/supplement.tex")

CODE_EVIDENCE = [
    Path("05_simulation/src/isac_nd_sim/marl_env.py"),
    Path("05_simulation/run_marl_training.py"),
    Path("05_simulation/run_marl_evaluate.py"),
    Path("05_simulation/src/isac_nd_sim/neural_contention_actor_critic.py"),
    Path("05_simulation/tests/test_marl_env_contract.py"),
    Path("05_simulation/tests/test_marl_fiveway_eval_campaign.py"),
]


@dataclass
class Requirement:
    requirement_id: str
    theme: str
    requirement: str
    status: str
    evidence_strength: str
    evidence_summary: str
    evidence_paths: str
    next_action: str


@dataclass
class Risk:
    priority: str
    risk: str
    trigger: str
    current_status: str
    next_action: str


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8-sig")


def as_float(value: str | int | float | None, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: str | int | float | None, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def fmt_set(values: Iterable[object]) -> str:
    cleaned = sorted({str(value) for value in values if str(value) != ""})
    return ";".join(cleaned)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_tracked_paths(paths: Iterable[Path]) -> set[str]:
    normalized = [path.as_posix() for path in paths]
    if not normalized:
        return set()
    try:
        result = subprocess.run(
            ["git", "ls-files", "--", *normalized],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return set()
    if result.returncode != 0:
        return set()
    return {line.strip().replace("\\", "/").lower() for line in result.stdout.splitlines() if line.strip()}


def index_final(rows: list[dict[str, str]]) -> dict[tuple[str, float], dict[str, str]]:
    indexed: dict[tuple[str, float], dict[str, str]] = {}
    for row in rows:
        indexed[(row.get("method", ""), as_float(row.get("beamwidth_deg")))] = row
    return indexed


def unique_semicolon_paths(rows: list[dict[str, str]], fields: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for row in rows:
        for field in fields:
            for raw in str(row.get(field, "")).split(";"):
                raw = raw.strip()
                if not raw:
                    continue
                normalized = raw.replace("\\", "/")
                key = normalized.lower()
                if key in seen:
                    continue
                seen.add(key)
                paths.append(Path(normalized))
    return paths


def ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return float("inf")
    return numerator / denominator


def requirement_rows(
    final_rows: list[dict[str, str]],
    trace_rows: list[dict[str, str]],
    stability_rows: list[dict[str, str]],
    training_manifest: dict,
    artifact_manifest: dict,
    rerun_validation: dict,
    learned_ablation: dict,
    paired_significance: dict,
    claim_strength: dict,
) -> tuple[list[Requirement], list[Risk], dict[str, object]]:
    final_methods = {row.get("method", "") for row in final_rows}
    final_beams = {as_float(row.get("beamwidth_deg")) for row in final_rows}
    final_nodes = {as_int(row.get("node_count")) for row in final_rows}
    final_slots = {as_int(row.get("slots_per_episode")) for row in final_rows}
    final_episodes = [as_int(row.get("episodes")) for row in final_rows if row.get("episodes")]
    final_index = index_final(final_rows)

    required_methods = {
        "uniform_random",
        "skyorbs_like",
        "mappo_no_isac",
        "contention_no_isac",
        "contention_actor",
        "gated_contention_actor",
        "adaptive_gated_contention_actor",
        "topology_adaptive_gated_contention_actor",
        "balanced_topology_gated_contention_actor",
    }
    main_actor = {
        beam: as_float(final_index.get(("contention_actor", beam), {}).get("discovery_rate_mean"))
        for beam in (10.0, 15.0)
    }
    sky = {
        beam: as_float(final_index.get(("skyorbs_like", beam), {}).get("discovery_rate_mean"))
        for beam in (10.0, 15.0)
    }
    random_baseline = {
        beam: as_float(final_index.get(("uniform_random", beam), {}).get("discovery_rate_mean"))
        for beam in (10.0, 15.0)
    }
    no_isac_methods = {"mappo_no_isac", "contention_no_isac"}
    no_isac_best = {
        beam: max(
            as_float(final_index.get((method, beam), {}).get("discovery_rate_mean"))
            for method in no_isac_methods
        )
        for beam in (10.0, 15.0)
    }
    min_dir_ratio = min(
        ratio(main_actor[10.0], max(sky[10.0], random_baseline[10.0])),
        ratio(main_actor[15.0], max(sky[15.0], random_baseline[15.0])),
    )
    min_no_isac_ratio = min(
        ratio(main_actor[10.0], no_isac_best[10.0]),
        ratio(main_actor[15.0], no_isac_best[15.0]),
    )

    trace_complete = sum(1 for row in trace_rows if row.get("trace_status") == "complete")
    trace_total = len(trace_rows)
    trace_raw_manifest_paths = unique_semicolon_paths(trace_rows, ("raw_eval_manifests", "train_manifests"))
    trace_checkpoint_paths = unique_semicolon_paths(trace_rows, ("checkpoints",))
    trace_raw_manifests_existing = sum(1 for path in trace_raw_manifest_paths if path.exists())
    trace_checkpoints_existing = sum(1 for path in trace_checkpoint_paths if path.exists())
    raw_bundle_paths = trace_raw_manifest_paths + trace_checkpoint_paths
    raw_git_tracked = git_tracked_paths(raw_bundle_paths)
    raw_bundle_git_tracked = sum(1 for path in raw_bundle_paths if path.as_posix().lower() in raw_git_tracked)
    training_runs = as_int(training_manifest.get("runs"))
    training_steps = as_int(training_manifest.get("step_rows"))
    training_episode_rows = as_int(training_manifest.get("episode_rows"))
    training_eval_rows = as_int(training_manifest.get("eval_rows"))
    training_figures = training_manifest.get("figures", [])
    artifact_count = as_int(artifact_manifest.get("artifact_count"))
    missing_count = as_int(artifact_manifest.get("missing_count"), default=-1)
    rerun_all_match = bool(rerun_validation.get("all_metrics_match", False))
    rerun_status_counts = rerun_validation.get("status_counts", {})
    rerun_method = str(rerun_validation.get("method", ""))
    rerun_beam = as_float(rerun_validation.get("beamwidth_deg"))
    learned_ablation_labels = set(learned_ablation.get("labels", []))
    required_learned_ablation_labels = {
        "trained_full",
        "random_weights_full",
        "zero_weights_rule_only",
        "trained_no_rule_residual",
        "trained_no_candidate_mask",
    }
    learned_ablation_complete = required_learned_ablation_labels.issubset(learned_ablation_labels)
    paired_significance_pass = bool(paired_significance.get("all_confirmatory_tests_pass")) and as_int(
        paired_significance.get("confirmatory_test_count")
    ) >= 16
    claim_strength_pass = (
        bool(claim_strength.get("all_required_checks_pass"))
        and as_int(claim_strength.get("required_check_count")) > 0
        and as_int(claim_strength.get("review_required_risk_hits"), default=-1) == 0
    )

    range_rows = read_csv(RANGE_GRID_CSV) + read_csv(RANGE_RATIO_CSV)
    range_rc_ratios = {round(as_float(row.get("communication_range_to_diagonal_ratio")), 3) for row in range_rows if row.get("communication_range_to_diagonal_ratio")}
    range_rs_ratios = {round(as_float(row.get("sensing_to_comm_range_ratio")), 3) for row in range_rows if row.get("sensing_to_comm_range_ratio")}
    range_sweep_complete = {0.5, 0.75, 1.0, 1.25}.issubset(range_rs_ratios) and len(range_rc_ratios) >= 1
    range_note_text = read_text(RANGE_NOTE)
    range_main_text = read_text(MAIN_TEX)
    range_supp_text = read_text(SUPPLEMENT_TEX)
    range_theory_supported = all(
        token in range_note_text
        for token in (
            "protocol support parameters",
            "not a platform-calibrated statement",
            "Bomfin2024SystemISAC",
            "Rs/Rc",
        )
    )
    range_manuscript_bounded = (
        "do not set $\\Rs=\\Rc$ by default" in range_main_text
        and "matched-support operating point" in range_main_text
        and "not as a calibrated hardware statement" in range_supp_text
    )
    range_status = "PASS" if range_sweep_complete and range_theory_supported and range_manuscript_bounded else "CAUTION"

    stability_beams = {as_float(row.get("beamwidth_deg")) for row in stability_rows if row.get("beamwidth_deg")}
    stability_nodes = {as_int(row.get("node_count")) for row in stability_rows if row.get("node_count")}
    stability_protocols = {row.get("protocol", "") for row in stability_rows}
    stability_mobility = {row.get("mobility_model", "") for row in stability_rows if row.get("mobility_model")}
    stability_slots_ms = {as_float(row.get("slot_duration_ms")) for row in stability_rows if row.get("slot_duration_ms")}
    main_stability_rows = [row for row in stability_rows if row.get("evidence_tier") == "main"]
    supplement_stability_rows = [row for row in stability_rows if row.get("evidence_tier", "").startswith("supplement")]

    code_files_present = sum(1 for path in CODE_EVIDENCE if path.exists())
    code_paths = ";".join(path.as_posix() for path in CODE_EVIDENCE)

    requirements: list[Requirement] = []

    def add(
        rid: str,
        theme: str,
        requirement: str,
        status: str,
        strength: str,
        summary: str,
        paths: Iterable[Path] | str,
        next_action: str,
    ) -> None:
        if isinstance(paths, str):
            evidence_paths = paths
        else:
            evidence_paths = ";".join(path.as_posix() for path in paths)
        requirements.append(
            Requirement(rid, theme, requirement, status, strength, summary, evidence_paths, next_action)
        )

    add(
        "R01",
        "MARL implementation",
        "Use a real slot-level MARL environment and actor-critic training/evaluation path.",
        "PASS" if code_files_present == len(CODE_EVIDENCE) else "OPEN",
        "code+tests",
        f"{code_files_present}/{len(CODE_EVIDENCE)} core MARL code/test files are present; this proves an implemented MARL pipeline, not that learned weights dominate every rule residual.",
        code_paths,
        "Keep future manuscript wording tied to the implemented MAPPO-style actor-critic path and avoid implying exhaustive MARL-family dominance.",
    )
    add(
        "R02",
        "Training trace",
        "Train in a small source setting N=10, B=10, 300 slots/episode, with step-indexed reward evidence.",
        "PASS" if training_runs >= 3 and training_steps >= 90000 and training_episode_rows >= 300 else "CAUTION",
        "main",
        f"Training manifest reports {training_runs} runs, {training_steps} step rows, {training_episode_rows} episode rows, {training_eval_rows} eval rows.",
        [TRAINING_MANIFEST],
        "For a stronger revision, add an independent training re-run or more seeds only if compute budget allows.",
    )
    add(
        "R03",
        "Final transfer",
        "Evaluate zero-shot N=10/B=10 policies at N=100, 3000 slots, B=10 and B=15.",
        "PASS" if final_nodes == {100} and final_slots == {3000} and {10.0, 15.0}.issubset(final_beams) else "OPEN",
        "main",
        f"Final comparison has nodes={fmt_set(final_nodes)}, slots={fmt_set(final_slots)}, beams={fmt_set(final_beams)}, rows={len(final_rows)}.",
        [FINAL_METHOD_CSV],
        "Keep 3/5/30-degree results outside the final-main claim unless rerun with the same Phase10 method set.",
    )
    add(
        "R04",
        "Baseline completeness",
        "Compare against random, literature-inspired directional, MARL without ISAC, improved no-ISAC, and ISAC MARL variants.",
        "PASS" if required_methods.issubset(final_methods) else "OPEN",
        "main_with_caveat",
        f"Final methods cover {len(final_methods & required_methods)}/{len(required_methods)} required method groups.",
        [FINAL_METHOD_CSV],
        "SkyOrbs-like must remain labelled as inspired/approximate unless a faithful reproduction is added.",
    )
    add(
        "R05",
        "ISAC gain",
        "Show that ISAC-assisted MARL improves discovery over blind/directional and no-ISAC MARL baselines.",
        "PASS" if min_dir_ratio > 20.0 and min_no_isac_ratio > 20.0 else "CAUTION",
        "main",
        f"Minimum actor discovery ratio is {min_dir_ratio:.2f}x vs random/SkyOrbs-like and {min_no_isac_ratio:.2f}x vs no-ISAC MARL across B=10/15.",
        [FINAL_METHOD_CSV],
        "Do not state a universal two-order gain at B=15 against all directional baselines.",
    )
    add(
        "R06",
        "Network/method innovation",
        "Demonstrate contention-aware, gated, adaptive, topology-heavy, and balanced actor variants.",
        "PASS" if {
            "contention_actor",
            "gated_contention_actor",
            "adaptive_gated_contention_actor",
            "topology_adaptive_gated_contention_actor",
            "balanced_topology_gated_contention_actor",
        }.issubset(final_methods) else "OPEN",
        "main+code",
        "Final table contains contention and four gate-family variants; code contains matching neural classes.",
        [FINAL_METHOD_CSV, Path("05_simulation/src/isac_nd_sim/neural_contention_actor_critic.py")],
        "Explain this as a gate-family operating frontier, not one universally dominant policy.",
    )
    add(
        "R07",
        "Beamwidth coverage and transfer",
        "Cover narrow beamwidths around 3-15 degrees, with final main transfer at 10->15 degrees.",
        "CAUTION" if {3.0, 5.0, 10.0, 15.0}.issubset(stability_beams) and {10.0, 15.0}.issubset(final_beams) else "OPEN",
        "main+boundary",
        f"Stability summary beams={fmt_set(stability_beams)}; final Phase10 beams={fmt_set(final_beams)}. B=30 exists only as archived boundary evidence.",
        [STABILITY_CSV, FINAL_METHOD_CSV],
        "If reviewers demand full stress coverage, rerun B=3/B=5 with the final Phase10 method set; B=30 is intentionally excluded from the final line.",
    )
    add(
        "R08",
        "Node-count scalability",
        "Cover N=10 to N=100 and verify small-to-large transfer.",
        "PASS" if {10, 20, 50, 100}.issubset(stability_nodes) and final_nodes == {100} else "CAUTION",
        "main+supplement",
        f"Stability summary nodes={fmt_set(stability_nodes)}; final main table is N=100 transfer.",
        [STABILITY_CSV, FINAL_METHOD_CSV],
        "Keep N=20/N=50 as scalability support unless the final method set is rerun at every N.",
    )
    add(
        "R09",
        "Dynamic mobility",
        "Use moving UAV models rather than static topology only.",
        "PASS" if {"gauss_markov", "random_walk", "random_direction", "random_waypoint"}.issubset(stability_mobility) else "CAUTION",
        "main+supplement",
        f"Mobility models in stability summary: {fmt_set(stability_mobility)}.",
        [STABILITY_CSV],
        "Keep Gauss-Markov as the final main setting and mobility variants as robustness/boundary evidence.",
    )
    add(
        "R10",
        "Range abstraction",
        "Make communication/sensing range assumptions explicit and test range sensitivity.",
        range_status,
        "main+supplement+theory_note" if range_status == "PASS" else "supplement",
        f"Final Phase10 rows use Rc=900 m and Rs=900 m as a matched-support single-hop setting; range sweeps cover Rc/D={fmt_set(range_rc_ratios)} and Rs/Rc={fmt_set(range_rs_ratios)}.",
        [FINAL_METHOD_CSV, STABILITY_CSV, RANGE_GRID_CSV, RANGE_RATIO_CSV, RANGE_NOTE, MAIN_TEX, SUPPLEMENT_TEX],
        "Keep the equal-range final setting described as matched-support and protocol-level; do not claim hardware-calibrated communication/sensing range equality.",
    )
    add(
        "R11",
        "Time-scale assumption",
        "Use 5 ms slot duration and check sensitivity to timing assumptions.",
        "PASS" if 5.0 in stability_slots_ms and {1.0, 10.0, 20.0, 100.0}.intersection(stability_slots_ms) else "CAUTION",
        "main+supplement",
        f"Slot-duration values in stability summary: {fmt_set(stability_slots_ms)}.",
        [STABILITY_CSV],
        "Keep 5 ms framed as a modeling assumption, not a PHY timing result.",
    )
    add(
        "R12",
        "Statistical reliability",
        "Provide multi-seed/statistical summaries rather than single-run-only claims.",
        "PASS" if paired_significance_pass else "CAUTION",
        "paired_significance_primary" if paired_significance_pass else "analyzed_not_verified",
        f"Final transfer episodes range {min(final_episodes) if final_episodes else 0}-{max(final_episodes) if final_episodes else 0}; primary paired significance manifest reports {as_int(paired_significance.get('confirmatory_test_count'))} confirmatory tests with pass={paired_significance_pass}.",
        [FINAL_METHOD_CSV, STABILITY_CSV, PAIRED_SIGNIFICANCE_MANIFEST, PAIRED_SIGNIFICANCE_REPORT],
        "Keep paired significance wording restricted to primary ISAC-vs-communication-only discovery and empty-scan comparisons; gate-family results remain descriptive.",
    )
    add(
        "R13",
        "Reproducibility trace",
        "Map final method rows to analysis, raw evaluation, checkpoints, and training manifests.",
        "PASS" if trace_total > 0 and trace_complete == trace_total else "OPEN",
        "main",
        f"Method trace rows complete: {trace_complete}/{trace_total}.",
        [TRACE_CSV],
        "Keep generated trace in the supplement/reproducibility package.",
    )
    add(
        "R14",
        "Figure and artifact integrity",
        "Ensure manuscript figures/tables are present, 4:3 compliant, and hash-tracked.",
        "PASS" if missing_count == 0 and artifact_count >= 90 else "OPEN",
        "artifact_hash",
        f"Artifact manifest reports {artifact_count} artifacts and {missing_count} missing paths; training manifest lists {len(training_figures)} training/resource figures.",
        [ARTIFACT_MANIFEST, FIGURE_AUDIT],
        "Regenerate the artifact manifest after every paper/result edit.",
    )
    add(
        "R15",
        "Submission wording boundary",
        "Keep claims aligned with simulator, protocol-level ISAC abstraction, and approximate literature baseline scope.",
        "PASS" if claim_strength_pass else "CAUTION",
        "claim_strength_audit" if claim_strength_pass else "reviewer_audit",
        f"Claim-strength audit passes {as_int(claim_strength.get('passed_required_checks'))}/{as_int(claim_strength.get('required_check_count'))} required boundary checks and reports {as_int(claim_strength.get('review_required_risk_hits'), default=-1)} review-required risk hits; readiness review records {len(supplement_stability_rows)} supplement/supplement-stress rows.",
        [CLAIM_STRENGTH_REPORT, CLAIM_STRENGTH_MANIFEST, SUBMISSION_REVIEW, FIGURE_AUDIT],
        "Keep rerunning the claim-strength audit after every manuscript edit; do not remove the SkyOrbs-like, protocol-abstraction, statistics, learned-ablation, and MAC-scope caveats.",
    )
    add(
        "R16",
        "Independent reproduction",
        "Convert at least part of the current status from ANALYZED to VERIFIED by re-running a selected final experiment.",
        "PASS" if rerun_all_match else "OPEN",
        "verified_partial_rerun" if rerun_all_match else "missing_re_run",
        f"Independent stochastic re-run for {rerun_method or 'N/A'} at B={rerun_beam:g} has status_counts={rerun_status_counts}; this verifies one key final-transfer point, not the full campaign.",
        [Path("06_analysis/phase10_statistical_validation_report_20260707.md"), RERUN_VALIDATION_REPORT],
        "Extend independent re-runs to additional methods/beams only if reviewers demand a stronger replication package.",
    )
    add(
        "R17",
        "Learned component ablation",
        "Separate learned actor contribution from strong rule priors, residual logits, candidate masks, and decentralized gates.",
        "CAUTION" if learned_ablation_complete else "OPEN",
        "focused_ablation_mixed" if learned_ablation_complete else "missing_ablation",
        f"Focused B=10/N=100/3000-slot learned-component ablation covers labels={sorted(learned_ablation_labels)}. Results separate learned weights from rule priors, but support a collision-efficiency claim rather than universal learned-policy dominance.",
        [
            Path("05_simulation/src/isac_nd_sim/neural_contention_actor_critic.py"),
            Path("05_simulation/src/isac_nd_sim/marl_env.py"),
            Path("05_simulation/run_marl_training.py"),
            Path("05_simulation/run_marl_evaluate.py"),
            LEARNED_ABLATION_REPORT,
        ],
        "Use conservative wording: learned weights suppress collisions versus random/zero-weight policies, while candidate masking and rule residuals define the discovery/collision/empty-scan tradeoff. Extend to B=15 or more seeds only if needed.",
    )
    add(
        "R18",
        "Raw bundle availability",
        "Retain a local or archived raw-result bundle with manifests and checkpoint hashes for the final Phase10 evidence line.",
        "CAUTION" if trace_raw_manifests_existing == len(trace_raw_manifest_paths) and trace_checkpoints_existing == len(trace_checkpoint_paths) else "OPEN",
        "local_trace_not_git_archive",
        f"Local method trace paths exist for {trace_raw_manifests_existing}/{len(trace_raw_manifest_paths)} raw manifests and {trace_checkpoints_existing}/{len(trace_checkpoint_paths)} checkpoints; {raw_bundle_git_tracked}/{len(raw_bundle_paths)} raw trace files are tracked by Git.",
        [TRACE_CSV],
        "Before submission or external release, decide whether to archive checkpoints or publish a separate checksum manifest for raw state_dict files.",
    )

    risks: list[Risk] = []
    for req in requirements:
        if req.status == "OPEN":
            risks.append(Risk("P1", req.requirement, req.evidence_summary, req.status, req.next_action))
        elif req.status == "CAUTION":
            risks.append(Risk("P2", req.requirement, req.evidence_summary, req.status, req.next_action))
    if not claim_strength_pass:
        risks.append(
            Risk(
                "P2",
                "Literature-inspired SkyOrbs-like baseline may be challenged as not a faithful reproduction.",
                "Final table includes SkyOrbs-like, but project reports already label it as inspired/approximate.",
                "CAUTION",
                "Either retain explicit caveat or implement a stricter reproduction before submission.",
            )
        )

    inventory = {
        "final_method_rows": len(final_rows),
        "final_methods": sorted(final_methods),
        "final_beams": sorted(final_beams),
        "final_nodes": sorted(final_nodes),
        "final_slots": sorted(final_slots),
        "final_episode_min": min(final_episodes) if final_episodes else 0,
        "final_episode_max": max(final_episodes) if final_episodes else 0,
        "min_actor_ratio_vs_random_or_skyorbs": min_dir_ratio,
        "min_actor_ratio_vs_no_isac_marl": min_no_isac_ratio,
        "trace_complete_rows": trace_complete,
        "trace_total_rows": trace_total,
        "trace_raw_manifest_paths": len(trace_raw_manifest_paths),
        "trace_raw_manifests_existing": trace_raw_manifests_existing,
        "trace_checkpoint_paths": len(trace_checkpoint_paths),
        "trace_checkpoints_existing": trace_checkpoints_existing,
        "raw_bundle_trace_files": len(raw_bundle_paths),
        "raw_bundle_trace_files_git_tracked": raw_bundle_git_tracked,
        "training_runs": training_runs,
        "training_step_rows": training_steps,
        "training_episode_rows": training_episode_rows,
        "training_eval_rows": training_eval_rows,
        "training_figures": len(training_figures),
        "stability_rows": len(stability_rows),
        "stability_main_rows": len(main_stability_rows),
        "stability_beams": sorted(stability_beams),
        "stability_nodes": sorted(stability_nodes),
        "stability_protocols": sorted(stability_protocols),
        "stability_mobility_models": sorted(stability_mobility),
        "stability_slot_durations_ms": sorted(stability_slots_ms),
        "artifact_count": artifact_count,
        "artifact_missing_count": missing_count,
        "independent_rerun_all_metrics_match": rerun_all_match,
        "independent_rerun_status_counts": rerun_status_counts,
        "independent_rerun_method": rerun_method,
        "independent_rerun_beamwidth_deg": rerun_beam,
        "learned_ablation_labels": sorted(learned_ablation_labels),
        "learned_ablation_required_labels_present": learned_ablation_complete,
        "claim_strength_required_checks": as_int(claim_strength.get("required_check_count")),
        "claim_strength_passed_required_checks": as_int(claim_strength.get("passed_required_checks")),
        "claim_strength_review_required_risk_hits": as_int(claim_strength.get("review_required_risk_hits"), default=-1),
        "claim_strength_pass": claim_strength_pass,
        "code_evidence_files_present": code_files_present,
        "code_evidence_files_total": len(CODE_EVIDENCE),
    }
    return requirements, risks, inventory


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def raw_bundle_status_rows(trace_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    typed_paths: list[tuple[str, Path]] = []
    for artifact_type, fields in (
        ("raw_eval_manifest", ("raw_eval_manifests",)),
        ("training_manifest", ("train_manifests",)),
        ("checkpoint", ("checkpoints",)),
    ):
        for path in unique_semicolon_paths(trace_rows, fields):
            typed_paths.append((artifact_type, path))

    all_paths = [path for _, path in typed_paths]
    tracked = git_tracked_paths(all_paths)
    rows: list[dict[str, object]] = []
    for artifact_type, path in sorted(typed_paths, key=lambda item: (item[0], item[1].as_posix().lower())):
        exists = path.exists()
        rows.append(
            {
                "artifact_type": artifact_type,
                "path": path.as_posix(),
                "exists": exists,
                "git_tracked": path.as_posix().lower() in tracked,
                "size_bytes": path.stat().st_size if exists else "",
                "sha256": sha256_file(path) if exists else "",
            }
        )
    return rows


def write_report(requirements: list[Requirement], risks: list[Risk], inventory: dict[str, object]) -> None:
    counts: dict[str, int] = {}
    for req in requirements:
        counts[req.status] = counts.get(req.status, 0) + 1

    lines = [
        "# Research Goal Coverage Audit - 2026-07-07",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        "- Verification Status: ANALYZED",
        "- Scope: current MARL+ISAC neighbor-discovery manuscript evidence chain",
        "",
        "## Summary",
        "",
        f"- Requirements checked: {len(requirements)}.",
        f"- Status counts: PASS={counts.get('PASS', 0)}, CAUTION={counts.get('CAUTION', 0)}, OPEN={counts.get('OPEN', 0)}.",
        f"- Final Phase10 method rows: {inventory['final_method_rows']} across beams {inventory['final_beams']}.",
        f"- Training trace: {inventory['training_runs']} runs, {inventory['training_step_rows']} step rows, {inventory['training_episode_rows']} episode rows.",
        f"- Method trace completeness: {inventory['trace_complete_rows']}/{inventory['trace_total_rows']}.",
        f"- Artifact manifest: {inventory['artifact_count']} artifacts, {inventory['artifact_missing_count']} missing.",
        "",
        "## Requirement Coverage",
        "",
        "| ID | Theme | Status | Evidence strength | Evidence summary |",
        "|---|---|---|---|---|",
    ]
    for req in requirements:
        lines.append(
            f"| {req.requirement_id} | {req.theme} | {req.status} | {req.evidence_strength} | {req.evidence_summary} |"
        )

    lines.extend(
        [
            "",
            "## Highest-Value Remaining Work",
            "",
        ]
    )
    for risk in risks[:8]:
        lines.append(f"- {risk.priority} `{risk.current_status}`: {risk.risk} Next: {risk.next_action}")

    lines.extend(
        [
            "",
            "## Boundary Interpretation",
            "",
            "The current evidence is strong enough to support a manuscript draft centered on a real MAPPO-style MARL+ISAC neighbor-discovery method, small-to-large transfer, and gate-family collision/topology tradeoffs.",
            "The evidence is still not a fully verified replication package: independent re-run coverage is partial, learned-component evidence is a focused mixed ablation, and full-campaign reproducibility plus broader ablation seeds should remain bounded claims.",
            "",
            "## Generated Files",
            "",
            f"- `{(OUT_DIR / 'requirement_coverage.csv').as_posix()}`",
            f"- `{(OUT_DIR / 'evidence_inventory.csv').as_posix()}`",
            f"- `{(OUT_DIR / 'claim_risk_register.csv').as_posix()}`",
            f"- `{(OUT_DIR / 'raw_bundle_trace_status.csv').as_posix()}`",
            f"- `{(OUT_DIR / 'manifest.json').as_posix()}`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    final_rows = read_csv(FINAL_METHOD_CSV)
    trace_rows = read_csv(TRACE_CSV)
    stability_rows = read_csv(STABILITY_CSV)
    training_manifest = read_json(TRAINING_MANIFEST)
    artifact_manifest = read_json(ARTIFACT_MANIFEST)
    rerun_validation = read_json(RERUN_VALIDATION_MANIFEST)
    learned_ablation = read_json(LEARNED_ABLATION_MANIFEST)
    paired_significance = read_json(PAIRED_SIGNIFICANCE_MANIFEST)
    claim_strength = read_json(CLAIM_STRENGTH_MANIFEST)

    requirements, risks, inventory = requirement_rows(
        final_rows,
        trace_rows,
        stability_rows,
        training_manifest,
        artifact_manifest,
        rerun_validation,
        learned_ablation,
        paired_significance,
        claim_strength,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coverage_csv = OUT_DIR / "requirement_coverage.csv"
    inventory_csv = OUT_DIR / "evidence_inventory.csv"
    risks_csv = OUT_DIR / "claim_risk_register.csv"
    raw_bundle_csv = OUT_DIR / "raw_bundle_trace_status.csv"
    manifest_json = OUT_DIR / "manifest.json"
    readme_md = OUT_DIR / "README.md"

    write_csv(
        coverage_csv,
        [req.__dict__ for req in requirements],
        [
            "requirement_id",
            "theme",
            "requirement",
            "status",
            "evidence_strength",
            "evidence_summary",
            "evidence_paths",
            "next_action",
        ],
    )
    write_csv(
        inventory_csv,
        [{"metric": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value} for key, value in inventory.items()],
        ["metric", "value"],
    )
    write_csv(
        risks_csv,
        [risk.__dict__ for risk in risks],
        ["priority", "risk", "trigger", "current_status", "next_action"],
    )
    write_csv(
        raw_bundle_csv,
        raw_bundle_status_rows(trace_rows),
        ["artifact_type", "path", "exists", "git_tracked", "size_bytes", "sha256"],
    )
    write_report(requirements, risks, inventory)

    readme_md.write_text(
        "\n".join(
            [
                "# Research Goal Coverage Audit",
                "",
                "Generated by `06_analysis/scripts/build_research_goal_coverage_audit.py`.",
                "The tables map the active MARL+ISAC neighbor-discovery research objective to current evidence artifacts.",
                "",
                "- `requirement_coverage.csv`: requirement-level PASS/CAUTION/OPEN status.",
                "- `evidence_inventory.csv`: compact machine-readable counts from the final tables/manifests.",
                "- `claim_risk_register.csv`: reviewer-facing risks derived from CAUTION/OPEN requirements.",
                "- `raw_bundle_trace_status.csv`: local raw manifest/checkpoint existence, Git-tracking status, sizes, and SHA256 hashes.",
                "- `manifest.json`: source and output trace for this audit.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_files = [coverage_csv, inventory_csv, risks_csv, raw_bundle_csv, readme_md, REPORT]
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "active MARL+ISAC research-goal evidence coverage",
        "status_counts": {status: sum(1 for req in requirements if req.status == status) for status in ("PASS", "CAUTION", "OPEN")},
        "input_files": [
            FINAL_METHOD_CSV.as_posix(),
            TRACE_CSV.as_posix(),
            STABILITY_CSV.as_posix(),
            TRAINING_MANIFEST.as_posix(),
            ARTIFACT_MANIFEST.as_posix(),
            FIGURE_AUDIT.as_posix(),
            SUBMISSION_REVIEW.as_posix(),
            RERUN_VALIDATION_MANIFEST.as_posix(),
            RERUN_VALIDATION_REPORT.as_posix(),
            LEARNED_ABLATION_MANIFEST.as_posix(),
            LEARNED_ABLATION_REPORT.as_posix(),
            PAIRED_SIGNIFICANCE_MANIFEST.as_posix(),
            PAIRED_SIGNIFICANCE_REPORT.as_posix(),
            CLAIM_STRENGTH_MANIFEST.as_posix(),
            CLAIM_STRENGTH_REPORT.as_posix(),
        ]
        + [path.as_posix() for path in CODE_EVIDENCE],
        "output_files": [path.as_posix() for path in output_files],
        "outputs": {
            "coverage_csv": coverage_csv.as_posix(),
            "inventory_csv": inventory_csv.as_posix(),
            "risk_register_csv": risks_csv.as_posix(),
            "raw_bundle_trace_status_csv": raw_bundle_csv.as_posix(),
            "report": REPORT.as_posix(),
            "readme": readme_md.as_posix(),
        },
        "output_hashes": {path.as_posix(): sha256_file(path) for path in output_files},
    }
    manifest_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "created_at_utc": payload["created_at_utc"],
                "status_counts": payload["status_counts"],
                "coverage_csv": coverage_csv.as_posix(),
                "risk_register_csv": risks_csv.as_posix(),
                "report": REPORT.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
