"""Paired comparisons with explicit compatibility and promotion gates."""

import itertools
import random
from pathlib import Path

from legal_rag.artifacts import file_hash, read_json

from .runner import BOOTSTRAP_SEED, load_results, summarize


def _interval(values: list[float], groups: list[str], resamples: int) -> dict:
    rng = random.Random(BOOTSTRAP_SEED)
    grouped = {}
    for value, group in zip(values, groups):
        grouped.setdefault(group, []).append(value)
    clusters = list(grouped.values())
    samples = []
    for _ in range(resamples):
        selected = rng.choices(clusters, k=len(clusters))
        samples.append(sum(sum(cluster) for cluster in selected) / sum(len(cluster) for cluster in selected))
    samples.sort()
    return {"delta": sum(values) / len(values), "low": samples[int(.025 * (resamples - 1))],
            "high": samples[int(.975 * (resamples - 1))], "level": .95, "seed": BOOTSTRAP_SEED,
            "resamples": resamples, "clusters": len(clusters)}


def comparison_issues(left: dict, right: dict) -> list[str]:
    issues = []
    if not left.get("complete") or not right.get("complete"):
        issues.append("incomplete_run")
    for field in ("benchmark_hash", "view", "qids", "formats", "identity", "code", "control_qids", "control_identity"):
        if left["inputs"].get(field) != right["inputs"].get(field):
            issues.append(f"incompatible_{field}")
    if left.get("source_hashes") != right.get("source_hashes"):
        issues.append("incompatible_source_hashes")
    return issues


def compare_runs(run_dirs: list[Path | str], *, resamples: int = 10000) -> dict:
    if resamples < 100:
        raise ValueError("At least 100 bootstrap resamples are required")
    runs = []
    for directory in run_dirs:
        path = Path(directory)
        manifest = read_json(path / "manifest.json")
        rows = load_results(path)
        integrity = []
        for name, expected in manifest.get("output_hashes", {}).items():
            if not (path / name).exists() or file_hash(path / name) != expected:
                integrity.append(f"artifact_hash_mismatch:{name}")
        keys = {(row.qid, row.format) for row in rows}
        expected_keys = {(qid, format_name) for qid in manifest["inputs"]["qids"] for format_name in manifest["inputs"]["formats"]}
        if keys != expected_keys:
            integrity.append("result_selection_mismatch")
        runs.append((path, manifest, rows, integrity))
    comparisons = []
    for (left_path, left, left_rows, left_integrity), (right_path, right, right_rows, right_integrity) in itertools.combinations(runs, 2):
        issues = [*left_integrity, *right_integrity, *comparison_issues(left, right)]
        item = {"left": str(left_path), "right": str(right_path), "issues": issues, "comparable": not issues,
                "interpretation": "Exploratory paired comparison on previously explored questions; not an unseen-test claim."}
        if issues:
            comparisons.append(item)
            continue
        item["formats"] = {}
        for format_name in left["inputs"]["formats"]:
            lrows = {row.qid: row for row in left_rows if row.format == format_name}
            rrows = {row.qid: row for row in right_rows if row.format == format_name}
            qids = left["inputs"]["qids"]
            if any(lrows[qid].score is None or rrows[qid].score is None for qid in qids):
                raise ValueError("A complete run contains missing scores")
            for role in ("generator", "judge", "control_judge"):
                def successful_usage(row):
                    if role == "generator":
                        return row.answer.usage[-1:] if row.answer.status != "error" else []
                    if role == "judge":
                        return row.judge_usage[-1:] if row.judge is not None else []
                    return row.control_usage[-1:] if row.control_judge is not None else []
                if any(usage.returned_model == "unknown" for rows in (lrows, rrows) for row in rows.values()
                       for usage in successful_usage(row)):
                    item["issues"].append(f"unknown_effective_{role}")
                def models(rows):
                    return {(usage.provider, usage.requested_model, usage.returned_model)
                            for row in rows.values() for usage in successful_usage(row)
                            if usage.returned_model != "unknown"}
                if models(lrows) != models(rrows):
                    item["issues"].append(f"incompatible_effective_{role}")
            deltas = [float(rrows[qid].score == 2) - float(lrows[qid].score == 2) for qid in qids]
            ordinal = [rrows[qid].score - lrows[qid].score for qid in qids]
            laws = [next((ref.document_id for ref in lrows[qid].expected_references if ref.document_id), qid) for qid in qids]
            lm, rm = summarize(list(lrows.values()), len(qids)), summarize(list(rrows.values()), len(qids))
            citations_safe = rm["citation_validity"] is not None and (lm["citation_validity"] is None or rm["citation_validity"] >= lm["citation_validity"])
            if not lm["citation_count"] and not rm["citation_count"]:
                citations_safe = True
            planned_control = left["inputs"].get("control_qids", [])
            def control_score(row):
                if row.control_judge:
                    return row.control_judge.correctness
                if row.answer.status == "error" or "empty_generated_answer" in row.flags:
                    return 0
                return None
            controlled = [qid for qid in planned_control if qid in qids and control_score(lrows[qid]) is not None and control_score(rrows[qid]) is not None]
            primary_control_delta = sum(deltas[qids.index(qid)] for qid in controlled)
            secondary_control_delta = sum(float(control_score(rrows[qid]) == 2) - float(control_score(lrows[qid]) == 2) for qid in controlled)
            reversal = bool(controlled and sum(deltas) * secondary_control_delta < 0)
            def full_selection(manifest):
                return manifest["selected_intents"] == manifest["view_intents"] and not manifest.get("view_provisional", False) and (
                    manifest["inputs"]["view"] != "legacy" or manifest["selected_intents"] == manifest["original_intents"])
            def distinct_control(manifest):
                primary = manifest["inputs"]["identity"]["judge"]
                secondary = manifest["inputs"].get("control_identity") or {}
                if not isinstance(primary, dict) or not isinstance(secondary, dict):
                    return False
                primary_model = (primary.get("provider"), primary.get("model"))
                secondary_model = (secondary.get("provider"), secondary.get("model"))
                return all(primary_model) and all(secondary_model) and primary_model != secondary_model
            gates = {"increased_fully_correct_rate": sum(deltas) > 0,
                     "deployable_treatment": right["inputs"]["profile"] != "oracle",
                     "no_hint_primary": format_name == "no_hint", "full_view_selection": full_selection(left) and full_selection(right),
                     "control_present": bool(left["inputs"].get("control_identity") and right["inputs"].get("control_identity")),
                     "distinct_control_model": bool(distinct_control(left) and distinct_control(right)),
                     "control_sample_coverage": len(controlled) >= min(20, len(qids)),
                     "no_new_generation_errors": rm["generation_errors"] <= lm["generation_errors"],
                     "citation_validity_not_worse": citations_safe,
                     "complete_judgments": True, "control_complete": left.get("control_complete", True) and right.get("control_complete", True),
                     "control_does_not_reverse_winner": not reversal}
            item["formats"][format_name] = {"intents": len(qids), "fully_correct": _interval(deltas, qids, resamples),
                "ordinal_mean": _interval(ordinal, qids, resamples), "law_cluster_sensitivity": _interval(deltas, laws, resamples),
                "law_cluster_definition": "First parsed expected law; sensitivity only, not an independence guarantee.",
                "improved": sum(value > 0 for value in deltas), "degraded": sum(value < 0 for value in deltas),
                "tied": sum(value == 0 for value in deltas), "control_paired": len(controlled),
                "control_primary_sample_delta": primary_control_delta / len(controlled) if controlled else None,
                "control_secondary_sample_delta": secondary_control_delta / len(controlled) if controlled else None,
                "promotion_gates": gates, "promoted": all(gates.values()), "tie_policy": "Keep the simpler treatment."}
        item["comparable"] = not item["issues"]
        if item["issues"]:
            for values in item["formats"].values():
                values["promoted"] = False
        comparisons.append(item)
    return {"schema_version": "comparison-v2", "comparisons": comparisons, "seed": BOOTSTRAP_SEED,
            "runs": [{"path": str(path), "integrity_issues": issues} for path, _, _, issues in runs]}
