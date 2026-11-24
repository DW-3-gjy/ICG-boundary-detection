"""Read-only, standard-library view of the patient-boundary study's progress.

Training counts are exact marker counts. Fractions are run-count equivalents,
not elapsed-time fractions. Time estimates are deliberately approximate and
never open model weights, images, annotations or sealed test caches.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

STALE_AFTER_SECONDS = 180
CHINA = timezone(timedelta(hours=8))
RUN_PATTERN = re.compile(r"^(?P<group>[^_]+)_(?P<fold>fold_\d+|refit)_s(?P<seed>\d+)$")
DESCRIPTIONS = {
    "A": "区域模型 · 旧＋新Z",
    "B": "边界分支 · 旧＋新Z",
    "C": "区域模型 · Z及荧光可见L",
    "D": "边界分支 · Z及荧光可见L",
    "E": "边界分支 · 仅新数据Z及L",
}


def _number(value, default=None):
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _integer(value, default=0):
    result = _number(value)
    return int(result) if result is not None else default


def _json(path, expected, warnings):
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, expected):
            raise ValueError("unexpected JSON type")
        return value
    except FileNotFoundError:
        return expected()
    except (OSError, ValueError, UnicodeError):
        warnings.append(f"{path.name}暂时不可读，下一次刷新会重试")
        return expected()


def _mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _timestamp(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=CHINA)
        return parsed.timestamp()
    except (ValueError, OverflowError):
        return None


def _elapsed(history):
    """Sum per-session counters without counting the wall-clock pause."""
    elapsed, previous = 0.0, None
    for item in history:
        if not isinstance(item, dict):
            continue
        current = _number(item.get("elapsed_seconds_this_session"))
        if current is None or current < 0:
            continue
        if previous is not None and current < previous:
            elapsed += previous
        previous = current
    return elapsed + (previous or 0.0)


def _recent_rate(history):
    """Seconds/update from the last session, never across a counter reset."""
    segment = []
    previous = None
    reset_update = 0
    for item in history:
        if not isinstance(item, dict):
            continue
        update = _integer(item.get("update"), -1)
        elapsed = _number(item.get("elapsed_seconds_this_session"))
        if update < 0 or elapsed is None or elapsed < 0:
            continue
        if previous and (elapsed < previous[1] or update <= previous[0]):
            segment = []
            reset_update = previous[0]
        segment.append((update, elapsed))
        previous = update, elapsed
    recent = segment[-8:]
    rates = [(b[1] - a[1]) / (b[0] - a[0])
             for a, b in zip(recent, recent[1:]) if b[0] > a[0] and b[1] > a[1]]
    if rates:
        return statistics.median(rates)
    if recent and recent[-1][0] > reset_update and recent[-1][1] > 0:
        return recent[-1][1] / (recent[-1][0] - reset_update)
    return None


def _last_traceback_is_new(study):
    """Do not mistake a traceback preceding later resumed progress for failure."""
    try:
        with (study / "training_console.log").open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 65536))
            tail = stream.read().decode("utf-8", errors="replace")
        traceback = tail.rfind("Traceback (most recent call last)")
        progress = max(tail.rfind('"run":'), tail.rfind('"run" :'))
        return traceback >= 0 and traceback > progress
    except OSError:
        return False


def _budget(values, maximum):
    if not values:
        return maximum
    return max(40, min(maximum, int(40 * round(statistics.median(values) / 40))))


def snapshot(study_path: Path) -> dict:
    study = Path(study_path)
    warnings = []
    now = time.time()
    protocol = _json(study / "protocol.json", dict, warnings)
    progress = _json(study / "progress.json", dict, warnings)
    pause = _json(study / "pause_state.json", dict, warnings)
    pipeline = _json(study / "report_pipeline_state.json", dict, warnings)
    report_error = _json(study / "report_pipeline_error.json", dict, warnings)
    final = _json(study / "final_summary.json", dict, warnings)
    benchmark = _json(study / "pipeline_benchmark_micro4.json", dict, warnings)

    group_ids = [g["id"] for g in protocol.get("groups", []) if isinstance(g, dict) and g.get("id")]
    group_ids = list(dict.fromkeys(group_ids)) or list("ABCDE")
    seeds = [_integer(v) for v in protocol.get("seeds", [42, 43, 44])]
    folds = list(protocol.get("planned_development_validation_folds", {})) or ["fold_1", "fold_2", "fold_3"]
    maximum = max(1, _integer(protocol.get("development_updates"), 2400))
    dev_total = len(group_ids) * len(folds) * len(seeds)
    refit_total = len(group_ids) * len(seeds)
    total = dev_total + refit_total
    expected = {f"{g}_{f}_s{s}": (g, f, s)
                for g in group_ids for f in folds + ["refit"] for s in seeds}
    completed, histories, rates = {}, {}, {}
    try:
        directories = list((study / "runs").iterdir())
    except OSError:
        directories = []
    for directory in directories:
        run_id = directory.name
        if run_id not in expected or not directory.is_dir():
            continue
        complete = _json(directory / "complete.json", dict, warnings)
        if complete and _integer(complete.get("updates")) > 0:
            completed[run_id] = complete
        history = _json(directory / "history.json", list, warnings)
        histories[run_id] = history
        rate = _recent_rate(history)
        if rate is not None:
            rates[run_id] = rate

    run_id = progress.get("run", "") if isinstance(progress.get("run", ""), str) else ""
    match = RUN_PATTERN.match(run_id)
    current_group = match["group"] if match else ""
    fold = match["fold"] if match else ""
    seed = int(match["seed"]) if match else None
    update = max(0, _integer(progress.get("update")))
    current_maximum = max(1, _integer(progress.get("max_updates"), maximum))
    current_fraction = min(1.0, update / current_maximum)
    if run_id not in expected:
        current_fraction = 0.0
    extra_fraction = current_fraction if run_id not in completed else 0.0
    progress_timestamp = _timestamp(progress.get("time")) or _mtime(study / "progress.json")
    age = max(0.0, now - progress_timestamp) if progress_timestamp is not None else 0.0
    dev_completed = sum(expected[r][1] != "refit" for r in completed)
    refit_completed = len(completed) - dev_completed

    paused_at = _timestamp(pause.get("paused_at")) or _mtime(study / "pause_state.json")
    explicitly_paused = (pause.get("status") == "paused_by_user" and
                         (progress_timestamp is None or paused_at is None or paused_at >= progress_timestamp))
    test_started = (study / "final_test_started.json").exists()
    final_available = bool(final)
    pipeline_stage = pipeline.get("stage", "")
    report_path = pipeline.get("final_report")
    report_ready = (pipeline_stage in ("complete", "complete_pending_human_visual_review")
                    and isinstance(report_path, str) and Path(report_path).is_file())
    stage_error = False
    if report_error:
        error_time = _timestamp(report_error.get("time")) or _mtime(study / "report_pipeline_error.json")
        if ((final_available or test_started) and error_time is not None
                and (progress_timestamp is None or error_time >= progress_timestamp)):
            stage_error = True
        else:
            warnings.append("有历史报告错误记录；未据此判断当前训练失败")

    if report_ready:
        status, status_text = "completed", "训练与测试已完成，报告已生成"
        if pipeline_stage == "complete_pending_human_visual_review":
            status_text += "（待排版复核）"
    elif explicitly_paused:
        status, status_text = "paused", "已按要求暂停，检查点已保留"
    elif stage_error:
        status, status_text = "error", "测试或报告流程发生错误，请检查日志"
    elif final_available or pipeline_stage == "verifying_metrics_and_visual_replay":
        status, status_text = "reporting", "测试结果已生成，正在复算与制作报告"
    elif test_started:
        status, status_text = "testing", "训练已完成，正在统一执行独立测试"
    elif len(completed) >= total:
        status, status_text = "waiting", "训练已完成，等待最终测试"
    elif progress and age > STALE_AFTER_SECONDS and _last_traceback_is_new(study):
        status, status_text = "error", "训练日志出现异常，请检查运行日志"
    elif progress and age > STALE_AFTER_SECONDS:
        status, status_text = "stale", "训练进度暂未更新，请检查进程是否仍在运行"
    elif progress:
        status, status_text = "running", "正在训练" if run_id not in completed else "正在切换下一条训练"
    else:
        status, status_text = "waiting", "等待训练进度日志"

    groups = []
    per_group_total = (len(folds) + 1) * len(seeds)
    for group in group_ids:
        gd = sum(expected[r][0] == group and expected[r][1] != "refit" for r in completed)
        gr = sum(expected[r][0] == group and expected[r][1] == "refit" for r in completed)
        partial = extra_fraction if current_group == group else 0.0
        groups.append({"id": group, "description": DESCRIPTIONS.get(group, group),
                       "dev_completed": gd, "refit_completed": gr, "total": per_group_total,
                       "fraction": min(1.0, (gd + gr + partial) / per_group_total)})

    eta = None
    eta_note = "粗略估计，不含最终测试和报告耗时；最终重训步数由开发验证决定。"
    global_dev_rates = [v for r, v in rates.items() if expected[r][1] != "refit"]
    global_refit_rates = [v for r, v in rates.items() if expected[r][1] == "refit"]
    dev_rate = statistics.median(global_dev_rates) if global_dev_rates else None
    training_only_rate = _number(benchmark.get("median_seconds_per_update_after_warmup"))
    refit_rate = (statistics.median(global_refit_rates) if global_refit_rates else
                  training_only_rate if training_only_rate and training_only_rate > 0 else
                  dev_rate * .65 if dev_rate else None)
    all_selected = [_integer(v.get("best_update")) for r, v in completed.items()
                    if expected[r][1] != "refit" and _integer(v.get("best_update")) > 0]
    budgets = {}
    estimated_budget = False
    for group in group_ids:
        selected = [_integer(v.get("best_update")) for r, v in completed.items()
                    if expected[r][0] == group and expected[r][1] != "refit"
                    and _integer(v.get("best_update")) > 0]
        if len(selected) == len(folds) * len(seeds):
            budgets[group] = _budget(selected, maximum)
        else:
            budgets[group] = _budget(selected or all_selected, maximum)
            estimated_budget = True

    if status == "running" and dev_rate is not None and refit_rate is not None:
        eta = 0.0
        for task_id, (group, task_fold, _) in expected.items():
            if task_id in completed:
                continue
            budget = maximum if task_fold != "refit" else budgets[group]
            if task_id == run_id:
                budget = max(0, current_maximum - update)
            if task_id in rates:
                rate = rates[task_id]
            elif task_fold == "refit":
                rate = refit_rate
            else:
                matched = [v for r, v in rates.items() if expected[r][:2] == (group, task_fold)]
                matched = matched or [v for r, v in rates.items() if expected[r][1] == task_fold]
                rate = statistics.median(matched) if matched else dev_rate
            eta += budget * rate
        if estimated_budget:
            eta_note += " 未定重训预算按已完成选轮的中位数估计（40至最大更新数范围），可能变化。"
    elif status == "completed":
        eta = 0.0
        eta_note = "训练、测试和报告生成已完成。"
    elif status == "running":
        eta_note = "等待足够的运行记录估算剩余训练时间；不含最终测试和报告。"
    elif status == "paused":
        eta_note = "当前已暂停，恢复训练后重新估算。"
    elif status in ("testing", "reporting"):
        eta_note = "训练已完成，测试或报告耗时不作训练ETA估算。"

    active_history = histories.get(run_id, [])
    latest_validation = None
    for item in reversed(active_history):
        if isinstance(item, dict):
            latest_validation = _number(item.get("validation_patient_macro_f1"))
            if latest_validation is not None:
                break
    return {
        "status": status, "status_text": status_text,
        "updated_at": progress.get("time", ""), "age_seconds": age,
        "current_run": run_id, "group": current_group,
        "fold_label": "最终重训" if fold == "refit" else f"第{fold.split('_')[-1]}折" if fold else "等待运行",
        "seed": seed, "update": update, "max_updates": current_maximum,
        "current_fraction": current_fraction,
        "completed_total": len(completed), "total_runs": total,
        "dev_completed": dev_completed, "dev_total": dev_total,
        "refit_completed": refit_completed, "refit_total": refit_total,
        "overall_fraction": min(1.0, (len(completed) + extra_fraction) / total),
        "overall_basis": "按训练运行次数计算，非按耗时；不含测试和报告阶段",
        "groups": groups,
        "active_elapsed_seconds": sum(_elapsed(history) for history in histories.values()),
        "elapsed_note": "按各次运行日志累计，扣除已识别暂停；不含初始化等开销，为近似值",
        "eta_seconds": eta, "eta_note": eta_note,
        "latest_validation": latest_validation,
        "best_validation": _number(progress.get("best_validation_f1")),
        "training_loss": _number(progress.get("training_loss")),
        "test_started": test_started, "final_results_available": final_available,
        "report_ready": report_ready, "warnings": list(dict.fromkeys(warnings)),
    }
