"""Progress must remain truthful across pauses, stale logs and stage changes."""
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from training_progress_state import snapshot, _elapsed, _recent_rate


NOW = datetime(2026, 11, 5, 10, 0, tzinfo=timezone.utc).timestamp()


class ProgressStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.study = Path(self.temp.name)
        self.clock = patch("training_progress_state.time.time", return_value=NOW)
        self.clock.start()

    def tearDown(self):
        self.clock.stop()
        self.temp.cleanup()

    def write(self, path, value):
        target = self.study / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value), encoding="utf-8")
        return target

    def progress(self, update=800, time="2026-11-05 17:59:50", run="A_fold_1_s43"):
        return self.write("progress.json", dict(run=run, update=update, max_updates=2400,
                                               training_loss=.2, best_validation_f1=.3, time=time))

    def complete(self, run="A_fold_1_s42", best_update=1200):
        return self.write(f"runs/{run}/complete.json", dict(run=run, updates=2400,
                         best_update=best_update, group=run[0], fold="fold_1", seed=42))

    def history(self, run="A_fold_1_s43", records=None):
        records = records or [(40, 20), (80, 40), (120, 60)]
        return self.write(f"runs/{run}/history.json", [dict(update=u, elapsed_seconds_this_session=t,
                          validation_patient_macro_f1=.2) for u, t in records])

    def test_missing_study_waits_without_inventing_progress(self):
        result = snapshot(self.study / "not-created")
        self.assertEqual(result["status"], "waiting")
        self.assertEqual(result["completed_total"], 0)
        self.assertEqual(result["total_runs"], 60)
        self.assertIsNone(result["eta_seconds"])

    def test_completed_active_run_is_not_counted_twice(self):
        self.complete()
        self.progress(update=2400, run="A_fold_1_s42")
        result = snapshot(self.study)
        self.assertEqual(result["completed_total"], 1)
        self.assertAlmostEqual(result["overall_fraction"], 1 / 60)
        self.assertAlmostEqual(result["groups"][0]["fraction"], 1 / 12)

    def test_partial_run_contributes_only_its_fraction(self):
        self.complete()
        self.progress(update=1200)
        result = snapshot(self.study)
        self.assertAlmostEqual(result["overall_fraction"], 1.5 / 60)

    def test_pause_has_priority_over_stale_progress(self):
        self.progress(time="2026-11-05 17:00:00")
        self.write("pause_state.json", dict(status="paused_by_user", paused_at="2026-11-05T17:01:00+08:00"))
        result = snapshot(self.study)
        self.assertEqual(result["status"], "paused")
        self.assertIsNone(result["eta_seconds"])

    def test_later_progress_revokes_historical_pause_state(self):
        self.progress()
        self.write("pause_state.json", dict(status="paused_by_user", paused_at="2026-11-05T17:00:00+08:00"))
        self.assertEqual(snapshot(self.study)["status"], "running")

    def test_explicit_resumed_state_does_not_repause(self):
        self.progress()
        self.write("pause_state.json", dict(status="resumed_running", paused_at="2026-11-05T17:59:59+08:00"))
        self.assertEqual(snapshot(self.study)["status"], "running")

    def test_stale_log_does_not_claim_active_training(self):
        self.progress(time="2026-11-05 17:00:00")
        self.assertEqual(snapshot(self.study)["status"], "stale")

    def test_historical_traceback_preceding_resumed_progress_is_not_failure(self):
        self.progress(time="2026-11-05 17:00:00")
        path = self.study / "training_console.log"
        path.write_text('Traceback (most recent call last)\nold failure\n{"run": "A_fold_1_s43"}\n', encoding="utf-8")
        self.assertEqual(snapshot(self.study)["status"], "stale")
        path.write_text('{"run": "A_fold_1_s43"}\nTraceback (most recent call last)\nnew failure\n', encoding="utf-8")
        self.assertEqual(snapshot(self.study)["status"], "error")

    def test_old_report_error_does_not_override_current_training(self):
        self.progress()
        self.write("report_pipeline_error.json", dict(error="old renderer error", time="2026-11-05 17:00:00"))
        result = snapshot(self.study)
        self.assertEqual(result["status"], "running")
        self.assertTrue(result["warnings"])

    def test_counter_reset_does_not_include_the_pause_in_runtime_or_eta(self):
        records = [dict(update=u, elapsed_seconds_this_session=t) for u, t in
                   [(40, 10), (80, 20), (120, 5), (160, 15)]]
        self.assertEqual(_elapsed(records), 35)
        self.assertEqual(_recent_rate(records), .25)

    def test_eta_is_in_seconds_and_excludes_completed_budget(self):
        self.complete(best_update=1200)
        self.progress(update=120)
        self.history()
        self.write("pipeline_benchmark_micro4.json", dict(median_seconds_per_update_after_warmup=.25))
        result = snapshot(self.study)
        expected = (43 * 2400 + (2400 - 120)) * .5 + 15 * 1200 * .25
        self.assertAlmostEqual(result["eta_seconds"], expected)
        self.assertEqual(result["active_elapsed_seconds"], 60)

    def test_atomic_write_or_invalid_json_is_nonfatal(self):
        (self.study / "progress.json").write_text('{"run":', encoding="utf-8")
        result = snapshot(self.study)
        self.assertEqual(result["status"], "waiting")
        self.assertTrue(result["warnings"])

    def test_test_and_report_stages_do_not_show_training_eta(self):
        self.progress()
        self.write("final_test_started.json", {})
        self.assertEqual(snapshot(self.study)["status"], "testing")
        self.write("final_summary.json", dict(groups={"A": {}}))
        self.assertEqual(snapshot(self.study)["status"], "reporting")
        report = self.study / "report.pdf"
        report.write_bytes(b"report exists")
        self.write("report_pipeline_state.json", dict(stage="complete_pending_human_visual_review", final_report=str(report)))
        result = snapshot(self.study)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["report_ready"])
        self.assertEqual(result["eta_seconds"], 0)
        self.assertIn("待排版复核", result["status_text"])

    def test_all_training_complete_is_not_mislabeled_stale_training(self):
        for group in "ABCDE":
            for fold in ["fold_1", "fold_2", "fold_3", "refit"]:
                for seed in [42, 43, 44]:
                    self.complete(f"{group}_{fold}_s{seed}")
        self.progress(update=2400, run="E_refit_s44", time="2026-11-05 17:00:00")
        result = snapshot(self.study)
        self.assertEqual(result["completed_total"], 60)
        self.assertEqual(result["overall_fraction"], 1.)
        self.assertEqual(result["status"], "waiting")
        self.assertIn("训练已完成", result["status_text"])


if __name__ == "__main__":
    unittest.main()
