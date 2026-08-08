from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trendmodel.tf_train import build_parser as build_tensorflow_parser
from trendmodel.tf_data import build_label_maps, load_task_records
from trendmodel.tf_features import STRUCTURED_FEATURE_NAMES
from trendmodel.tf_model import build_transformer_challenger
from trendmodel.tf_train import train_model
from trendmodel.train import build_parser as build_project_parser


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "examples.jsonl"


class TrainingArchitectureCliTest(unittest.TestCase):
    def test_project_train_defaults_to_bigru(self) -> None:
        args = build_project_parser().parse_args(["train"])

        self.assertEqual(args.architecture, "bigru")

    def test_project_train_accepts_transformer_challenger(self) -> None:
        args = build_project_parser().parse_args(
            ["train", "--architecture", "transformer-challenger"]
        )

        self.assertEqual(args.architecture, "transformer-challenger")

    def test_tensorflow_train_parser_matches_project_parser(self) -> None:
        args = build_tensorflow_parser().parse_args(
            ["train", "--architecture", "transformer-challenger"]
        )

        self.assertEqual(args.architecture, "transformer-challenger")

    def test_train_rejects_unknown_architecture_before_tensorflow_import(self) -> None:
        with self.assertRaisesRegex(ValueError, "architecture must be"):
            train_model("data/examples.jsonl", "artifacts/test", architecture="cnn")

    def test_transformer_builds_and_round_trips_when_tensorflow_is_available(self) -> None:
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("TensorFlow is not installed in this dependency-free test runtime")

        records = load_task_records(DATA_PATH)
        model = build_transformer_challenger(
            [record.original_prompt for record in records],
            build_label_maps(records),
        )
        inputs = {
            "original_prompt": tf.constant([records[0].original_prompt]),
            "structured_features": tf.zeros((1, len(STRUCTURED_FEATURE_NAMES))),
        }
        outputs = model.predict(inputs, verbose=0)
        self.assertEqual(
            set(outputs),
            {
                "workload_type",
                "intent",
                "complexity",
                "effort_bucket",
                "effort_ordinal",
                "effort_log_minutes",
                "needs_clarification",
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "challenger.keras"
            model.save(artifact_path)
            reloaded = tf.keras.models.load_model(artifact_path, compile=False)
            reloaded_outputs = reloaded.predict(inputs, verbose=0)
            self.assertEqual(set(reloaded_outputs), set(outputs))


if __name__ == "__main__":
    unittest.main()
