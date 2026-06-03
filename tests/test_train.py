import numpy as np
import pandas as pd

import tif.train
import tif.utils


def test_pad_token_sequences_uses_utils_max_tokens() -> None:
    sequences = pd.Series([[1, 2, 3], [4] * (tif.utils.MAX_TOKENS + 5), None])

    matrix = tif.train._pad_token_sequences(sequences)

    assert matrix.shape == (3, tif.utils.MAX_TOKENS)
    assert matrix[0, :3].tolist() == [1, 2, 3]
    assert matrix[1, -1] == 4
    assert np.count_nonzero(matrix[2]) == 0


def test_sequence_plan_uses_lag_features() -> None:
    variables, steps = tif.train._sequence_plan(["a_lag_1", "a_lag_2", "b_lag_1", "plain"], (2, 1, 0))

    assert variables == ["a"]
    assert steps == [2, 1, 0]


def test_training_config_reads_architecture_environment(monkeypatch) -> None:
    monkeypatch.setenv("TIF_SEQUENCE_STEPS", "6,3,1")
    monkeypatch.setenv("TIF_TEXT_KERNEL_SIZES", "2,4")
    monkeypatch.setenv("TIF_FUSION_HIDDEN_SIZE", "96")

    config = tif.train.TrainingConfig.from_environment()

    assert config.sequence_steps == (6, 3, 1)
    assert config.text_kernel_sizes == (2, 4)
    assert config.fusion_hidden_size == 96


def test_training_history_frame_extracts_neural_histories() -> None:
    summary = {
        "models": {
            "last_value": {"type": "baseline"},
            "numeric_mlp": {
                "type": "deep_numeric",
                "history": [
                    {"epoch": 1, "train_loss": 2.0, "validation_loss": 3.0},
                    {"epoch": 2, "train_loss": 1.0, "validation_loss": 2.0},
                ],
            },
        }
    }

    history = tif.train._training_history_frame(summary)

    assert history["model_name"].tolist() == ["numeric_mlp", "numeric_mlp"]
    assert history["validation_loss"].tolist() == [3.0, 2.0]
