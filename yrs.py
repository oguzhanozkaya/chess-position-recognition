"""Yelp review sentiment classification with scratch PyTorch models.

Run locally with `just run`. The pipeline reads `data/train.csv` and
`data/test.csv` on every run, builds a vocabulary in memory from the training
split, trains a TextCNN sentiment classifier, and writes evaluation artifacts
under `output/`.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

os.environ["MPLBACKEND"] = "Agg"
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
REPORTS_DIR = OUTPUT_DIR / "reports"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

SEED = 67
EPOCHS = 20
PATIENCE = 4
BATCH_SIZE = 16384
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
MIN_DELTA = 0.0005
DEVICE = "cuda"
VALIDATION_SIZE = 0.1
MAX_VOCAB_SIZE = 120_000
MIN_TOKEN_FREQUENCY = 2
MAX_SEQUENCE_LENGTH = 512
EMBEDDING_DIM = 512
FILTER_COUNT = 512
KERNEL_SIZES = (2, 3, 4, 5)
HIDDEN_SIZE = 512
DROPOUT = 0.42
WORD_DROPOUT = 0.04
LABEL_SMOOTHING = 0.02
DATALOADER_WORKERS = 4
MIXED_PRECISION = True
COMPILE_MODEL = False
ROW_LIMIT = 0
CHECKPOINT_INTERVAL_EPOCHS = 1

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1
LABELS = ("negative", "positive")
RAW_LABEL_TO_ID = {1: 0, 2: 1}
ID_TO_LABEL = {index: label for index, label in enumerate(LABELS)}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[!?.,;:()\-]")


@dataclass(frozen=True)
class Config:
    seed: int = SEED
    epochs: int = EPOCHS
    patience: int = PATIENCE
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    min_delta: float = MIN_DELTA
    device: str = DEVICE
    validation_size: float = VALIDATION_SIZE
    max_vocab_size: int = MAX_VOCAB_SIZE
    min_token_frequency: int = MIN_TOKEN_FREQUENCY
    max_sequence_length: int = MAX_SEQUENCE_LENGTH
    embedding_dim: int = EMBEDDING_DIM
    filter_count: int = FILTER_COUNT
    kernel_sizes: tuple[int, ...] = KERNEL_SIZES
    hidden_size: int = HIDDEN_SIZE
    dropout: float = DROPOUT
    word_dropout: float = WORD_DROPOUT
    label_smoothing: float = LABEL_SMOOTHING
    dataloader_workers: int = DATALOADER_WORKERS
    mixed_precision: bool = MIXED_PRECISION
    compile_model: bool = COMPILE_MODEL
    row_limit: int = ROW_LIMIT
    checkpoint_interval_epochs: int = CHECKPOINT_INTERVAL_EPOCHS


def ensure_dirs() -> None:
    for path in (FIGURES_DIR, MODELS_DIR, PREDICTIONS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("device: cuda requested but unavailable, using cpu")
        return torch.device("cpu")
    return torch.device(name)


def validate_data_files() -> None:
    missing = [str(path) for path in (TRAIN_CSV, TEST_CSV) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Yelp CSV file(s): {', '.join(missing)}")


def read_yelp_csv(path: Path, config: Config) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        header=None,
        names=["raw_label", "text"],
        usecols=[0, 1],
        nrows=config.row_limit or None,
    )
    frame = frame.dropna(subset=["raw_label", "text"]).copy()
    frame["target"] = frame["raw_label"].astype(int).map(RAW_LABEL_TO_ID)
    frame = frame.dropna(subset=["target"]).copy()
    frame["target"] = frame["target"].astype(np.int64)
    frame["text"] = frame["text"].astype(str)
    return frame.reset_index(drop=True)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_vocabulary(texts: pd.Series, config: Config) -> tuple[dict[str, int], list[str]]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))
    tokens = [
        token for token, count in counter.most_common(config.max_vocab_size - 2) if count >= config.min_token_frequency
    ]
    id_to_token = [PAD_TOKEN, UNK_TOKEN, *tokens]
    token_to_id = {token: index for index, token in enumerate(id_to_token)}
    return token_to_id, id_to_token


def encode_texts(texts: pd.Series, token_to_id: dict[str, int], config: Config) -> np.ndarray:
    encoded = np.full((len(texts), config.max_sequence_length), PAD_ID, dtype=np.int64)
    for row_index, text in enumerate(texts):
        token_ids = [token_to_id.get(token, UNK_ID) for token in tokenize(text)[: config.max_sequence_length]]
        if token_ids:
            encoded[row_index, : len(token_ids)] = token_ids
    return encoded


class ReviewDataset(Dataset):
    def __init__(self, token_ids: np.ndarray, targets: np.ndarray, word_dropout: float = 0.0) -> None:
        self.token_ids = token_ids
        self.targets = targets
        self.word_dropout = word_dropout

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.as_tensor(self.token_ids[index], dtype=torch.long)
        if self.word_dropout > 0:
            keep_pad = tokens.eq(PAD_ID)
            mask = torch.rand(tokens.shape) < self.word_dropout
            tokens = tokens.masked_fill(mask & ~keep_pad, UNK_ID)
        target = torch.as_tensor(self.targets[index], dtype=torch.long)
        return tokens, target


class TextCNN(nn.Module):
    """Kim-style TextCNN implemented directly with PyTorch modules."""

    def __init__(self, vocab_size: int, config: Config) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=PAD_ID)
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(config.embedding_dim, config.filter_count, kernel_size=kernel_size)
                for kernel_size in config.kernel_sizes
            ]
        )
        conv_output_size = config.filter_count * len(config.kernel_sizes)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(conv_output_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, len(LABELS)),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids).transpose(1, 2)
        pooled = [torch.relu(convolution(embedded)).amax(dim=2) for convolution in self.convolutions]
        return self.classifier(torch.cat(pooled, dim=1))


def make_loader(
    token_ids: np.ndarray,
    targets: np.ndarray,
    config: Config,
    shuffle: bool,
    word_dropout: float = 0.0,
) -> DataLoader:
    return DataLoader(
        ReviewDataset(token_ids, targets, word_dropout=word_dropout),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.dataloader_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.dataloader_workers > 0,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    split: str,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_rows = 0
    total_correct = 0
    for batch_index, (tokens, targets) in enumerate(loader, start=1):
        tokens = tokens.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(tokens)
                loss = criterion(logits, targets)
            if training:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
        batch_size = len(targets)
        batch_correct = int(logits.argmax(dim=1).eq(targets).sum().detach().cpu())
        batch_loss = float(loss.detach().cpu())
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += batch_correct
        total_rows += batch_size
        print(
            f"epoch {epoch:03d} {split} batch {batch_index:04d}/{len(loader):04d}: "
            f"batch_loss={batch_loss:.5f} batch_accuracy={batch_correct / batch_size:.4f} "
            f"running_loss={total_loss / total_rows:.5f} running_accuracy={total_correct / total_rows:.4f}"
        )
    return total_loss / max(total_rows, 1), total_correct / max(total_rows, 1)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    split: str,
    device: torch.device,
    use_amp: bool,
) -> pd.DataFrame:
    model.eval()
    probabilities = []
    targets = []
    for tokens, batch_targets in loader:
        tokens = tokens.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(tokens)
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        targets.append(batch_targets.numpy())
    probs = np.concatenate(probabilities, axis=0)
    y_true = np.concatenate(targets, axis=0)
    y_pred = probs.argmax(axis=1)
    frame = pd.DataFrame(
        {
            "split": split,
            "true_id": y_true,
            "predicted_id": y_pred,
            "true_label": [ID_TO_LABEL[int(value)] for value in y_true],
            "predicted_label": [ID_TO_LABEL[int(value)] for value in y_pred],
            "negative_probability": probs[:, 0],
            "positive_probability": probs[:, 1],
            "confidence": probs.max(axis=1),
        }
    )
    return frame


def save_epoch_checkpoint(
    model: nn.Module,
    epoch: int,
    config: Config,
    id_to_token: list[str],
    validation_loss: float,
) -> None:
    if config.checkpoint_interval_epochs <= 0 or epoch % config.checkpoint_interval_epochs != 0:
        return
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "config": asdict(config),
            "id_to_token": id_to_token,
            "labels": LABELS,
            "validation_loss": validation_loss,
        },
        MODELS_DIR / f"text_cnn_epoch_{epoch:03d}.pt",
    )


def classification_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["true_id"].to_numpy()
    y_pred = frame["predicted_id"].to_numpy()
    probabilities = frame[["negative_probability", "positive_probability"]].to_numpy()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "log_loss": float(log_loss(y_true, probabilities, labels=list(range(len(LABELS))))),
    }


def save_figure(path: Path) -> Path:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_training_history(history: pd.DataFrame) -> None:
    history.to_csv(REPORTS_DIR / "training_history.csv", index=False)
    lines = ["# Training History", "", history.to_markdown(index=False)]
    (REPORTS_DIR / "training_history.md").write_text("\n".join(lines), encoding="utf-8")
    plt.figure(figsize=(8, 4))
    plt.plot(history["epoch"], history["train_loss"], label="train")
    plt.plot(history["epoch"], history["validation_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    save_figure(FIGURES_DIR / "training_loss.png")


def evaluate(predictions: pd.DataFrame, history: pd.DataFrame, config: Config, vocabulary_size: int) -> None:
    predictions.to_csv(PREDICTIONS_DIR / "predictions.csv", index=False)
    predictions.to_parquet(PREDICTIONS_DIR / "predictions.parquet", index=False)
    metrics = pd.DataFrame(
        [
            {"model_name": "scratch_text_cnn", "split": split, **classification_metrics(group)}
            for split, group in predictions.groupby("split", sort=True)
        ]
    )
    metrics.to_csv(REPORTS_DIR / "metrics.csv", index=False)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics.to_dict(orient="records"), indent=2), encoding="utf-8")
    (REPORTS_DIR / "metrics.md").write_text("# Metrics\n\n" + metrics.to_markdown(index=False) + "\n", encoding="utf-8")

    test_predictions = predictions[predictions["split"] == "test"]
    precision, recall, f1, support = precision_recall_fscore_support(
        test_predictions["true_id"],
        test_predictions["predicted_id"],
        labels=list(range(len(LABELS))),
        zero_division=0,
    )
    class_report = pd.DataFrame(
        {
            "label": LABELS,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    class_report.to_csv(REPORTS_DIR / "class_report.csv", index=False)
    (REPORTS_DIR / "class_report.md").write_text(
        "# Test Class Report\n\n" + class_report.to_markdown(index=False) + "\n", encoding="utf-8"
    )

    summary = {
        "config": asdict(config),
        "vocabulary_size": vocabulary_size,
        "best_validation_loss": float(history["validation_loss"].min()),
        "epochs_ran": int(history["epoch"].max()),
    }
    (REPORTS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cm = confusion_matrix(
        test_predictions["true_id"], test_predictions["predicted_id"], labels=list(range(len(LABELS)))
    )
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Test Confusion Matrix")
    plt.xticks(range(len(LABELS)), LABELS)
    plt.yticks(range(len(LABELS)), LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            plt.text(col, row, str(cm[row, col]), ha="center", va="center", color="black")
    save_figure(FIGURES_DIR / "confusion_matrix.png")

    plt.figure(figsize=(6, 4))
    test_predictions["confidence"].hist(bins=30)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Rows")
    plt.title("Test Prediction Confidence")
    save_figure(FIGURES_DIR / "prediction_confidence.png")

    plt.figure(figsize=(6, 4))
    test_predictions["true_label"].value_counts().reindex(LABELS).plot(kind="bar")
    plt.xlabel("Label")
    plt.ylabel("Rows")
    plt.title("Test Class Distribution")
    save_figure(FIGURES_DIR / "class_distribution.png")


def train_model(config: Config) -> None:
    ensure_dirs()
    set_seed(config.seed)
    validate_data_files()
    train_frame = read_yelp_csv(TRAIN_CSV, config)
    test_frame = read_yelp_csv(TEST_CSV, config)
    train_part, validation_part = train_test_split(
        train_frame,
        test_size=config.validation_size,
        random_state=config.seed,
        stratify=train_frame["target"],
    )
    train_part = train_part.reset_index(drop=True)
    validation_part = validation_part.reset_index(drop=True)
    print(
        "data: "
        f"train={len(train_part):,} validation={len(validation_part):,} test={len(test_frame):,} "
        f"source={DATA_DIR}"
    )

    token_to_id, id_to_token = build_vocabulary(train_part["text"], config)
    print(f"vocab: size={len(id_to_token):,} max_sequence_length={config.max_sequence_length}")
    train_tokens = encode_texts(train_part["text"], token_to_id, config)
    validation_tokens = encode_texts(validation_part["text"], token_to_id, config)
    test_tokens = encode_texts(test_frame["text"], token_to_id, config)
    train_targets = train_part["target"].to_numpy(dtype=np.int64)
    validation_targets = validation_part["target"].to_numpy(dtype=np.int64)
    test_targets = test_frame["target"].to_numpy(dtype=np.int64)

    device = select_device(config.device)
    train_loader = make_loader(train_tokens, train_targets, config, shuffle=True, word_dropout=config.word_dropout)
    validation_loader = make_loader(validation_tokens, validation_targets, config, shuffle=False)
    test_loader = make_loader(test_tokens, test_targets, config, shuffle=False)
    model = TextCNN(len(id_to_token), config).to(device)
    if config.compile_model:
        model = torch.compile(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    use_amp = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    print(f"train: model=scratch_text_cnn device={device} amp={use_amp} batch_size={config.batch_size}")

    best_state = None
    best_validation_loss = float("inf")
    stale_epochs = 0
    history_rows = []
    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, epoch, "train", optimizer, scaler, use_amp
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, criterion, device, epoch, "validation", use_amp=use_amp
        )
        scheduler.step(validation_loss)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        save_epoch_checkpoint(model, epoch, config, id_to_token, validation_loss)
        if validation_loss < best_validation_loss - config.min_delta:
            best_validation_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            print(f"train: early stopping after epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history = pd.DataFrame(history_rows)
    write_training_history(history)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "id_to_token": id_to_token,
            "labels": LABELS,
        },
        MODELS_DIR / "scratch_text_cnn.pt",
    )

    evaluation_loader = make_loader(train_tokens, train_targets, config, shuffle=False)
    predictions = pd.concat(
        [
            predict(model, evaluation_loader, "train", device, use_amp),
            predict(model, validation_loader, "validation", device, use_amp),
            predict(model, test_loader, "test", device, use_amp),
        ],
        ignore_index=True,
    )
    evaluate(predictions, history, config, len(id_to_token))


def run_pipeline(config: Config) -> None:
    train_model(config)
    print("done: outputs written under output/")


if __name__ == "__main__":
    run_pipeline(Config())
