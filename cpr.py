"""Chess board position recognition with a scratch PyTorch CNN.

Run locally with `just run`. The pipeline reads chessboard images from
`data/train/` and `data/test/` on every run, parses square labels from each
filename, trains one scratch convolutional model, and writes evaluation
artifacts under `output/`.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
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

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

SEED = 67
EPOCHS = 40
PATIENCE = 6
BATCH_SIZE = 128
LEARNING_RATE = 0.0008
WEIGHT_DECAY = 0.0001
MIN_DELTA = 0.0002
DEVICE = "cuda"
VALIDATION_SIZE = 0.1
IMAGE_SIZE = 256
DATALOADER_WORKERS = 4
MIXED_PRECISION = True
COMPILE_MODEL = False
ROW_LIMIT = 0
CHECKPOINT_INTERVAL_EPOCHS = 1
DROPOUT = 0.15
LABEL_SMOOTHING = 0.01
EMPTY_CLASS_WEIGHT = 0.45

BOARD_SIZE = 8
SQUARE_COUNT = BOARD_SIZE * BOARD_SIZE
PIECES = ("empty", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")
PIECE_TO_ID = {piece: index for index, piece in enumerate(PIECES)}
ID_TO_PIECE = {index: piece for index, piece in enumerate(PIECES)}
FILE_EXTENSIONS = ("*.jpeg", "*.jpg", "*.png")
CHANNEL_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
CHANNEL_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


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
    image_size: int = IMAGE_SIZE
    dataloader_workers: int = DATALOADER_WORKERS
    mixed_precision: bool = MIXED_PRECISION
    compile_model: bool = COMPILE_MODEL
    row_limit: int = ROW_LIMIT
    checkpoint_interval_epochs: int = CHECKPOINT_INTERVAL_EPOCHS
    dropout: float = DROPOUT
    label_smoothing: float = LABEL_SMOOTHING
    empty_class_weight: float = EMPTY_CLASS_WEIGHT


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


def validate_data_dirs() -> None:
    missing = [str(path) for path in (TRAIN_DIR, TEST_DIR) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing chess image directory/directories: {', '.join(missing)}")


def image_paths(directory: Path, config: Config) -> list[Path]:
    paths: list[Path] = []
    for extension in FILE_EXTENSIONS:
        paths.extend(directory.glob(extension))
    paths = sorted(paths)
    if config.row_limit > 0:
        return paths[: config.row_limit]
    return paths


def parse_fen_stem(stem: str) -> np.ndarray:
    ranks = stem.split("-")
    if len(ranks) != BOARD_SIZE:
        raise ValueError(f"Expected 8 FEN ranks in filename stem, got {len(ranks)}: {stem}")

    labels: list[int] = []
    for rank in ranks:
        rank_labels: list[int] = []
        for char in rank:
            if char.isdigit():
                rank_labels.extend([PIECE_TO_ID["empty"]] * int(char))
            elif char in PIECE_TO_ID:
                rank_labels.append(PIECE_TO_ID[char])
            else:
                raise ValueError(f"Invalid FEN character {char!r} in filename stem: {stem}")
        if len(rank_labels) != BOARD_SIZE:
            raise ValueError(f"Expected rank width 8, got {len(rank_labels)} in filename stem: {stem}")
        labels.extend(rank_labels)
    return np.asarray(labels, dtype=np.int64).reshape(BOARD_SIZE, BOARD_SIZE)


def build_frame(paths: list[Path], split: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": split,
            "image_path": [str(path) for path in paths],
            "image_name": [path.name for path in paths],
            "fen": [path.stem.replace("-", "/") for path in paths],
        }
    )


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - CHANNEL_MEAN) / CHANNEL_STD
    return torch.from_numpy(array.transpose(2, 0, 1))


def augment_image_and_target(image: Image.Image, target: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
        target = np.fliplr(target).copy()
    if random.random() < 0.5:
        image = ImageOps.flip(image)
        target = np.flipud(target).copy()
    if random.random() < 0.8:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.8:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.4:
        image = ImageEnhance.Color(image).enhance(random.uniform(0.9, 1.1))
    return image, target


class ChessPositionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, config: Config, augment: bool = False) -> None:
        self.paths = [Path(value) for value in frame["image_path"].tolist()]
        self.targets = [parse_fen_stem(path.stem) for path in self.paths]
        self.config = config
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[index]) as raw_image:
            image = raw_image.convert("RGB")
        target = self.targets[index]
        if self.augment:
            image, target = augment_image_and_target(image, target)
        return image_to_tensor(image, self.config.image_size), torch.as_tensor(target, dtype=torch.long)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(inputs) + self.shortcut(inputs))


class BoardCNN(nn.Module):
    """Scratch CNN that emits one piece-class distribution per board square."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 64, dropout=config.dropout / 2),
            ResidualBlock(64, 128, stride=2, dropout=config.dropout / 2),
            ResidualBlock(128, 128, dropout=config.dropout / 2),
            ResidualBlock(128, 192, stride=2, dropout=config.dropout),
            ResidualBlock(192, 192, dropout=config.dropout),
            ResidualBlock(192, 256, stride=2, dropout=config.dropout),
            ResidualBlock(256, 256, dropout=config.dropout),
            ResidualBlock(256, 384, stride=2, dropout=config.dropout),
            ResidualBlock(384, 384, dropout=config.dropout),
        )
        self.head = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(config.dropout),
            nn.Conv2d(256, len(PIECES), kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.features(images))
        if logits.shape[-2:] != (BOARD_SIZE, BOARD_SIZE):
            logits = nn.functional.adaptive_avg_pool2d(logits, (BOARD_SIZE, BOARD_SIZE))
        return logits


def make_loader(frame: pd.DataFrame, config: Config, shuffle: bool, augment: bool = False) -> DataLoader:
    return DataLoader(
        ChessPositionDataset(frame, config, augment=augment),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.dataloader_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.dataloader_workers > 0,
    )


def batch_metrics(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int, int, int, int, int]:
    predictions = logits.argmax(dim=1)
    correct = predictions.eq(targets)
    occupied = targets.ne(PIECE_TO_ID["empty"])
    board_correct = correct.flatten(1).all(dim=1)
    return (
        int(correct.sum().detach().cpu()),
        int(correct.numel()),
        int((correct & occupied).sum().detach().cpu()),
        int(occupied.sum().detach().cpu()),
        int(board_correct.sum().detach().cpu()),
        int(targets.shape[0]),
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
) -> tuple[float, float, float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_images = 0
    total_square_correct = 0
    total_squares = 0
    total_occupied_correct = 0
    total_occupied = 0
    total_board_correct = 0
    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
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
        square_correct, squares, occupied_correct, occupied, board_correct, images_seen = batch_metrics(logits, targets)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_images += images_seen
        total_square_correct += square_correct
        total_squares += squares
        total_occupied_correct += occupied_correct
        total_occupied += occupied
        total_board_correct += board_correct
        print(
            f"epoch {epoch:03d} {split} batch {batch_index:04d}/{len(loader):04d}: "
            f"batch_loss={float(loss.detach().cpu()):.5f} "
            f"square_accuracy={square_correct / max(squares, 1):.4f} "
            f"running_square_accuracy={total_square_correct / max(total_squares, 1):.4f}"
        )
    return (
        total_loss / max(total_images, 1),
        total_square_correct / max(total_squares, 1),
        total_occupied_correct / max(total_occupied, 1),
        total_board_correct / max(total_images, 1),
    )


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, split: str, device: torch.device, use_amp: bool) -> pd.DataFrame:
    model.eval()
    rows = []
    dataset = loader.dataset
    if not isinstance(dataset, ChessPositionDataset):
        raise TypeError("Expected ChessPositionDataset for prediction")
    offset = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            probabilities = torch.softmax(model(images), dim=1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        target_array = targets.numpy()
        for batch_row in range(target_array.shape[0]):
            path = dataset.paths[offset + batch_row]
            for rank in range(BOARD_SIZE):
                for file_index in range(BOARD_SIZE):
                    true_id = int(target_array[batch_row, rank, file_index])
                    predicted_id = int(predictions[batch_row, rank, file_index])
                    confidence = float(probabilities[batch_row, predicted_id, rank, file_index])
                    rows.append(
                        {
                            "split": split,
                            "image_name": path.name,
                            "fen": path.stem.replace("-", "/"),
                            "square": f"{chr(ord('a') + file_index)}{BOARD_SIZE - rank}",
                            "rank_index": rank,
                            "file_index": file_index,
                            "true_id": true_id,
                            "predicted_id": predicted_id,
                            "true_label": ID_TO_PIECE[true_id],
                            "predicted_label": ID_TO_PIECE[predicted_id],
                            "confidence": confidence,
                            "correct": true_id == predicted_id,
                            "occupied": true_id != PIECE_TO_ID["empty"],
                        }
                    )
        offset += target_array.shape[0]
    return pd.DataFrame(rows)


def save_epoch_checkpoint(
    model: nn.Module,
    epoch: int,
    config: Config,
    validation_loss: float,
    validation_square_accuracy: float,
) -> None:
    if config.checkpoint_interval_epochs <= 0 or epoch % config.checkpoint_interval_epochs != 0:
        return
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "config": asdict(config),
            "labels": PIECES,
            "validation_loss": validation_loss,
            "validation_square_accuracy": validation_square_accuracy,
        },
        MODELS_DIR / f"board_cnn_epoch_{epoch:03d}.pt",
    )


def save_figure(path: Path) -> Path:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_training_history(history: pd.DataFrame) -> None:
    history.to_csv(REPORTS_DIR / "training_history.csv", index=False)
    (REPORTS_DIR / "training_history.md").write_text(
        "# Training History\n\n" + history.to_markdown(index=False) + "\n", encoding="utf-8"
    )
    plt.figure(figsize=(8, 4))
    plt.plot(history["epoch"], history["train_loss"], label="train")
    plt.plot(history["epoch"], history["validation_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    save_figure(FIGURES_DIR / "training_loss.png")

    plt.figure(figsize=(8, 4))
    plt.plot(history["epoch"], history["train_square_accuracy"], label="train square")
    plt.plot(history["epoch"], history["validation_square_accuracy"], label="validation square")
    plt.plot(history["epoch"], history["validation_occupied_accuracy"], label="validation occupied")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    save_figure(FIGURES_DIR / "training_accuracy.png")


def split_metrics(group: pd.DataFrame) -> dict[str, float]:
    image_square_correct = group.groupby("image_name")["correct"].all()
    occupied = group[group["occupied"]]
    return {
        "square_accuracy": float(group["correct"].mean()),
        "occupied_square_accuracy": float(occupied["correct"].mean()) if len(occupied) else 0.0,
        "empty_square_accuracy": float(group.loc[~group["occupied"], "correct"].mean()),
        "board_accuracy": float(image_square_correct.mean()),
        "square_count": int(len(group)),
        "image_count": int(group["image_name"].nunique()),
    }


def write_reports(predictions: pd.DataFrame, history: pd.DataFrame, config: Config) -> None:
    predictions.to_csv(PREDICTIONS_DIR / "predictions.csv", index=False)
    predictions.to_parquet(PREDICTIONS_DIR / "predictions.parquet", index=False)

    metrics = pd.DataFrame(
        [
            {"model_name": "scratch_board_cnn", "split": split, **split_metrics(group)}
            for split, group in predictions.groupby("split")
        ]
    )
    metrics.to_csv(REPORTS_DIR / "metrics.csv", index=False)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics.to_dict(orient="records"), indent=2), encoding="utf-8")
    (REPORTS_DIR / "metrics.md").write_text("# Metrics\n\n" + metrics.to_markdown(index=False) + "\n", encoding="utf-8")

    test_predictions = predictions[predictions["split"] == "test"]
    precision, recall, f1, support = precision_recall_fscore_support(
        test_predictions["true_id"],
        test_predictions["predicted_id"],
        labels=list(range(len(PIECES))),
        zero_division=0,
    )
    class_report = pd.DataFrame(
        {"label": PIECES, "precision": precision, "recall": recall, "f1": f1, "support": support}
    )
    class_report.to_csv(REPORTS_DIR / "class_report.csv", index=False)
    (REPORTS_DIR / "class_report.md").write_text(
        "# Test Class Report\n\n" + class_report.to_markdown(index=False) + "\n", encoding="utf-8"
    )

    summary = {
        "config": asdict(config),
        "labels": PIECES,
        "best_validation_loss": float(history["validation_loss"].min()),
        "best_validation_square_accuracy": float(history["validation_square_accuracy"].max()),
        "epochs_ran": int(history["epoch"].max()),
    }
    (REPORTS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cm = confusion_matrix(
        test_predictions["true_id"], test_predictions["predicted_id"], labels=list(range(len(PIECES)))
    )
    plt.figure(figsize=(9, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title("Test Square Confusion Matrix")
    plt.xticks(range(len(PIECES)), PIECES, rotation=45)
    plt.yticks(range(len(PIECES)), PIECES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    save_figure(FIGURES_DIR / "confusion_matrix.png")

    plt.figure(figsize=(8, 4))
    test_predictions["confidence"].hist(bins=30)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Squares")
    plt.title("Test Square Prediction Confidence")
    save_figure(FIGURES_DIR / "prediction_confidence.png")

    plt.figure(figsize=(8, 4))
    test_predictions["true_label"].value_counts().reindex(PIECES).plot(kind="bar")
    plt.xlabel("Square Label")
    plt.ylabel("Squares")
    plt.title("Test Square Label Distribution")
    save_figure(FIGURES_DIR / "class_distribution.png")


def train_model(config: Config) -> None:
    ensure_dirs()
    set_seed(config.seed)
    validate_data_dirs()
    train_paths = image_paths(TRAIN_DIR, config)
    test_paths = image_paths(TEST_DIR, config)
    if not train_paths or not test_paths:
        raise FileNotFoundError("Expected image files under data/train/ and data/test/.")

    train_part_paths, validation_paths = train_test_split(
        train_paths,
        test_size=config.validation_size,
        random_state=config.seed,
        shuffle=True,
    )
    train_frame = build_frame(train_part_paths, "train")
    validation_frame = build_frame(validation_paths, "validation")
    test_frame = build_frame(test_paths, "test")
    print(
        "data: "
        f"train={len(train_frame):,} validation={len(validation_frame):,} test={len(test_frame):,} "
        f"source={DATA_DIR} image_size={config.image_size}"
    )

    device = select_device(config.device)
    train_loader = make_loader(train_frame, config, shuffle=True, augment=True)
    validation_loader = make_loader(validation_frame, config, shuffle=False)
    test_loader = make_loader(test_frame, config, shuffle=False)
    model = BoardCNN(config).to(device)
    if config.compile_model:
        model = torch.compile(model)

    class_weights = torch.ones(len(PIECES), device=device)
    class_weights[PIECE_TO_ID["empty"]] = config.empty_class_weight
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    use_amp = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    print(f"train: model=scratch_board_cnn device={device} amp={use_amp} batch_size={config.batch_size}")

    best_state = None
    best_validation_square_accuracy = 0.0
    stale_epochs = 0
    history_rows = []
    for epoch in range(1, config.epochs + 1):
        train_loss, train_square_accuracy, train_occupied_accuracy, train_board_accuracy = run_epoch(
            model, train_loader, criterion, device, epoch, "train", optimizer, scaler, use_amp
        )
        validation_loss, validation_square_accuracy, validation_occupied_accuracy, validation_board_accuracy = (
            run_epoch(model, validation_loader, criterion, device, epoch, "validation", use_amp=use_amp)
        )
        scheduler.step(validation_loss)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_square_accuracy": train_square_accuracy,
                "train_occupied_accuracy": train_occupied_accuracy,
                "train_board_accuracy": train_board_accuracy,
                "validation_loss": validation_loss,
                "validation_square_accuracy": validation_square_accuracy,
                "validation_occupied_accuracy": validation_occupied_accuracy,
                "validation_board_accuracy": validation_board_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        save_epoch_checkpoint(model, epoch, config, validation_loss, validation_square_accuracy)
        if validation_square_accuracy > best_validation_square_accuracy + config.min_delta:
            best_validation_square_accuracy = validation_square_accuracy
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
        {"model_state": model.state_dict(), "config": asdict(config), "labels": PIECES},
        MODELS_DIR / "scratch_board_cnn.pt",
    )

    evaluation_loader = make_loader(train_frame, config, shuffle=False)
    predictions = pd.concat(
        [
            predict(model, evaluation_loader, "train", device, use_amp),
            predict(model, validation_loader, "validation", device, use_amp),
            predict(model, test_loader, "test", device, use_amp),
        ],
        ignore_index=True,
    )
    write_reports(predictions, history, config)


def run_pipeline(config: Config) -> None:
    train_model(config)
    print("done: outputs written under output/")


if __name__ == "__main__":
    run_pipeline(Config())
