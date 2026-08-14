from __future__ import annotations

import random
from typing import Optional

import numpy as np
import tensorflow as tf

from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Retrain_Keras_Method.Functions.config import FinalTrainingConfig
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.data_pipeline import (
    encode_labels,
    fit_transform_channel_scalers,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.keras_model import (
    SharedBackboneConfig,
    build_shared_backbone_keras_model,
    torch_to_keras_input,
)
from IRS_Insecticide_Residual.Utility_Functions import Preprocessing


class KerasSignalSequence(tf.keras.utils.Sequence):
    """Batch generator with PyTorch-equivalent random temporal shift augmentation."""

    def __init__(
        self,
        x: np.ndarray,
        y_one_hot: np.ndarray,
        batch_size: int,
        random_shift_max_points: int = 0,
        random_shift_fill_mode: str = "zero",
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if random_shift_fill_mode not in {"zero", "edge", "wrap"}:
            raise ValueError(
                "random_shift_fill_mode must be one of {'zero', 'edge', 'wrap'}, "
                f"got {random_shift_fill_mode!r}"
            )
        self.x = np.asarray(x, dtype=np.float32)
        self.y_one_hot = np.asarray(y_one_hot, dtype=np.float32)
        self.batch_size = int(batch_size)
        self.random_shift_max_points = int(random_shift_max_points)
        self.random_shift_fill_mode = random_shift_fill_mode
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last) and len(self.x) > self.batch_size
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.x))
        self.on_epoch_end()

    def __len__(self) -> int:
        if self.drop_last:
            return max(len(self.indices) // self.batch_size, 1)
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, batch_idx: int) -> tuple[np.ndarray, np.ndarray]:
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, len(self.indices))
        batch_indices = self.indices[start:end]
        x_batch = self.x[batch_indices].copy()
        y_batch = self.y_one_hot[batch_indices]
        if self.random_shift_max_points > 0:
            x_batch = self._apply_random_shift(x_batch)
        return x_batch, y_batch

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def _apply_random_shift(self, x_batch: np.ndarray) -> np.ndarray:
        for sample_idx in range(x_batch.shape[0]):
            shift = int(
                self.rng.integers(
                    -self.random_shift_max_points,
                    self.random_shift_max_points + 1,
                )
            )
            if shift == 0:
                continue

            original = x_batch[sample_idx].copy()
            shifted = np.roll(original, shift=shift, axis=0)
            if self.random_shift_fill_mode == "wrap":
                x_batch[sample_idx] = shifted
                continue

            if shift > 0:
                if self.random_shift_fill_mode == "zero":
                    shifted[:shift, :] = 0.0
                else:
                    shifted[:shift, :] = original[:1, :]
            else:
                tail_width = -shift
                if self.random_shift_fill_mode == "zero":
                    shifted[shift:, :] = 0.0
                else:
                    shifted[shift:, :] = original[-1:, :]
            x_batch[sample_idx] = shifted
        return x_batch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def create_optimizer(config: FinalTrainingConfig) -> tf.keras.optimizers.Optimizer:
    adamw = getattr(tf.keras.optimizers, "AdamW", None)
    if adamw is not None:
        return adamw(
            learning_rate=config.lr,
            weight_decay=config.weight_decay,
            beta_1=config.optimizer_beta_1,
            beta_2=config.optimizer_beta_2,
            epsilon=config.optimizer_epsilon,
            amsgrad=config.optimizer_amsgrad,
        )
    print("tf.keras.optimizers.AdamW is unavailable; falling back to Adam without decoupled weight decay.")
    return tf.keras.optimizers.Adam(
        learning_rate=config.lr,
        beta_1=config.optimizer_beta_1,
        beta_2=config.optimizer_beta_2,
        epsilon=config.optimizer_epsilon,
        amsgrad=config.optimizer_amsgrad,
    )


def to_one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    return tf.keras.utils.to_categorical(y, num_classes=num_classes).astype(np.float32)


def build_compiled_model(input_length: int, in_channels: int, num_classes: int, config: FinalTrainingConfig) -> tf.keras.Model:
    model_config = SharedBackboneConfig(
        input_length=input_length,
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=0.1,
        leaky_relu_slope=0.05,
    )
    model = build_shared_backbone_keras_model(
        model_config,
        include_softmax=False,
        match_pytorch_flatten=config.match_pytorch_flatten,
    )
    model.compile(
        optimizer=create_optimizer(config),
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=config.label_smoothing,
        ),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")],
    )
    return model


def fit_keras_model(
    model: tf.keras.Model,
    x_train_keras: np.ndarray,
    y_train_one_hot: np.ndarray,
    config: FinalTrainingConfig,
    epochs: int,
    validation_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    callbacks: Optional[list[tf.keras.callbacks.Callback]] = None,
    seed_offset: int = 0,
) -> tf.keras.callbacks.History:
    train_sequence = KerasSignalSequence(
        x_train_keras,
        y_train_one_hot,
        batch_size=config.batch_size,
        random_shift_max_points=config.random_shift_max_points,
        random_shift_fill_mode=config.random_shift_fill_mode,
        shuffle=True,
        drop_last=len(x_train_keras) > config.batch_size,
        seed=config.random_seed + seed_offset,
    )
    return model.fit(
        train_sequence,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1,
    )


def choose_final_epochs(
    config: FinalTrainingConfig,
    x_trainval: np.ndarray,
    y_trainval_labels: np.ndarray,
) -> tuple[int, dict[str, object]]:
    """Choose fixed final-train epoch count from Keras CV, unless provided."""
    if config.final_epochs is not None:
        if config.final_epochs < 1:
            raise ValueError(f"final_epochs must be >= 1, got {config.final_epochs}")
        return int(config.final_epochs), {"source": "user", "final_epochs": int(config.final_epochs)}

    if not config.run_cv_for_epoch_selection:
        raise ValueError("Set final_epochs or enable run_cv_for_epoch_selection.")

    cv_indices = Preprocessing.build_stratified_cv_indices(
        y_trainval_labels,
        n_splits=config.cv_folds_for_epoch_selection,
        random_seed=config.random_seed,
    )
    best_epochs = []
    best_val_accs = []
    for fold_id, (train_idx, val_idx) in enumerate(cv_indices):
        print(f"Selecting epoch count with Keras CV fold {fold_id + 1}/{len(cv_indices)}")
        x_fold_train, x_fold_val, _ = fit_transform_channel_scalers(
            x_trainval[train_idx],
            x_trainval[val_idx],
            clip_max_value=config.clip_max_value,
        )
        y_fold_train, label_to_idx, _ = encode_labels(y_trainval_labels[train_idx], config.class_order)
        y_fold_val = np.asarray([label_to_idx[label] for label in y_trainval_labels[val_idx]], dtype=np.int64)
        y_fold_train_one_hot = to_one_hot(y_fold_train, len(label_to_idx))
        y_fold_val_one_hot = to_one_hot(y_fold_val, len(label_to_idx))

        model = build_compiled_model(
            input_length=x_fold_train.shape[2],
            in_channels=x_fold_train.shape[1],
            num_classes=len(label_to_idx),
            config=config,
        )
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_categorical_accuracy",
            mode="max",
            patience=config.patience,
            restore_best_weights=True,
        )
        callbacks: list[tf.keras.callbacks.Callback] = [early_stop]
        if config.use_lr_scheduler:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    mode="min",
                    factor=config.scheduler_factor,
                    patience=config.scheduler_patience,
                    min_lr=config.scheduler_min_lr,
                )
            )
        history = fit_keras_model(
            model,
            torch_to_keras_input(x_fold_train),
            y_fold_train_one_hot,
            config,
            epochs=config.max_cv_epochs,
            validation_data=(torch_to_keras_input(x_fold_val), y_fold_val_one_hot),
            callbacks=callbacks,
            seed_offset=fold_id,
        )
        val_acc = np.asarray(history.history["val_categorical_accuracy"], dtype=np.float32)
        best_epoch = int(np.argmax(val_acc) + 1)
        best_val_acc = float(val_acc.max())
        best_epochs.append(best_epoch)
        best_val_accs.append(best_val_acc)
        print(f"Fold {fold_id} best_epoch={best_epoch} best_val_acc={best_val_acc:.4f}")
        tf.keras.backend.clear_session()

    sorted_best_epochs = sorted(best_epochs)
    if len(sorted_best_epochs) < 2:
        final_epochs = max(int(sorted_best_epochs[0]), 1)
    else:
        final_epochs = max(int(sorted_best_epochs[-2]), 1)
    print(f"Keras CV best epochs: {best_epochs}")
    print(f"Using final_epochs={final_epochs} from second-largest Keras CV best epoch.")
    return final_epochs, {
        "source": "keras_cv_second_largest_best_epoch",
        "best_epochs": best_epochs,
        "sorted_best_epochs": sorted_best_epochs,
        "final_epochs": final_epochs,
        "mean_best_val_acc": float(np.mean(np.asarray(best_val_accs, dtype=np.float32))),
    }


def evaluate_model(
    model: tf.keras.Model,
    x_test_keras: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    logits = model.predict(x_test_keras, batch_size=batch_size, verbose=0).astype(np.float32, copy=False)
    probs = tf.nn.softmax(logits, axis=1).numpy()
    pred_idx = np.argmax(probs, axis=1).astype(np.int64)
    accuracy = float(np.mean(pred_idx == y_test))
    return accuracy, pred_idx, probs.astype(np.float32, copy=False), logits
