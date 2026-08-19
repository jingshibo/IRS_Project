from __future__ import annotations

import numpy as np


def predict_keras_logits_prob(
    model: object,
    x_pytorch_layout: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import tensorflow as tf

    from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.keras_model import (
        torch_to_keras_input,
    )

    keras_logits = model.predict(
        torch_to_keras_input(x_pytorch_layout), # Convert PyTorch tensor layout to Keras tensor layout
        batch_size=batch_size,
        verbose=0,
    ).astype(np.float32, copy=False)
    keras_prob = tf.nn.softmax(keras_logits, axis=1).numpy().astype(np.float32, copy=False)
    return keras_logits, keras_prob


__all__ = [
    "predict_keras_logits_prob",
]
