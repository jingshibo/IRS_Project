# Embedded Implementation

This folder separates embedded model generation into three candidate paths:

```text
Embedded_Implementation/
  Shared_Functions/          # shared helpers used by multiple methods
  Retrain_Keras_Method/       # train final model directly in Keras
  Pytorch_to_Keras_Method/    # train in PyTorch, transfer weights to Keras
  LiteRT_Torch_Method/        # export PyTorch model with LiteRT Torch
```

Shared helpers used by multiple methods:

```text
Shared_Functions/config.py              # shared preprocessing/model defaults
Shared_Functions/data_pipeline.py       # Excel loading, preprocessing, scaling, labels
Shared_Functions/keras_model.py         # Keras architecture for Keras/TFLite paths
Shared_Functions/export_tflite.py       # shared .keras -> .tflite converter
Shared_Functions/tflite_utils.py        # shared Keras-exported TFLite validation helpers
Shared_Functions/calibration.py         # representative calibration sample selection
Shared_Functions/metrics.py             # NumPy softmax and logit-parity helpers
Shared_Functions/deployment_artifacts.py # optional PyTorch CV-fold artifact saver
```

Each method folder keeps the runnable orchestration script at the method root
and method-specific helper code under that method's own `Functions/` folder.

## Method Selection Guidance

Based on the current comparison runs, `Pytorch_to_Keras_Method` is the preferred
deployment path. It has produced slightly better results than `LiteRT_Torch_Method`
and these two are clearly better than `Retrain_Keras_Method` in the tested setup.

Use the methods in this order unless new validation results show otherwise:

```text
1. Pytorch_to_Keras_Method
   Best current deployment choice. Train with the stronger PyTorch pipeline,
   transfer weights into the matching Keras architecture, then export TFLite.

2. LiteRT_Torch_Method
   Useful when direct PyTorch-to-LiteRT export is preferred, but current results
   are slightly behind the PyTorch-to-Keras path.

3. Retrain_Keras_Method
   Simplest Keras-native deployment path, but current results are clearly weaker
   than the PyTorch-trained methods.
```

Likely reasons for the current ranking:

- The PyTorch training pipeline is the most established path in this project.
  Its optimizer settings, augmentation flow, epoch selection, and model
  implementation have been tuned around this dataset, so it is the strongest
  training baseline.
- `Pytorch_to_Keras_Method` keeps that PyTorch model as the source of truth. It
  transfers the trained weights into the matching Keras architecture, verifies
  PyTorch-vs-Keras logit parity, and then uses the standard Keras/TFLite export
  path. This limits the problem to architecture matching and weight transfer
  instead of retraining/optimizing the model in a second framework.
- `LiteRT_Torch_Method` avoids the Keras transfer step, but it relies on the
  LiteRT Torch conversion path and AI Edge Quantizer. That path is newer and may
  still differ from the original PyTorch execution because of supported-op
  coverage, serialization details, calibration behavior, or quantization
  handling. Those differences can make the saved TFLite model slightly less
  faithful to the PyTorch model.
- `Retrain_Keras_Method` trains a fresh Keras model instead of reusing the
  already strong PyTorch weights. Even with matching architecture and
  preprocessing, results can shift because of optimizer implementation,
  augmentation order, random initialization, layer behavior, and training-loop
  details. This gives it a larger gap to close than `Pytorch_to_Keras_Method`,
  which only has to preserve the trained PyTorch model through conversion. 

Treat this as an empirical selection rule, not a permanent guarantee. Recheck the
three methods after changing data, preprocessing, model architecture, training
settings, TensorFlow/LiteRT versions, or quantization recipes.

