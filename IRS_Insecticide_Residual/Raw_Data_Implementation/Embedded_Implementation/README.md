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
