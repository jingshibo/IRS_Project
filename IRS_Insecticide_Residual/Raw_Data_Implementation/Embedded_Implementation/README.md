# Embedded Implementation

This folder separates embedded model generation into three candidate paths:

```text
Embedded_Implementation/
  Functions/                 # shared helpers used by multiple methods
  Retrain_Keras_Method/       # train final model directly in Keras
  Pytorch_to_Keras_Method/    # train in PyTorch, transfer weights to Keras
  LiteRT_Torch_Method/        # export PyTorch model with LiteRT Torch
```

Shared helpers:

```text
Functions/config.py              # shared preprocessing/model defaults
Functions/data_pipeline.py       # Excel loading, preprocessing, scaling, labels
Functions/keras_model.py         # Keras architecture for Keras/TFLite paths
Functions/export_tflite.py       # shared .keras -> .tflite converter
Functions/deployment_artifacts.py # optional PyTorch CV-fold artifact saver
```

Each method folder keeps the runnable orchestration script at the method root
and method-specific helper code under that method's own `Functions/` folder.
