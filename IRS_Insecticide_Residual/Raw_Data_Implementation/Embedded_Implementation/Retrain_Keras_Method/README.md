# Retrain Keras Method

This method trains the final deployable model directly in Keras, exports TFLite
variants, validates the saved TFLite files with the desktop interpreter, and
saves one audit/deployment artifact bundle.

Despite the model name, `Classify_Raw_Data.py` currently builds **3 input
channels**:

```python
selected_value_types = ["original", "first_diff_filtered", "second_diff_filtered"]
```

With `signal_segments = ((0, 1000), (1800, 3500))` and `downsample step=5`, the
expected deployed input is:

```text
PyTorch-style layout: [batch, 3, 540]
Keras layout:         [batch, 540, 3]
```

## Layout

Method-specific files:

```text
Retrain_Keras_Method/
  final_model_training.py
  Functions/
    config.py
    training_utils.py
    final_artifacts.py
```

Shared files used by all embedded-generation methods:

```text
Embedded_Implementation/Shared_Functions/
  config.py
  data_pipeline.py
  keras_model.py
  export_tflite.py
  tflite_utils.py
  calibration.py
  metrics.py
  deployment_artifacts.py
```

Other method-specific paths:

```text
LiteRT_Torch_Method/
Pytorch_to_Keras_Method/
```

## Train Final Keras Model

Edit the settings block near the top of `final_model_training.py`, then run the
file. This method intentionally follows the same style as
`Classify_Raw_Data.py`.

Main settings:

```python
EXCEL_PATH = "/home/shibojing/data/Practice/Stage3a_all_mixed.xlsx"

# Run CV to choose final epochs.
FINAL_EPOCHS = None
RUN_CV_FOR_EPOCH_SELECTION = True

# Or use a fixed final epoch count.
FINAL_EPOCHS = 50
RUN_CV_FOR_EPOCH_SELECTION = False

# Export TFLite variants directly after Keras training.
TFLITE_VARIANTS = ("float", "dynamic_wi8_afp32", "full_int8")
SKIP_TFLITE_EXPORT = False
SKIP_TFLITE_VALIDATION = False
```

Run this file directly in PyCharm, or run it from the project root:

```bash
python -m IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Retrain_Keras_Method.final_model_training
```

Default output folder:

```text
Retrain_Keras_Method/Results
```

Important outputs:

```text
shared_backbone_final.keras
shared_backbone_float.tflite                # only when SKIP_TFLITE_EXPORT = False
shared_backbone_dynamic_wi8_afp32.tflite    # only when SKIP_TFLITE_EXPORT = False
shared_backbone_full_int8.tflite            # only when SKIP_TFLITE_EXPORT = False
representative_final.npy
representative_indices_final.npy
scalers_final.npz
deployment_metadata_final.json
test_predictions_final.npz
```

The final Keras training step uses all trainval data. The holdout test split is
kept separate and is used only for final evaluation and saved TFLite validation.

Keras CV epoch selection uses the second-largest fold best epoch, matching the
final-epoch selection policy used by the LiteRT Torch path.

By default, the Keras model uses PyTorch-compatible flatten order for structural
parity with the original model.

During Keras CV epoch selection, the script uses `ReduceLROnPlateau` by default
to mirror the tuned PyTorch training loop more closely. Disable it only for a
controlled comparison:

```python
USE_LR_SCHEDULER = False
```

## Export TFLite

The TFLite exporter is shared:

```bash
python -m IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.export_tflite \
  --keras-model IRS_Insecticide_Residual/Raw_Data_Implementation/Embedded_Implementation/Retrain_Keras_Method/Results/shared_backbone_final.keras \
  --output IRS_Insecticide_Residual/Raw_Data_Implementation/Embedded_Implementation/Retrain_Keras_Method/Results/shared_backbone_full_int8.tflite \
  --representative-npy IRS_Insecticide_Residual/Raw_Data_Implementation/Embedded_Implementation/Retrain_Keras_Method/Results/representative_final.npy
```

For a debug float model:

```bash
python -m IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.export_tflite \
  --keras-model IRS_Insecticide_Residual/Raw_Data_Implementation/Embedded_Implementation/Retrain_Keras_Method/Results/shared_backbone_final.keras \
  --output IRS_Insecticide_Residual/Raw_Data_Implementation/Embedded_Implementation/Retrain_Keras_Method/Results/shared_backbone_float.tflite \
  --float
```

For a dynamic-range model with quantized weights and float input/output:

```bash
python -m IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.export_tflite \
  --keras-model IRS_Insecticide_Residual/Raw_Data_Implementation/Embedded_Implementation/Retrain_Keras_Method/Results/shared_backbone_final.keras \
  --output IRS_Insecticide_Residual/Raw_Data_Implementation/Embedded_Implementation/Retrain_Keras_Method/Results/shared_backbone_dynamic_wi8_afp32.tflite \
  --dynamic-range
```

## Optional PyTorch Parity Tools

CV-fold artifact saving is shared:

```python
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.deployment_artifacts import (
    save_deployment_artifacts,
)
```

PyTorch-to-Keras weight transfer is method-specific:

```bash
python -m IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Pytorch_to_Keras_Method.Functions.transfer_pytorch_weights \
  --pytorch-weights IRS_Insecticide_Residual/Raw_Data_Implementation/Embedded_Implementation/Pytorch_to_Keras_Method/Results/shared_backbone_final.pth \
  --output IRS_Insecticide_Residual/Raw_Data_Implementation/Embedded_Implementation/Pytorch_to_Keras_Method/Results/shared_backbone.keras \
  --metadata-output IRS_Insecticide_Residual/Raw_Data_Implementation/Embedded_Implementation/Pytorch_to_Keras_Method/Results/shared_backbone_metadata.json \
  --input-length 540 \
  --in-channels 3 \
  --num-classes 3
```

## ESP32-S3 Notes

The model outputs logits, not softmax probabilities. For classification on the
device, `argmax(logits)` is enough.

The microcontroller preprocessing must match the Python preprocessing and the
per-channel `StandardScaler` arrays exactly. For 3 channels and 540 points:

```text
mean[3][540]
scale[3][540]
```
