# LiteRT Torch Method

This method trains the final model in PyTorch, exports the trained PyTorch model
directly to `.tflite` through LiteRT Torch, then validates PyTorch-vs-LiteRT
logits.

LiteRT Torch currently uses the PyTorch model input layout directly. For this
project that means:

```text
PyTorch/LiteRT input layout: [batch, 3, 540]
```

Despite the model name, `Classify_Raw_Data.py` currently builds **3 input
channels**:

```python
selected_value_types = ["original", "first_diff_filtered", "second_diff_filtered"]
```

## Layout

Method-specific files:

```text
LiteRT_Torch_Method/
  final_model_training.py
  Functions/
    config.py
    training_utils.py
    final_artifacts.py
    litert_export.py
```

Shared files used by all embedded-generation methods:

```text
Embedded_Implementation/Functions/
  config.py
  data_pipeline.py
  keras_model.py
  export_tflite.py
  deployment_artifacts.py
```

## Requirements

The direct converter follows the current LiteRT Torch API:

```python
edge_model = litert_torch.convert(model.eval(), sample_inputs)
edge_model.export("model.tflite")
```

Install the current packages in the environment where you run conversion:

```bash
pip install litert-torch ai-edge-litert
```

If the stable package has a conversion issue, try the nightly:

```bash
pip install --pre litert-torch-nightly
```

## Train Final PyTorch Model and Export LiteRT

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
```

Run this file directly in PyCharm, or run it from the project root:

```bash
python -m IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.LiteRT_Torch_Method.final_model_training
```

Default output folder:

```text
IRS_Insecticide_Residual/Raw_Data_Implementation/Embedded_Implementation/LiteRT_Torch_Method/artifacts/final_model
```

Important outputs:

```text
shared_backbone_final.pth
shared_backbone_litert_float.tflite
representative_final.npy
scalers_final.npz
deployment_metadata_final.json
test_predictions_final.npz
```

The final PyTorch training step uses all trainval data. The holdout test split
is still kept separate and is used only for final evaluation and
PyTorch-vs-LiteRT parity checking.

By default, the final PyTorch training step does **not** use
`ReduceLROnPlateau`, because there is no inner validation split to monitor.
PyTorch CV epoch selection still uses the tuned scheduler behavior. If you want
the final full-train run to reduce LR based on training loss, set:

```python
FINAL_USE_TRAIN_LOSS_SCHEDULER = True
```

## Optional Quantization

The method can apply a no-calibration AI Edge Quantizer recipe after the float
LiteRT export:

```python
QUANTIZE_RECIPE = "dynamic_wi8_afp32"
```

For ESP32-S3, calibrated full-int8 W8A8 is usually the target. This script does
not hide that behind a guessed API because AI Edge Quantizer calibration APIs
are version-sensitive. First confirm the float LiteRT export and parity, then
use a calibrated `static_wi8_ai8` quantization flow with `representative_final.npy`.

## Validation

The script checks:

```text
PyTorch logits vs LiteRT Torch edge_model sample logits
PyTorch logits vs desktop LiteRT/TFLite interpreter logits
holdout accuracy from PyTorch
holdout accuracy from LiteRT/TFLite, when an interpreter is installed
```

Use `--skip-tflite-validation` only when no desktop interpreter is installed.
