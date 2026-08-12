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
pip install litert-torch ai-edge-litert ai-edge-quantizer
```

`ai-edge-litert` is used for desktop validation of the exported `.tflite` file.
`ai-edge-quantizer` is used for optional post-export quantization. TensorFlow is
not required by this method.

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
LiteRT_Torch_Method/results
```

The default path is computed from the location of `Functions/config.py`, so it
does not depend on the current working directory used by PyCharm.

Important outputs:

```text
shared_backbone_final.pth
shared_backbone_litert_float.tflite
shared_backbone_litert_dynamic_wi8_afp32.tflite
shared_backbone_litert_static_wi8_ai8.tflite
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

The method applies both quantization recipes after the float LiteRT export so
the results can be compared side by side:

```python
QUANTIZE_RECIPES = ("dynamic_wi8_afp32", "static_wi8_ai8")
```

This writes:

```text
shared_backbone_litert_dynamic_wi8_afp32.tflite
shared_backbone_litert_static_wi8_ai8.tflite
```

`dynamic_wi8_afp32` uses int8 weights with float32 activations. `static_wi8_ai8`
uses calibrated full-int8 weights and activations. The calibrated path uses
`x_train_norm[:REPRESENTATIVE_COUNT]` as representative calibration data, reads
the LiteRT model signature/input name, runs `Quantizer.calibrate(...)`, then
writes the full-int8 `.tflite`.

## Validation

The script checks:

```text
PyTorch logits vs LiteRT Torch edge_model sample logits
holdout accuracy from PyTorch
holdout accuracy from float TFLite
holdout accuracy from dynamic weight-int8 TFLite
holdout accuracy from calibrated full-int8 TFLite
logit differences for each TFLite variant vs PyTorch
```

The full arrays are saved in `test_predictions_final.npz`. The saved variant
keys are:

```text
torch_logits
float_logits
half_quant_dynamic_wi8_afp32_logits
full_quant_static_wi8_ai8_logits
```

The summary is saved in `deployment_metadata_final.json` under
`comparison_summary`.

Set `SKIP_TFLITE_VALIDATION = True` only when no desktop interpreter is installed.
