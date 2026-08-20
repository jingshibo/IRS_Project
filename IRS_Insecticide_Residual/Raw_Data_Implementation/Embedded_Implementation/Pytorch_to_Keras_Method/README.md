# PyTorch to Keras Method

This method trains the final model in PyTorch, transfers the learned weights
into the shared Keras architecture, verifies PyTorch-vs-Keras logits, then
exports TFLite.

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
Pytorch_to_Keras_Method/
  final_model_training.py
  Functions/
    config.py
    training_utils.py
    final_artifacts.py
    transfer_pytorch_weights.py
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

## Train Final PyTorch Model and Transfer to Keras

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

# Export TFLite variants directly after Keras transfer.
TFLITE_VARIANTS = ("float", "dynamic_wi8_afp32", "full_int8")
SKIP_TFLITE_EXPORT = False
SKIP_TFLITE_VALIDATION = False
```

Run this file directly in PyCharm, or run it from the project root:

```bash
python -m IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Pytorch_to_Keras_Method.final_model_training
```

Default output folder:

```text
Pytorch_to_Keras_Method/Results
```

The default path is computed from the location of `Functions/config.py`, so it
does not depend on the current working directory used by PyCharm.

Important outputs:

```text
shared_backbone_final.pth
shared_backbone_final.keras
shared_backbone_float.tflite                # only when SKIP_TFLITE_EXPORT = False
shared_backbone_dynamic_wi8_afp32.tflite    # only when SKIP_TFLITE_EXPORT = False
shared_backbone_full_int8.tflite            # only when SKIP_TFLITE_EXPORT = False
representative_final.npy
scalers_final.npz
deployment_metadata_final.json
test_predictions_final.npz
```

The final PyTorch training step uses all trainval data. The holdout test split
is still kept separate and is used only for final evaluation and
PyTorch-vs-Keras parity checking.

PyTorch CV epoch selection can use the configured validation-loss scheduler.
The final PyTorch training step then runs for the selected fixed epoch count
without a train-loss scheduler, because there is no inner validation split to
monitor.

## Transfer Existing PyTorch Weights Only

If you already have a trained `.pth` checkpoint and only want to copy weights to
Keras, use:

```bash
python -m IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Pytorch_to_Keras_Method.Shared_Functions.transfer_pytorch_weights \
  --pytorch-weights path/to/shared_backbone_final.pth \
  --output IRS_Insecticide_Residual/Raw_Data_Implementation/Embedded_Implementation/Pytorch_to_Keras_Method/artifacts/shared_backbone.keras \
  --metadata-output IRS_Insecticide_Residual/Raw_Data_Implementation/Embedded_Implementation/Pytorch_to_Keras_Method/artifacts/shared_backbone_metadata.json \
  --input-length 540 \
  --in-channels 3 \
  --num-classes 3
```

## Export TFLite Separately

The TFLite exporter is shared:

```bash
python -m IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.export_tflite \
  --keras-model IRS_Insecticide_Residual/Raw_Data_Implementation/Embedded_Implementation/Pytorch_to_Keras_Method/Results/shared_backbone_final.keras \
  --output IRS_Insecticide_Residual/Raw_Data_Implementation/Embedded_Implementation/Pytorch_to_Keras_Method/Results/shared_backbone_full_int8.tflite \
  --representative-npy IRS_Insecticide_Residual/Raw_Data_Implementation/Embedded_Implementation/Pytorch_to_Keras_Method/Results/representative_final.npy
```

For a debug float model:

```bash
python -m IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.export_tflite \
  --keras-model IRS_Insecticide_Residual/Raw_Data_Implementation/Embedded_Implementation/Pytorch_to_Keras_Method/Results/shared_backbone_final.keras \
  --output IRS_Insecticide_Residual/Raw_Data_Implementation/Embedded_Implementation/Pytorch_to_Keras_Method/Results/shared_backbone_float.tflite \
  --float
```

For a dynamic-range model with quantized weights and float input/output:

```bash
python -m IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.export_tflite \
  --keras-model IRS_Insecticide_Residual/Raw_Data_Implementation/Embedded_Implementation/Pytorch_to_Keras_Method/Results/shared_backbone_final.keras \
  --output IRS_Insecticide_Residual/Raw_Data_Implementation/Embedded_Implementation/Pytorch_to_Keras_Method/Results/shared_backbone_dynamic_wi8_afp32.tflite \
  --dynamic-range
```
