# Learning Notes

## 1. Sequence-aware validation split

I noticed that GTSRB training images contain frame sequences. Images from the
same sequence are very similar. A random image split can place nearby frames in
both the training set and the validation set. This makes the validation result
too optimistic.

I changed the validation loader to group samples by `(class_id, sequence_id)`.
Each sequence now belongs to only one split. This gives a more realistic
validation result.

## 2. Post-Training Quantization (PTQ) & QONNX Export

I integrated Brevitas to run configurable PTQ and QAT experiments on the trained
VGG model for FPGA deployment.

* **Calibration Setup**: Used Brevitas calibration mode with `model.eval()` to
  collect activation statistics while keeping batch normalization and dropout
  deterministic.
* **Scale Propagation**: Used `return_quant_tensor=True` on the `QuantReLU` layers directly preceding `QuantLinear` layers to successfully pass input scales for 32-bit bias quantization.
* **QONNX Export**: Used `brevitas.export.export_qonnx()` on the CPU to avoid
  device conflicts during export.
