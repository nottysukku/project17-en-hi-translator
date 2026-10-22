"""Tiny one-conv models to find what the Neural-ART compiler runs in hardware."""
import os
import sys

import numpy as np
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

out = sys.argv[1]
os.makedirs(out, exist_ok=True)


class R(CalibrationDataReader):
    def __init__(self, shape):
        self.d = iter([{"x": np.random.randn(*shape).astype(np.float32)} for _ in range(4)])

    def get_next(self):
        return next(self.d, None)


for c, w, k, d in [(128, 4096, 3, 1), (128, 2048, 3, 1), (128, 1024, 3, 1), (64, 4096, 3, 1), (32, 4096, 7, 1),
                   (128, 1024, 11, 1), (128, 1024, 7, 3)]:
    m = torch.nn.Sequential(torch.nn.Conv2d(c, c, (1, k), padding=(0, d * (k - 1) // 2), dilation=(1, d)),
                            torch.nn.LeakyReLU(0.1)).eval()
    name = "c%d_w%d_k%d_d%d" % (c, w, k, d)
    f = os.path.join(out, name + ".onnx")
    torch.onnx.export(m, torch.randn(1, c, 1, w), f, input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
    quantize_static(f, f.replace(".onnx", "_i8.onnx"), R((1, c, 1, w)), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    print(name)
