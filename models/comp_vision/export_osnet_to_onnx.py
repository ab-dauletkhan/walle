import torch, torch.nn as nn
import torchreid

ckpt_path = "models/osnet_x1_0_msmt17.pth"
onnx_path = "models/reid-2.onnx"

# build OSNet x0.25 backbone
model = torchreid.models.build_model(name='osnet_x0_25', num_classes=1000, pretrained=False)
torchreid.utils.load_pretrained_weights(model, ckpt_path)

# remove classification head if present
if hasattr(model, 'classifier'):
    model.classifier = nn.Identity()

model.eval()

dummy = torch.randn(1, 3, 256, 128)  # NCHW (RGB), standard ReID size
torch.onnx.export(
    model, dummy, onnx_path,
    input_names=["input"], output_names=["feat"],
    dynamic_axes={"input": {0: "N"}, "feat": {0: "N"}},
    opset_version=12
)
print("Exported to", onnx_path)
