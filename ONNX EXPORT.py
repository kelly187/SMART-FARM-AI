#export your model to cross platform onnx file to run anywhere
import torch, torch.onnx, onnx

print('Initialising and running')
device = torch.device('cpu')

model = torch.jit.load('mobilenet_model.pth', map_location=device)
model.eval()


dummy_input = torch.randn(1,3,224,224)
#batch size=1, channels=3, height and width =224
file = 'model.onnx'
torch.onnx.export(model,dummy_input,file,input_names=['input'], output_names=['output'], opset_version=11)
ox = onnx.load('model.onnx')
onnx.checker.check_model(ox)
print('ALL GOOD')
