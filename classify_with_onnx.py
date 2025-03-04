import torch, onnxruntime as ort, numpy as np, torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

path ='model.onnx'
session = ort.InferenceSession(path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),

#transforms.Lambda(lambda x:x[:3,:,:])
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


with open('labels.txt', 'r') as f:
    class_map = [line.strip() for line in f]
#print(len(class_map))

#replace with path to image to classify
path = 'TEST/cow.jpg'
img = Image.open(path)
img = img.convert('RGB')
img = transform(img)
img = img.unsqueeze(0).numpy()

    
outputs = session.run([output_name], {input_name: img})
predict = np.array(outputs).squeeze()

#predict = outputs #F.softmax(outputs, dim=1)[0] #.item()
    #prediction = prediction.indices
prediction = F.softmax(torch.tensor(predict), dim=0).numpy()
predicted = np.argmax(prediction)
    #confidence = confidence
    #predicted_label = class_map[prediction]
confidence = prediction[predicted]
#replace with pahth to labels.txt
with open('labels.txt', 'r') as f:
    class_map = [line.strip() for line in f]    
label = class_map[predicted]    
    
print(f'Predicted Class Index: {predicted}')
print(f'Predicted Label: {label}')
print(f'Confidence Score: {confidence * 100:.2f}%')
    
