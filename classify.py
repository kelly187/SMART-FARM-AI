# Classify an image using the trained model 'mobilenet_model.pth'
#Step 1, load numpy libararies
import torch, numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

device = torch.device('cpu')
# Initialize the CNN model, loss function, and optimizer
model = torch.load('mobilenet_model.pth', map_location=device)#, weights_only=False)
model.eval()

transform = transforms.Compose([
transforms.Resize((256, 256)),
transforms.ToTensor(),

#transforms.Lambda(lambda x:x[:3,:,:])

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



#point to your labels file
with open('labels.txt', 'r') as f:
    class_map = [line.strip() for line in f]
#print(len(class_map))

#if isinstance(class_map, np.ndarray):
#    class_map = class_map.tolist()
#print(type(class_map))    
#print(class_map[19])

#unique_labels = np.unique(labels)

#point to your image to be classified
path = 'TEST/caney.jpeg'
img = Image.open(path)
img = img.convert('RGB')
img = transform(img)
img = img.unsqueeze(0)

with torch.no_grad():
    
    output = model(img)
    predict = F.softmax(output, dim=1)[0] #.item()
    #prediction = prediction.indices
    prediction = torch.argmax(predict).item()
    #prediction = prediction
    #confidence = confidence
    predicted_label = class_map[prediction]
    confidence = predict[prediction].item()
    
    
    print(f'Predicted Class Index: {prediction}')
    print(f'Predicted Label: {predicted_label}')
    print(f'Confidence Score: {confidence:.4f} ({confidence * 100:.2f}%)')
    
