#Note, you should take advantage of Google Colab's free Gpu as training an AI Model is quite CPU intensive
#also upload your images to train to PLANT_DATAS FOLDER and create respective subfolders for each class of images

# Step 1: Mount Google Drive, this should be run once and will throw error if re-run
from google.colab import drive
drive.mount('/content/drive')

# Step 2a: Import Sysytem libraries
import os

#import Pytorch and its associated sub modules
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# Step 3: Set dataset path
data_dir = "/content/drive/My Drive/Colab Notebooks/PLANT_DATAS"  # Replace with your dataset folder path

# Step 4: Define transformations and load dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for MobileNet
    transforms.ToTensor(),         # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize for MobileNet
])
print('load images')
dataset = datasets.ImageFolder(data_dir, transform=transform)
print('images loaded')
print('arrange env')
# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# DataLoaders for training and validation
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# Step 5: Load MobileNetV2 and modify it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, len(dataset.classes))  # Adjust for number of classes
model = model.to(device)

# Step 6: Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print('Training started')
# Step 7: Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Print batch progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
              f"Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%")

    # Save model after each epoch
    torch.save(model.state_dict(), f"/content/drive/My Drive/Colab Notebooks/mobilenet_epoch_{epoch+1}.pth")
    print(f"Epoch {epoch+1} complete. Model saved.")

# Step 8: Save the final model
torch.save(model.state_dict(), "/content/drive/My Drive/Colab Notebooks/mobilenet_weights.pth")  # Save weights only
torch.save(model, "/content/drive/My Drive/Colab Notebooks/mobilenet_model.pth")                 # Save entire model
print("Training complete. Final model saved.")
