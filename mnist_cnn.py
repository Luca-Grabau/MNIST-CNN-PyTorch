# PyTorch related libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
from torchvision import datasets, transforms

# Data manipulation libraries 
import os
import numpy as np

# Metrics libraries
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Define transform to normalize data
transform = transforms.Compose([transforms.ToTensor()])

# Define transform to normalize and augment training data
train_transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.ToTensor(),
])

# Download and load training data with augmented transformations
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
trainloader = DataLoader(train_data, batch_size=64, shuffle=True)

# Download and load test data with basic transformation
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Define the Network Architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # 1 input channel, 16 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 500)
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(500)
        
        self.fc2 = nn.Linear(500, 10)
        
        # Dropout layer to reduce overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional layers with ReLU activation
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the image
        x = x.view(-1, 32 * 7 * 7)
        
        # Fully connected layer with ReLU and batch normalization
        x = F.relu(self.bn1(self.fc1(x)))
        
        # Apply Dropout
        x = self.dropout(x)
        
        # Final output layer
        x = self.fc2(x)
        return x

# Instantiate the Network and Move to GPU
model = Net().to('cuda')

# Load pre-trained weights if available
pretrained_path =  '' # Provide the correct path to your saved checkpoint
if os.path.exists(pretrained_path):
    model.load_state_dict(torch.load(pretrained_path))
    print("Loaded pre-trained model weights.")
else:
    print("No pre-trained weights found. Starting from scratch.")

# Initialize TensorBoard writer
tensorboard_writer = SummaryWriter()

# Specify loss function and optimizer with weight decay
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)

# Create a learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

# record loss during training for visualizing later
loss_over_time = []

# Training the network
for epoch in range(10):
    running_loss = 0.0
    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        loss.backward()
        dot = make_dot(loss, params=dict(model.named_parameters()))
        dot.format = 'png'
        dot.render(filename='graph')  # this saves the graph in a file named "graph.png"
        optimizer.step()

        running_loss += loss.item()

        # Log loss for each iteration
        iteration = epoch * len(trainloader) + i
        tensorboard_writer.add_scalar('Loss', loss.item(), iteration)

        if i % 100 == 99:
            print('[Epoch %d, Batch %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            loss_over_time.append(running_loss)
            running_loss = 0.0

    # Calculate average loss for the epoch
    avg_loss = running_loss / len(trainloader)

    # Log average loss to TensorBoard
    tensorboard_writer.add_scalar('Epoch Loss', avg_loss, epoch)

    # Adjust learning rate using the scheduler
    scheduler.step(running_loss)  # Pass the loss value to the scheduler

    # Save model checkpoint after each epoch
    torch.save(model.state_dict(), f'model_checkpoint_epoch_{epoch + 1}.pt')

# Close TensorBoard writer
tensorboard_writer.close()

plt.figure()
plt.plot(loss_over_time)
plt.title('Loss over time')
plt.xlabel('100 Batches')
plt.ylabel('Loss')
plt.show()

# Visualizing Convolutional layer weights
weights = model.conv1.weight.data.cpu()
w = weights.numpy()

columns = 4
rows = 4
fig = plt.figure(figsize=(8,8))

for i in range(0, columns*rows):
    fig.add_subplot(rows, columns, i+1)
    img = w[i][0]
    plt.imshow(img, cmap='gray')
plt.show()

# Move model to evaluation mode
model.eval()

correct = 0
total = 0

# Gather all predictions and true labels
all_predictions = []
all_labels = []

# No need to track grads here
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())          # Append true labels
        all_predictions.extend(predicted.cpu().numpy())  # Append predicted labels


print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

# Calculate and print additional metrics
precision = precision_score(all_labels, all_predictions, average='macro')
recall = recall_score(all_labels, all_predictions, average='macro')
f1 = f1_score(all_labels, all_predictions, average='macro')

print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)

# Print confusion matrix
confusion_mat = confusion_matrix(all_labels, all_predictions)
print('Confusion Matrix:')
print(confusion_mat)

# Print classification report
class_names = [str(i) for i in range(10)]  # Assuming 10 classes
classification_rep = classification_report(all_labels, all_predictions, target_names=class_names)
print('Classification Report:')
print(classification_rep)

# Calculate and print per-class accuracy
per_class_accuracy = confusion_mat.diagonal() / confusion_mat.sum(axis=1)
for class_idx, acc in enumerate(per_class_accuracy):
    print(f'Class {class_idx} Accuracy: {acc:.2f}')

# prediction
outputs = model(images[0].unsqueeze(0))
_, prediction = torch.max(outputs.data, 1)
print("Prediction: ", prediction.item())
  
# visualize first test image and model prediction
image = images[0].cpu().numpy().squeeze()
plt.figure()
plt.imshow(image, 'gray')
plt.plot(loss_over_time)
plt.title('Loss over time')
plt.xlabel('100 Batches')
plt.ylabel('Loss')
plt.show() # This line ensures that the plot is displayed, wasn't needed initially but began to not display after further code development