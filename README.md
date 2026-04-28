# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1:
Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).

### STEP 2:
Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.

### STEP 3:
Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.

### STEP 4:
Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.

### STEP 5:
Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.

### STEP 6:
Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.

## PROGRAM

### Name:Thamizharasi G

### Register Number:212224100059

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size): 
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,4) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

        
# Initialize the Model, Loss Function, and Optimizer

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```

### Dataset Information
<img width="1219" height="195" alt="image" src="https://github.com/user-attachments/assets/b31d12e3-1cf6-4a15-91bd-3b33478f95a0" />

### OUTPUT
<img width="323" height="42" alt="image" src="https://github.com/user-attachments/assets/298c4031-0db3-4261-b48e-5a7ae8ce6fe9" />

## Confusion Matrix
<img width="675" height="569" alt="image" src="https://github.com/user-attachments/assets/37f79006-6336-4aa3-8f69-56847b333cc3" />

## Classification Report
<img width="557" height="434" alt="image" src="https://github.com/user-attachments/assets/4b02275d-843c-40fc-bbbe-d21e382571fe" />

### New Sample Data Prediction
<img width="361" height="102" alt="image" src="https://github.com/user-attachments/assets/008f53e7-f46f-48a2-8ebc-7f9b55476df5" />

## RESULT
Neural network classification model for the given dataset is successfully developed.
