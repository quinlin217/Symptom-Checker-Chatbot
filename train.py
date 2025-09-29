import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import RNNModel
from sklearn.model_selection import train_test_split


# Load intents from JSON
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# Loop through each sentence in intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_data = np.array(X_train)
y_data = np.array(y_train)

# Define the RNN model configuration
input_size = len(X_train[0])
hidden_size = 32
output_size = len(tags)
num_layers = 5  # Specify the number of layers for your RNN model

# Create a dataset and data loader for training
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.x_data = X
        self.y_data = y

    def __getitem__(self, index):
        x = torch.from_numpy(self.x_data[index]).float() 
        y = torch.tensor(self.y_data[index]).long() 
        return x, y

    def __len__(self):
        return self.n_samples
    
# Split dataset
X_train, X_val, y_train, y_val = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42#, stratify=y_train
)

train_dataset = ChatDataset(X_train, y_train)
val_dataset = ChatDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

# Set device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_and_validate(model, train_loader, val_loader, num_epochs=1000, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)

            words = words.view(-1, len(words), input_size)
            outputs = model(words)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for (words, labels) in val_loader:
                words = words.to(device)
                labels = labels.to(dtype=torch.long).to(device)
                words = words.view(-1, len(words), input_size)
                outputs = model(words)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    return best_state, best_val_acc

# Initialize the RNN model
model = RNNModel(input_size, hidden_size, output_size, num_layers).to(device)
best_state, best_val_acc = train_and_validate(model, train_loader, val_loader)
print(f"Best validation accuracy: {best_val_acc:.4f}")

full_dataset = ChatDataset(X_data, y_data)
full_loader = DataLoader(full_dataset, batch_size=1, shuffle=True, num_workers=0)

final_model = RNNModel(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)

# Train the RNN model
num_epochs = 1000
for epoch in range(num_epochs):
    for (words, labels) in full_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        words = words.view(-1, len(words), input_size)
        outputs = final_model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"Final Training Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# Save the trained RNN model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags,
    "intents": intents
}

FILE = "data_rnn.pth"
torch.save(data, FILE)

print(f'Training complete. RNN model saved to {FILE}')
