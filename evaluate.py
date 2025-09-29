import torch
from nltk_utils import bag_of_words, tokenize, stem
from model import RNNModel
import json

# Load trained model
FILE = "data_rnn.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
intents = data["intents"]

model = RNNModel(input_size, hidden_size, output_size, num_layers=data["num_layers"])
model.load_state_dict(data["model_state"])
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Preprocess sentence
def preprocess_sentence(sentence):
    tokens = tokenize(sentence)
    tokens = [stem(w) for w in tokens]
    bow = bag_of_words(tokens, all_words)
    return torch.tensor(bow, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)

# Predict tag
def predict_tag(sentence):
    bow = preprocess_sentence(sentence)
    with torch.no_grad():
        outputs = model(bow)
        _, predicted = torch.max(outputs, dim=1)
        tag = tags[predicted.item()]
    return tag

# Example test dataset
test_sentences = [
    ("Hi there", "greeting"),
    ("Bye bye", "goodbye"),
    ("Can you tell me a joke?", "joke"),
    ("I have cold, cough, tiredness", "Common Cold"),
    ("I am sneezing and have itchy eyes", "Allergies"),
    ("I feel shortness of breath and wheezing", "Asthma"),
    ("I have joint pain and stiffness", "Rheumatoid Arthritis"),
    ("How do you work?", "work"),
    ("Who are you?", "who"),
    ("Thanks a lot!", "Thanks")
]

# Evaluate accuracy
correct = 0
for sentence, true_tag in test_sentences:
    pred_tag = predict_tag(sentence)
    print(f"Sentence: {sentence}")
    print(f"Predicted: {pred_tag}, True: {true_tag}")
    print("---")
    if pred_tag == true_tag:
        correct += 1

accuracy = correct / len(test_sentences)
print(f"Accuracy on test set: {accuracy*100:.2f}%")
