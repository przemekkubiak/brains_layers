from transformers import BertTokenizerFast, BertModel
from transformers import RobertaTokenizerFast, RobertaModel
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from aggregated_bert import texts, labels

"""# Load BERT
model_name = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_hidden_states=True)
model.eval()"""

# Load Roberta
model_name = 'roberta-base'
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name, output_hidden_states=True)
model.eval()

# Tokenize the input texts
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')


# Obtain hidden states
with torch.no_grad():
    outputs = model(**inputs)
    hidden_states = outputs.hidden_states  # Tuple of length 13 (embeddings + 12 hidden layers)


# Extract representations from the last hidden layer (layer 12)
layer12_hidden_state = hidden_states[12]   # Shape: (batch_size, seq_length, hidden_size)

# Average the token embeddings to get sentence representations
sentence_representation = torch.mean(layer12_hidden_state, dim=1)

# Convert to NumPy arrays
X = sentence_representation.numpy()
y = np.array(labels)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train a classifier
clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy (Standard Roberta): {accuracy}")

# Define class names
class_names = ['Sentences', 'Words', 'Jabberwocky', 'Nonwords']

# Generate classification report
report = classification_report(y_test, y_pred, target_names=class_names)
print("Classification Report:")
print(report)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
