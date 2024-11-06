import torch
from transformers import BertTokenizerFast, BertModel, RobertaTokenizerFast, RobertaModel
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

dataset1 = 'data/regev_stimuli/Dataset1_SWJN_Stimuli.csv'
dataset2 = 'data/regev_stimuli/Dataset2_SN_Stimuli.csv'
synthetic_dataset = 'data/synthetic_dataset.csv'


dataset1 = pd.read_csv(dataset1)
dataset2 = pd.read_csv(dataset2)
synthetic_dataset = pd.read_csv(synthetic_dataset)

# Concatenate the two datasets
dataset = pd.concat([dataset1, dataset2], ignore_index=True)
dataset = pd.concat([dataset, synthetic_dataset], ignore_index=True)

# Shuffle the dataset
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Sample 250 data points for each condition label
dataset = dataset.groupby('condition').apply(lambda x: x.sample(n=200, random_state=42)).reset_index(drop=True)

def prepare_data(dataset):
    texts = dataset['stimulus_string'].tolist()
    labels = dataset['condition'].map({'SENTENCES': 0, 'WORDS': 1, 'JABBERWOCKY': 2, 'NONWORDS': 3}).tolist()
    return texts, labels

texts, labels = prepare_data(dataset)

# Tokenize the input texts
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

if __name__ == '__main__':

    # Obtain hidden states
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states  # Tuple of length 13 (embeddings + 12 hidden layers)

    # Extract representations from layer 4 and layer 12
    layer4_hidden_state = hidden_states[4]     # Shape: (batch_size, seq_length, hidden_size)
    layer12_hidden_state = hidden_states[12]   # Shape: (batch_size, seq_length, hidden_size)

    # Average the token embeddings to get sentence representations
    layer4_representation = torch.mean(layer4_hidden_state, dim=1)
    layer12_representation = torch.mean(layer12_hidden_state, dim=1)

    # Concatenate the representations
    aggregated_representation = torch.cat((layer4_representation, layer12_representation), dim=1)

    # Convert to NumPy arrays
    X = aggregated_representation.numpy()
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
    print(f"Test Accuracy (Aggregated Roberta): {accuracy}")

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