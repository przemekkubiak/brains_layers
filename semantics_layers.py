import pandas as pd
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Load pre-trained model and fast tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = TFBertModel.from_pretrained(model_name, output_hidden_states=True)

# Load the SNLI dataset from local files
train_file = 'data/SNLI_Corpus/snli_1.0_train.csv'
val_file = 'data/SNLI_Corpus/snli_1.0_dev.csv'
test_file = 'data/SNLI_Corpus/snli_1.0_test.csv'

train_data = pd.read_csv(train_file)
val_data = pd.read_csv(val_file)

# Filter out rows with NaN values or '-' in the labels
train_data = train_data[train_data['similarity'] != '-']
train_data = train_data.dropna(subset=['similarity'])
val_data = val_data[val_data['similarity'] != '-']
val_data = val_data.dropna(subset=['similarity'])

# Check if the similarity column contains numerical values or string labels
try:
    train_data['similarity'] = train_data['similarity'].astype(float)
    val_data['similarity'] = val_data['similarity'].astype(float)
    similarity_is_numerical = True
except ValueError:
    similarity_is_numerical = False

# Prepare data (using a small subset for demonstration)
subset_size = 1000  # Adjust this number based on your system's capacity
train_data = train_data.sample(n=subset_size, random_state=42)
val_data = val_data.sample(n=subset_size, random_state=42)

# Function to get texts and labels
def prepare_data(data_split, similarity_is_numerical):
    texts = list(zip(data_split['sentence1'], data_split['sentence2']))
    if similarity_is_numerical:
        labels = data_split['similarity'].astype(float).tolist()
    else:
        labels = data_split['similarity'].map({'entailment': 0, 'neutral': 1, 'contradiction': 2}).tolist()
    return texts, labels

train_texts, train_labels = prepare_data(train_data, similarity_is_numerical)
val_texts, val_labels = prepare_data(val_data, similarity_is_numerical)

# Check for NaN values in the labels
print(f"Train labels contain NaN: {pd.isna(train_labels).any()}")
print(f"Val labels contain NaN: {pd.isna(val_labels).any()}")

# Function to tokenize texts and align labels
def tokenize_and_align_labels(texts, labels):
    tokenized_inputs = tokenizer(
        [text[0] for text in texts], 
        [text[1] for text in texts], 
        truncation=True, 
        padding=True,  # Automatically pad to the longest sequence in the batch
        return_tensors='tf'
    )
    return tokenized_inputs, labels

# Tokenize and align labels using the fast tokenizer
train_inputs, train_labels = tokenize_and_align_labels(train_texts, train_labels)
val_inputs, val_labels = tokenize_and_align_labels(val_texts, val_labels)

# Debug prints
print(f"Train inputs shape: {train_inputs['input_ids'].shape}, Train labels length: {len(train_labels)}")
print(f"Val inputs shape: {val_inputs['input_ids'].shape}, Val labels length: {len(val_labels)}")

# Function to extract representations and train classifier for each layer
def evaluate_layers(inputs, labels, inputs_val, labels_val):
    performance = []
    num_layers = len(model.bert.encoder.layer) + 1  # Including the embedding layer

    # Get hidden states for training data
    outputs_train = model(inputs)
    hidden_states_train = outputs_train.hidden_states

    # Get hidden states for validation data
    outputs_val = model(inputs_val)
    hidden_states_val = outputs_val.hidden_states

    for layer_idx in range(num_layers):
        # Average token representations to get a single representation per sentence pair
        reps_train = tf.reduce_mean(hidden_states_train[layer_idx], axis=1).numpy()
        layer_labels_train = np.array(labels).reshape(-1)

        # Debug prints
        print(f"Layer {layer_idx} - Train reps shape: {reps_train.shape}, Train labels shape: {layer_labels_train.shape}")

        # Train classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(reps_train, layer_labels_train)

        # Validation
        reps_val = tf.reduce_mean(hidden_states_val[layer_idx], axis=1).numpy()
        layer_labels_val = np.array(labels_val).reshape(-1)

        # Debug prints
        print(f"Layer {layer_idx} - Val reps shape: {reps_val.shape}, Val labels shape: {layer_labels_val.shape}")

        val_preds = clf.predict(reps_val)
        acc = accuracy_score(layer_labels_val, val_preds)
        performance.append(acc)

    return performance

# Evaluate layers
layer_performance = evaluate_layers(train_inputs, train_labels, val_inputs, val_labels)

# Plot the performance
plt.figure(figsize=(10, 6))
plt.plot(range(len(layer_performance)), layer_performance, marker='o')
plt.title('Layer-wise Performance on Semantic Task (SNLI)')
plt.xlabel('Layer')
plt.ylabel('Accuracy')
plt.xticks(range(len(layer_performance)))
plt.grid(True)
plt.show()