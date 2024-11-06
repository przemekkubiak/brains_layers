import torch
from transformers import BertTokenizerFast, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from itertools import chain
from sklearn.preprocessing import LabelEncoder

# Load pre-trained model and fast tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_hidden_states=True)

# Function to read and parse the CoNLL-2003 dataset
def read_conll2003(filename):
    sentences = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as f:
        tokens = []
        pos_tags = []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    labels.append(pos_tags)
                    tokens = []
                    pos_tags = []
                continue
            if line.startswith('-DOCSTART-') or line == '':
                continue
            else:
                splits = line.split()
                if len(splits) >= 2:
                    word = splits[0]
                    pos_tag = splits[1]
                    tokens.append(word)
                    pos_tags.append(pos_tag)
        if tokens:
            sentences.append(tokens)
            labels.append(pos_tags)
    return sentences, labels

# Paths to your local CoNLL-2003 data files
train_file = 'data/conll2003/train.txt'
val_file = 'data/conll2003/valid.txt'

# Read the data files
train_sentences, train_labels = read_conll2003(train_file)
val_sentences, val_labels = read_conll2003(val_file)

# Limit to a smaller subset
subset_size = 100  # Adjust this number based on your system's capacity
train_sentences = train_sentences[:subset_size]
train_labels = train_labels[:subset_size]
val_sentences = val_sentences[:subset_size]
val_labels = val_labels[:subset_size]

# Convert lists of tokens to texts
train_texts = [' '.join(tokens) for tokens in train_sentences]
val_texts = [' '.join(tokens) for tokens in val_sentences]

# Combine all labels to create a label encoder
all_labels = list(chain(*train_labels, *val_labels))
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Function to tokenize texts and align labels
def tokenize_and_align_labels(texts, labels):
    tokenized_inputs = tokenizer(texts, truncation=True, padding=True, return_tensors='pt', is_split_into_words=True)
    labels_list = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoder.transform([label[word_idx]])[0])
                previous_word_idx = word_idx
            else:
                # For subword tokens
                label_ids.append(-100)
        labels_list.append(label_ids)
    return tokenized_inputs, labels_list

# Tokenize and align labels using the fast tokenizer
train_inputs, train_aligned_labels = tokenize_and_align_labels(train_sentences, train_labels)
val_inputs, val_aligned_labels = tokenize_and_align_labels(val_sentences, val_labels)

# Function to extract representations and train classifier for each layer
def evaluate_layers(inputs, labels, inputs_val, labels_val, batch_size=32):
    performance = []
    num_layers = len(model.encoder.layer) + 1  # Including the embedding layer

    # Function to get representations in batches
    def get_hidden_states(inputs, labels):
        all_hidden_states = [[] for _ in range(num_layers)]
        all_labels = []
        num_samples = inputs['input_ids'].size(0)
        for i in range(0, num_samples, batch_size):
            batch_inputs = {k: v[i:i+batch_size] for k, v in inputs.items()}
            batch_labels = labels[i:i+batch_size]
            with torch.no_grad():
                outputs = model(**batch_inputs)
                hidden_states = outputs.hidden_states
            for layer_idx in range(num_layers):
                layer_hidden = hidden_states[layer_idx]
                all_hidden_states[layer_idx].append(layer_hidden)
            all_labels.extend(batch_labels)
        # Concatenate batches
        all_hidden_states = [torch.cat(h, dim=0) for h in all_hidden_states]
        return all_hidden_states, all_labels

    # Get hidden states for training and validation data
    hidden_states_train, labels_train = get_hidden_states(inputs, labels)
    hidden_states_val, labels_val = get_hidden_states(inputs_val, labels_val)

    for layer_idx in range(len(hidden_states_train)):
        # Flatten representations and labels
        reps = hidden_states_train[layer_idx].reshape(-1, hidden_states_train[layer_idx].size(-1)).cpu().numpy()
        layer_labels = torch.tensor(labels_train).view(-1)
        valid_indices = layer_labels != -100
        reps = reps[valid_indices]
        layer_labels = layer_labels[valid_indices].numpy()

        # Train classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(reps, layer_labels)

        # Validation
        val_reps = hidden_states_val[layer_idx].reshape(-1, hidden_states_val[layer_idx].size(-1)).cpu().numpy()
        val_layer_labels = torch.tensor(labels_val).view(-1)
        val_valid_indices = val_layer_labels != -100
        val_reps = val_reps[val_valid_indices]
        val_layer_labels = val_layer_labels[val_valid_indices].numpy()

        val_preds = clf.predict(val_reps)
        acc = accuracy_score(val_layer_labels, val_preds)
        performance.append(acc)

    return performance

# Evaluate layers
layer_performance = evaluate_layers(train_inputs, train_aligned_labels, val_inputs, val_aligned_labels)

# Plot the performance
plt.figure(figsize=(10, 6))
plt.plot(range(len(layer_performance)), layer_performance, marker='o')
plt.title('Layer-wise Performance on POS Tagging')
plt.xlabel('Layer')
plt.ylabel('Accuracy')
plt.xticks(range(len(layer_performance)))
plt.grid(True)
plt.show()