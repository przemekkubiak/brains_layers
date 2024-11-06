import pandas as pd
import random
import nltk
from nltk import CFG
import string

# Load the SNLI dataset
snli_path = 'data/SNLI_Corpus/snli_1.0_train.csv'
snli_data = pd.read_csv(snli_path)

# Extract the first 1000 sentences
sentences = snli_data['sentence1'].head(1000).tolist()

# Function to shuffle words in a sentence
def shuffle_sentence(sentence):
    words = sentence.split()
    random.shuffle(words)
    return ' '.join(words)

# Generate shuffled sentences for the "words" category
shuffled_sentences = [shuffle_sentence(sentence) for sentence in sentences]

# Function to generate made-up words
def generate_made_up_word(length):
    consonants = 'bcdfghjklmnpqrstvwxyz'
    vowels = 'aeiou'
    word = ''
    for i in range(length):
        if i % 2 == 0:
            word += random.choice(consonants)
        else:
            word += random.choice(vowels)
    return word

# Create lists of made-up nouns and verbs
nouns = [generate_made_up_word(random.randint(3, 7)) for _ in range(50)]
verbs = [generate_made_up_word(random.randint(3, 7)) for _ in range(50)]
adjectives = [generate_made_up_word(random.randint(3, 7)) for _ in range(50)]

# Define the grammar with made-up nouns, verbs, and adjectives
grammar = CFG.fromstring(f"""
S -> NP VP
NP -> Det N | Det Adj N | Det N PP
VP -> V NP | V NP PP
PP -> P NP
Det -> 'the' | 'a' | 'an' | 'this' | 'that' | 'these' | 'those'
P -> 'in' | 'on' | 'at' | 'by' | 'with' | 'about' | 'against' | 'between' | 'into' | 'through' | 'during' | 'before' | 'after' | 'above' | 'below' | 'to' | 'from' | 'up' | 'down' | 'over' | 'under'
Adj -> {' | '.join(f"'{adj}'" for adj in adjectives)}
N -> {' | '.join(f"'{noun}'" for noun in nouns)}
V -> {' | '.join(f"'{verb}'" for verb in verbs)}
""")

# Function to generate sentences
def generate_sentence(grammar, start_symbol):
    if isinstance(start_symbol, nltk.Nonterminal):
        productions = grammar.productions(lhs=start_symbol)
        production = random.choice(productions)
        sentence = []
        for sym in production.rhs():
            sentence.extend(generate_sentence(grammar, sym))
        return sentence
    else:
        return [start_symbol]

# Generate "jabberwocky" sentences
jabberwocky_sentences = [' '.join(generate_sentence(grammar, grammar.start())) for _ in range(1000)]

# Function to generate nonword sentences
def generate_nonword(length):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    return ''.join(random.choice(letters) for _ in range(length))

def generate_nonword_sentence(num_words):
    return ' '.join(generate_nonword(random.randint(3, 7)) for _ in range(num_words))

# Generate "nonwords" sentences
nonword_sentences = [generate_nonword_sentence(random.randint(5, 10)) for _ in range(1000)]

# Convert all text to uppercase and remove punctuation
def preprocess_text(text):
    text = text.upper()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Apply preprocessing to all categories
sentences = [preprocess_text(sentence) for sentence in sentences]
shuffled_sentences = [preprocess_text(sentence) for sentence in shuffled_sentences]
jabberwocky_sentences = [preprocess_text(sentence) for sentence in jabberwocky_sentences]
nonword_sentences = [preprocess_text(sentence) for sentence in nonword_sentences]

# Combine all categories into a single dataset
data = {
    'condition': ['SENTENCES'] * 1000 + ['WORDS'] * 1000 + ['JABBERWOCKY'] * 1000 + ['NONWORDS'] * 1000,
    'stimulus_string': sentences + shuffled_sentences + jabberwocky_sentences + nonword_sentences
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Remove duplicate rows
df = df.drop_duplicates()

# Save to CSV
df.to_csv('synthetic_dataset.csv', index=False)

print("Synthetic dataset created and saved to 'synthetic_dataset.csv'")