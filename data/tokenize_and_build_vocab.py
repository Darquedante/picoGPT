import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

# Ensure you have the necessary NLTK data
#nltk.download('punkt')

def tokenize_and_build_vocab(*file_paths):
    global_vocab = Counter()
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read().lower()  # Assuming lowercase normalization
            tokens = word_tokenize(text)
            global_vocab.update(tokens)
    
    # Here, global_vocab contains all tokens across datasets
    # You can trim the vocabulary based on frequency, if necessary
    return set(global_vocab.keys())

# Paths to your cleaned dataset files
dataset_paths = [
    'movie_lines_cleaned.txt',
    'new_persona.txt',
    'sherlock_holmes.txt'
]

# Build the shared vocabulary
shared_vocab = tokenize_and_build_vocab(*dataset_paths)

print(f"Shared vocabulary size: {len(shared_vocab)}")
