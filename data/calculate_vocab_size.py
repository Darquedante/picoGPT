import re

# Define a function to tokenize and calculate the vocabulary size of a given text file
def calculate_vocab_size(file_path):
    with open(file_path, 'rb') as file:
        text = file.read().decode('utf-8', errors='replace')  # Decode text with error handling
    tokens = re.findall(r'\b\w+\b', text.lower())  # Tokenize text and normalize to lowercase
    vocab_size = len(set(tokens))  # Calculate unique tokens
    return vocab_size

# Paths to the cleaned dataset files
dataset_files = {
    "movie_lines_cleaned": "movie_lines_cleaned.txt",
    "new_persona": "new_persona.txt",
    "sherlock_holmes": "sherlock_holmes.txt"
}

# Calculate and print the vocabulary size for each dataset
vocab_sizes = {}
for name, path in dataset_files.items():
    vocab_size = calculate_vocab_size(path)
    vocab_sizes[name] = vocab_size
    print(f"Vocabulary size of {name}: {vocab_size}")

# Return the vocabulary sizes for further analysis
vocab_sizes
