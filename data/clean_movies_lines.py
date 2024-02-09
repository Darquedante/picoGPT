def clean_movie_lines(input_path, output_path):
    cleaned_lines = []
    with open(input_path, 'rb') as file:
        for line in file:
            try:
                # Decode the line using utf-8 encoding and replace any errors
                line = line.decode('utf-8', errors='replace')
                # Splitting by " +++$+++" and extracting the dialogue text
                dialogue = line.split(" +++$+++")[-1].strip()
                # Optional: convert dialogue to lowercase
                dialogue = dialogue.lower()
                cleaned_lines.append(dialogue)
            except UnicodeDecodeError:
                # Skip lines that cannot be decoded properly
                continue
    
    with open(output_path, 'w', encoding='utf-8') as file:
        for line in cleaned_lines:
            file.write(f"{line}\n")

# Define input and output paths
input_path = "movie_lines.txt"
output_path = "movie_lines_cleaned.txt"

# Clean the movie lines dataset
clean_movie_lines(input_path, output_path)

# Indicate completion and output path of the cleaned data
output_path
