import os

def create_smol_file(filename, num_lines=25):
    # Read the first few lines of the original file
    with open(filename, 'rb') as f:
        content = []
        for _ in range(num_lines):
            line = f.readline()
            if not line:
                break
            content.append(line.decode('utf-8', errors='replace'))
    
    # Create a new filename with "_smol" appended
    base_name, ext = os.path.splitext(filename)
    smol_filename = f"{base_name}_smol{ext}"
    
    # Write the sampled content into the new file
    with open(smol_filename, 'w', encoding='utf-8') as f:
        f.writelines(content)

# Get the current directory
current_directory = os.getcwd()

# List all files in the current directory
file_list = [file for file in os.listdir(current_directory) if file.endswith('.txt')]

# Process each text file
for filename in file_list:
    create_smol_file(filename)
