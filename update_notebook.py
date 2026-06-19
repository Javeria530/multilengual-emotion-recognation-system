import json

filename = "mtkd4ser.ipynb"
with open(filename, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        new_source = []
        for line in cell.get('source', []):
            # Update git clone URL
            if "!git clone" in line:
                line = line.replace("https://github.com/aalto-speech/mtkd4ser.git", "https://github.com/Javeria530/multilengual-emotion-recognation-system.git")
            
            # Update execution script
            if "!python" in line and "main.py" in line:
                line = line.replace("/content/mtkd4ser/main.py", "/content/multilengual-emotion-recognation-system/main.py")
                line = line.replace("--N_EPOCHS 5", "--N_EPOCHS 100")
                
                # Prepend the env variable to the cell if it's the execution cell
                new_source.append('import os\n')
                new_source.append('os.environ["MTKD_BASE_DIR"] = "/content/" # Set this to your Drive path if data is on Drive\n')
            
            new_source.append(line)
        cell['source'] = new_source

with open(filename, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Notebook updated successfully.")
