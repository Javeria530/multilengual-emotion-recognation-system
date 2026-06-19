import os

files = ['data.py', 'models.py', 'pipeline.py', 'utils.py']

for f in files:
    with open(f, 'r') as file:
        content = file.read()
    
    # Fix double f-strings
    content = content.replace('ff"{BASE_DIR}', 'f"{BASE_DIR}')
    
    # Fix the fallback definition
    content = content.replace('os.environ.get("MTKD_BASE_DIR", f"{BASE_DIR}")', 'os.environ.get("MTKD_BASE_DIR", "/m/triton/scratch/elec/t405-puhe/p/bijoym1")')
    
    with open(f, 'w') as file:
        file.write(content)

print("Syntax errors fixed.")
