import os
import glob
import tarfile

coswara_data_dir = os.path.abspath("Coswara-Data-master")
extracted_data_dir = os.path.join(coswara_data_dir, "Extracted_data")

if not os.path.exists(extracted_data_dir):
    os.makedirs(extracted_data_dir)

dirs = glob.glob(os.path.join(coswara_data_dir, "202*"))

for d in dirs:
    parts = sorted(glob.glob(os.path.join(d, "*.tar.gz.*")))
    
    if not parts:
        continue

    print(f"Extraindo {d}...")

    # juntar partes
    merged_file = os.path.join(d, "merged.tar.gz")
    
    with open(merged_file, "wb") as outfile:
        for part in parts:
            with open(part, "rb") as infile:
                outfile.write(infile.read())

    # extrair
    with tarfile.open(merged_file, "r:gz") as tar:
        tar.extractall(path=extracted_data_dir)

print("Extração completa!")