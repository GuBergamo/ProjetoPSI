import subprocess
import time
import sys


def run_script(script_name):

    print("\n======================================")
    print(f"RUNNING: {script_name}")
    print("======================================")

    start = time.time()

    result = subprocess.run(
    [sys.executable, script_name]
    )

    end = time.time()

    elapsed = end - start

    if result.returncode != 0:

        print(f"\nERROR while running {script_name}")

        exit()

    print(f"\nFinished in {elapsed:.2f} seconds")

# ============================================================
# PIPELINE
# ============================================================

pipeline = [
    "prepare_dataset.py",    # Ver quantas classifações há em 'covid.status' do Coswara

    "filter_dataset.py",    # Filtrar o dataset para os arquivos de audio escolhidos, gerando o 'balanced_dataset.csv'

    "extract_features.py",  # Extrair features MFCC

    "train_svm_parallel.py",# Treinar svm (paralelizando as operações para ficar mais rápido mas sem prints em tempo real)

    "plots.py",             # Plotar os resultados obtidos do svm

    "train_RNN.py"          # Treinar RNN
]

# ============================================================
# RUN EVERYTHING
# ============================================================

total_start = time.time()

for script in pipeline:

    run_script(script)

total_end = time.time()

print("\n======================================")
print("ACABOU!!")
print("======================================")

print(
    f"Total time: "
    f"{(total_end-total_start)/60:.2f} minutes"
)