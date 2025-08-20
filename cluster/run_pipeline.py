import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..', '..'))

from AngelFISH.src import Receipt, run_pipeline, silence
silence()

receipt_path = sys.argv[1]
print(f'running receipt at {receipt_path}')
run_pipeline(receipt_path)