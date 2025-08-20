import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..', '..'))

from src import Receipt, run_pipeline, silence
silence()

receipt_path = sys.argv[1]

run_pipeline(receipt_path)