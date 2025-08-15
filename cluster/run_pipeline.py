import sys
sys.path.append('..')

from src import Receipt, run_pipeline

receipt_path = sys.argv[1]

run_pipeline(receipt_path)