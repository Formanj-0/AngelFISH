import sys
sys.path.append('..')

from src import Receipt, run_step

receipt_path = sys.argv[1]
step_name = sys.argv[2]

run_step(receipt_path, step_name)