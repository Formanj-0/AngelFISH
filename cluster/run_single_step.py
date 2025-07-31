import sys
sys.path.append('..')

from src import Receipt
from src.steps import get_task

receipt_path = sys.argv[1]
step_name = sys.argv[2]

receipt = Receipt(path=receipt_path)
task_class = get_task(receipt['steps'][step_name]['task_name'])
task = task_class(receipt, step_name)
task.process()