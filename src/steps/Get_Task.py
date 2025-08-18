

from AngelFISH.src.Steps import segment, download_data, return_data, get_cell_properties, clear_local_data, detect_spots, match_masks, filter_csv, export_images, reconcile_data

# Function wrappers
class ExportImagesTask:
    @staticmethod
    def task_name():
        return "export_images"

    def __init__(self, receipt, step_name):
        self.receipt = receipt
        self.step_name = step_name

    def process(self):
        return export_images(self.receipt, self.step_name)

class ReconcileDataTask:
    @staticmethod
    def task_name():
        return "reconcile_data"

    def __init__(self, receipt, step_name):
        self.receipt = receipt
        self.step_name = step_name

    def process(self):
        return reconcile_data(self.receipt, self.step_name)


# get tasks
known_tasks = [segment, download_data, return_data, get_cell_properties, clear_local_data, detect_spots,
               match_masks, filter_csv, ExportImagesTask, ReconcileDataTask]

def get_task(task_name):
    for possible_task in known_tasks:
        if task_name == possible_task.task_name():
            return possible_task
    raise BaseException(f'Unknown task: {task_name}')

















