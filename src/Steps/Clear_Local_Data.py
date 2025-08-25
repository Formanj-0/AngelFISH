import shutil
import os
import time
import psutil
import sys

from AngelFISH.src import abstract_task, load_data, close_data



class clear_local_data(abstract_task):
    def __init__(self, receipt, step_name):
        self.receipt = receipt
        self.step_name = step_name
        output_dir = os.path.dirname(self.receipt['arguments']['local_location'])
        self.output_path = os.path.join(output_dir, f"{self.receipt['arguments']['local_location']}_cleared.txt")

    @classmethod
    def task_name(cls):
        return 'clear_local_data'

    def process(self):
        start_time = time.time() 

        # These steps wont do anthing if the receipt already has the step
        # adds the step name to step order
        if self.step_name not in self.receipt['step_order']:
            self.receipt['step_order'].append(self.step_name)
        
        # makes sure the is a place for params
        if self.step_name not in self.receipt['steps'].keys():
            self.receipt['steps'][self.step_name] = {}

        # makes sure that the task_name is save (you can have multiple tasks of the same task)
        self.receipt['steps'][self.step_name]['task_name'] = self.task_name()

        local_location = self.receipt['arguments']['local_location']

        data = load_data(self.receipt)
        close_data(data)

        if os.path.exists(local_location):
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(local_location)
                    break
                except Exception as e:
                    if sys.platform == "win32":
                        for proc in psutil.process_iter(['pid', 'open_files', 'name']):
                            try:
                                flist = proc.info['open_files']
                                if flist:
                                    for nt in flist:
                                        if nt.path.startswith(local_location):
                                            proc.terminate()
                                            proc.wait(timeout=3)
                            except Exception:
                                continue
                        time.sleep(1)
                        if attempt == max_retries - 1:
                            raise RuntimeError(f"Failed to remove {local_location} after {max_retries} attempts: {e}")
                    else:
                        if attempt == max_retries - 1:
                            raise RuntimeError(f"Failed to remove {local_location} after {max_retries} attempts: {e}")
                        time.sleep(1)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "a") as f:
            f.write(f"{self.step_name} completed in {time.time() - start_time:.2f} seconds\n")

        return self.receipt

    @staticmethod
    def image_processing_function():
        pass


































