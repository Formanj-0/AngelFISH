# %%
import sys
import os
import luigi
import sciluigi as sl
import subprocess
import logging

sys.path.append('..')  # adjust if needed
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger('SMB').setLevel(logging.WARNING)
logging.getLogger('napari').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('in_n_out').setLevel(logging.WARNING)
logging.getLogger('numcodecs').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('luigi').setLevel(logging.WARNING)
logging.getLogger('numexpr').setLevel(logging.WARNING)
logging.getLogger('luigi-interface').setLevel(logging.WARNING)
logging.getLogger('sciluigi-interface').setLevel(logging.WARNING)
logging.getLogger('cellpose').setLevel(logging.WARNING)

from src import Receipt
from src.steps import get_task

# python -m luigi --module angelfish_pipeline_local AngelFISHWorkflow --receipt-path new_pipeline.json --local-scheduler
# python -m luigi --module angelfish_pipeline_local AngelFISHWorkflow --receipt-path new_pipeline.json

class AngelFISHLuigiTask(sl.ExternalTask):
    receipt_path = luigi.Parameter()
    step_name = luigi.Parameter()
    output_path = luigi.Parameter()

    @property
    def task_name(self):
        return self.step_name

    def out_doneflag(self):
        return sl.TargetInfo(self, self.output_path)

    def run(self):
        receipt = Receipt(path=self.receipt_path)
        task_class = get_task(receipt['steps'][self.step_name]['task_name'])
        task = task_class(receipt, self.step_name)

        # Option 1: Use subprocess to call the SLURM wrapper script remotely
        command = f"sbatch run_step.sh {self.receipt_path} {self.step_name}"
        subprocess.run(['ssh', 'user@remote-hpc', command], check=True)


class AngelFISHWorkflow(sl.WorkflowTask):
    receipt_path = luigi.Parameter()

    def out_doneflag(self):
        receipt = Receipt(path=self.receipt_path)
        return sl.TargetInfo(self, f'{receipt['meta_arguments']['analysis_name']}.txt')

    def workflow(self):
        receipt = Receipt(path=self.receipt_path)
        step_order = receipt['step_order']

        previous_task = None
        task_refs = []
        for step_name in step_order:
            path = os.path.join(receipt['dirs']['status_dir'], f'step_{step_name}.txt')
            step_task = self.new_task(
                step_name,
                AngelFISHLuigiTask,
                receipt_path=self.receipt_path,
                step_name=step_name,
                output_path=path
            )

            # Add dependency chain
            if previous_task is not None:
                step_task.in_upstream = previous_task.out_doneflag

            previous_task = step_task
            task_refs.append(step_task)


        # with open(self.out_doneflag().path, "w") as f:
        #     f.write("done\n")

        return task_refs

# if __name__ == '__main__': # this is being mean an I dont wanna fix it rn
#     sl.run_local(main_task_cls=AngelFISHWorkflow)