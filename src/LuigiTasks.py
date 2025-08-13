
import sys
import os
import luigi
import sciluigi as sl
import logging
import yaml
import paramiko
import time
from luigi.contrib.ssh import RemoteTarget
import subprocess

from src import Receipt


class AngelFISHLuigiTask(luigi.Task):
    receipt_path = luigi.Parameter()
    remote_path = luigi.Parameter()
    step_name = luigi.Parameter()
    output_path = luigi.Parameter()
    config_path = luigi.Parameter()
    public_key = luigi.Parameter()

    def out_doneflag(self):
        return RemoteTarget(self.output_path, host='keck.engr.colostate.edu', username='formanj', key_file=self.public_key)

    def run(self):
        # Load the configuration
        conf = yaml.safe_load(open(str(self.config_path)))
        usr = str(conf['user']['username'])
        pwd = str(conf['user']['password'])
        remote_address = str(conf['user']['remote_address'])
        port = 22

        # Create SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(remote_address, port, usr, pwd)

        remote_receipt_path = os.path.basename(self.receipt_path)
        # remote_receipt_path = remote_receipt_path.replace('\\', '/')

        # Submit the SLURM job and capture job ID
        sbatch_command = f'sbatch --parsable run_step.sh {remote_receipt_path} {self.step_name}'
        combined_command = f'cd {self.remote_path}; {sbatch_command}'

        stdin, stdout, stderr = ssh.exec_command(combined_command)
        job_submission_output = stdout.read().decode().strip()
        job_submission_error = stderr.read().decode()

        # Parse job ID
        if job_submission_error:
            raise RuntimeError(f"SLURM job submission failed: {job_submission_error}")

        job_id = job_submission_output.split(';')[0]
        print(f"Submitted SLURM job with ID: {job_id}")

        # Poll until job completes
        import time

        def is_job_active(ssh_client, job_id):
            check_command = f'squeue -j {job_id} -h'
            stdin, stdout, stderr = ssh_client.exec_command(check_command)
            result = stdout.read().decode()
            return bool(result.strip())  # job is active if output is non-empty

        wait_time = 10  # seconds
        max_wait = 3600  # 1 hour timeout
        elapsed = 0

        while is_job_active(ssh, job_id):
            if elapsed >= max_wait:
                raise TimeoutError(f"SLURM job {job_id} did not complete within {max_wait} seconds.")
            print(f"[{job_id}] Still running... waiting {wait_time}s")
            time.sleep(wait_time)
            elapsed += wait_time

        print(f"[{job_id}] Job complete.")

        # Close the SSH connection
        ssh.close()




class Upload_Task(luigi.Task):
    output_path = luigi.Parameter()
    receipt_path = luigi.Parameter()
    remote_path = luigi.Parameter()
    config_path = luigi.Parameter()
    public_key = luigi.Parameter()

    def out_doneflag(self):
        return RemoteTarget(self.output_path, host='keck.engr.colostate.edu', username='formanj', key_file=self.public_key)

    def run(self):
        conf = yaml.safe_load(open(self.config_path))
        usr = str(conf['user']['username'])
        pwd = str(conf['user']['password'])
        remote_address = str(conf['user']['remote_address'])
        port = 22

        # Create SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(remote_address, port, usr, pwd)

        # Remote path to file with all directories
        remote_receipt_path = os.path.join(self.remote_path, os.path.basename(self.receipt_path))
        remote_receipt_path = remote_receipt_path.replace('\\', '/')

        # Transfer the file
        sftp = ssh.open_sftp()
        sftp.put(self.receipt_path, remote_receipt_path)
        sftp.close()


class AngelFISHWorkflow(sl.WorkflowTask):
    receipt_path = luigi.Parameter()
    cluster_path = luigi.Parameter()
    config_path = luigi.Parameter()
    public_key = luigi.Parameter()

    def workflow(self):
        # upload 
        task_refs = []
        self.cluster_path = self.cluster_path.replace('\\', '/')
        receipt_filename = os.path.basename(self.receipt_path).replace('\\', '/')
        remote_receipt_path = os.path.join(self.cluster_path, receipt_filename).replace('\\', '/')
        # status_dir = r'C:\Users\formanj\GitHub\AngelFISH\cluster\status'
        
        step_task = self.new_task(
                            'upload_task',
                            Upload_Task,
                            output_path=remote_receipt_path, # os.path.join(status_dir, 'upload_receipt.done'),
                            receipt_path=self.receipt_path,
                            remote_path=self.cluster_path,
                            config_path=self.config_path,
                            public_key=self.public_key
                        )
        previous_task = step_task
        task_refs.append(step_task)

        receipt = Receipt(path=self.receipt_path)
        step_order = receipt['step_order']
        name = os.path.basename(receipt['meta_arguments']['nas_location'])
        database_loc = os.path.dirname(self.cluster_path)
        database_loc = os.path.join(database_loc, 'database')
        remote_local_location = os.path.join(database_loc, name).replace('\\', '/')
        remote_analysis_dir = os.path.join(remote_local_location, receipt['meta_arguments']['analysis_name'])
        remote_status_dir = os.path.join(remote_analysis_dir, 'status')

        for step_name in step_order:
            StepTask = type(
                step_name,          # Unique class name
                (AngelFISHLuigiTask,),            # Base class
                {}                                # No extra attributes needed
            )
            
            path = os.path.join(remote_status_dir, f'step_{step_name}.txt').replace('\\', '/')
            step_task = self.new_task(
                                step_name,
                                StepTask,
                                receipt_path=remote_receipt_path,
                                step_name=step_name,
                                output_path=path, # os.path.join(status_dir, f'{step_name}.done'),
                                remote_path=self.cluster_path,
                                config_path=self.config_path,
                                public_key=self.public_key
                                )

            # Add dependency chain
            if previous_task is not None:
                step_task.in_upstream = previous_task.out_doneflag
            previous_task = step_task
            task_refs.append(step_task)
        return task_refs