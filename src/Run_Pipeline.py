import yaml
import paramiko
import os
import time

from AngelFISH.src import Receipt
from AngelFISH.src.Steps import get_task


def run_step(receipt_path, step_name):
    receipt = Receipt(path=receipt_path)
    task_class = get_task(receipt['steps'][step_name]['task_name'])
    task = task_class(receipt, step_name)
    task.process()


def run_pipeline(receipt_path):
    receipt = Receipt(path=receipt_path)
    for sn in receipt['step_order']:
        run_step(receipt_path, sn)
    return receipt


def wait_for_job_completion(ssh, job_id, poll_interval=60):
    while True:
        stdin, stdout, stderr = ssh.exec_command(f'squeue -j {job_id} -h')
        job_status = stdout.read().decode().strip()
        if not job_status:
            # Job no longer in queue, check final status with sacct
            stdin, stdout, stderr = ssh.exec_command(f'sacct -j {job_id} --format=State --noheader')
            final_status = stdout.read().decode().strip()
            print(f"Job {job_id} finished with status: {final_status}")
            return final_status
        else:
            print(f"Job {job_id} still running...")
            time.sleep(poll_interval)


def run_pipeline_remote(receipt_path, remote_path):
        config_path = os.path.join(os.path.dirname(__file__), 'config_cluster.yml')

        conf = yaml.safe_load(open(config_path))
        usr = str(conf['user']['username'])
        pwd = str(conf['user']['password'])
        remote_address = str(conf['user']['remote_address'])
        port = 22

        # Create SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(remote_address, port, usr, pwd)

        # Remote path to file with all directories
        remote_receipt_path = os.path.join(remote_path, os.path.basename(receipt_path))
        remote_receipt_path = remote_receipt_path.replace('\\', '/')

        # Transfer the file
        sftp = ssh.open_sftp()
        sftp.put(receipt_path, remote_receipt_path)
        sftp.close()

        # Submit the SLURM job and capture job ID
        sbatch_command = f'sbatch --parsable run_pipeline.sh {remote_receipt_path}'
        combined_command = f'cd {remote_path}; {sbatch_command}'

        stdin, stdout, stderr = ssh.exec_command(combined_command)
        job_submission_output = stdout.read().decode().strip()
        job_submission_error = stderr.read().decode()

        # Parse job ID
        if job_submission_error:
            raise RuntimeError(f"SLURM job submission failed: {job_submission_error}")

        job_id = job_submission_output.split(';')[0]
        print(f"Submitted SLURM job with ID: {job_id}")

        wait_for_job_completion(ssh, job_id, 10)