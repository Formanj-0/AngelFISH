import yaml
import paramiko
import os
import time
from datetime import datetime
from paramiko.ssh_exception import SSHException, NoValidConnectionsError
from time import sleep

from AngelFISH.src import Receipt
from AngelFISH.src.Steps import get_task
import random

def create_ssh_client(host, port, username, password, retries=3, timeout=30):
    """Creates an SSH client with retries and timeout settings."""
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    for attempt in range(retries):
        try:
            print(f"Attempting SSH connection... ({attempt + 1}/{retries})")
            ssh_client.connect(host, port, username, password, timeout=timeout)
            return ssh_client
        except (SSHException, NoValidConnectionsError) as e:
            print(f"Connection failed: {e}")
            if attempt < retries - 1:
                wait_time = random.uniform(1, 5)
                print(f"Retrying in {wait_time:.2f} seconds...")
                sleep(wait_time)
            else:
                raise RuntimeError(f"SSH connection failed after {retries} attempts.")
    return None

def run_step(receipt_path, step_name):
    receipt = Receipt(path=receipt_path)
    task_class = get_task(receipt['steps'][step_name]['task_name'])
    task = task_class(receipt, step_name)
    print(f'=================== {step_name} ===================')
    task.process()


def run_pipeline(receipt_path, new_nas_loc:str=None, new_loc_loc:str=None):
    receipt = Receipt(path=receipt_path)
    for sn in receipt['step_order']:
        run_step(receipt_path, sn)
    return receipt_path


def wait_for_job_completion(ssh, job_id, poll_interval=60):
    while True:
        stdin, stdout, stderr = ssh.exec_command(f'squeue -j {job_id} -h')
        job_status = stdout.read().decode().strip()
        if not job_status:
            # Job no longer in queue, check final status with sacct
            stdin, stdout, stderr = ssh.exec_command(f'sacct -j {job_id} --format=State --noheader')
            final_status = stdout.read().decode().strip()
            # print(f"Job {job_id} finished with status: {final_status}")
            return final_status
        else:
            # print(f"Job {job_id} still running...")
            time.sleep(poll_interval)


def run_pipeline_remote(receipt_path, remote_path, new_nas_loc=None, new_loc_loc=None, use_default:bool=True):
    """Runs the pipeline remotely, handling SSH, file transfer, and SLURM submission."""
    if use_default:
        receipt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'default_pipelines', receipt_path))
    receipt = Receipt(path=receipt_path)
    if new_nas_loc is not None:
        receipt['arguments']['nas_location'] = new_nas_loc
    if new_loc_loc is not None:
        receipt['arguments']['local_location'] = new_loc_loc

    new_dir = os.getcwd()
    filename_without_ext = os.path.splitext(os.path.basename(receipt_path))[0]
    current_time = datetime.now().strftime('%Y%m%d')
    rand_num = random.randint(1000, 9999)
    receipt_path = os.path.join(new_dir, f'{filename_without_ext}_{current_time}_{rand_num}.json')
    receipt.save(receipt_path)

    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config_cluster.yml'))
    conf = yaml.safe_load(open(config_path))
    usr = str(conf['user']['username'])
    pwd = str(conf['user']['password'])
    remote_address = str(conf['user']['remote_address'])
    port = 22

    # Create SSH client
    ssh = create_ssh_client(remote_address, port, usr, pwd)

    # Remote path to file with all directories
    remote_receipt_path = os.path.join(remote_path, 'pipelines', os.path.basename(receipt_path))
    remote_receipt_path = remote_receipt_path.replace('\\', '/')

    # Transfer the file
    sftp = ssh.open_sftp()
    sftp.put(receipt_path, remote_receipt_path)
    sftp.close()
    os.remove(receipt_path)

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
    if new_nas_loc:
        print(f'{new_nas_loc}')
    print(f"Submitted SLURM job with ID: {job_id}")

    final_status = wait_for_job_completion(ssh, job_id, 10)
    ssh.close()

    return final_status


def process_path(receipt_path, remote_path, nas_path):
    return run_pipeline_remote(receipt_path, remote_path, nas_path, None)