from src.steps import segment, download_data

known_tasks = [segment, download_data]

def get_task(task_name):
    for possible_task in known_tasks:
        if task_name == possible_task.task_name():
            return possible_task
    raise BaseException(f'Unknown task: {task_name}')

















