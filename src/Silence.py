import logging

def silence():
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger('napari').setLevel(logging.WARNING)
    logging.getLogger('in_n_out').setLevel(logging.WARNING)
    logging.getLogger('numcodecs').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('paramiko').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


