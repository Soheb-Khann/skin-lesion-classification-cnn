import random
import numpy as np
import torch
import os


def set_seed(seed=42):

    # PYTHON RANDOM
    random.seed(seed)

    # NUMPY
    np.random.seed(seed)

    # PYTORCH (CPU)
    torch.manual_seed(seed)

    # PYTORCH (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # CUDNN SETTINGS
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ENV SEED
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f" Seed set to {seed}")

# DATALOADER WORKER SEE
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)