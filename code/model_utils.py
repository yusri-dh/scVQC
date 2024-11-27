import random
import numpy as np
import torch


# For reproducibility
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

    torch.cuda.manual_seed_all(s)
    # add additional seed
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
