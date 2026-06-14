def set_seed(seed: int):
    import random
    import os
    import numpy as np
    import torch

    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN (reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False