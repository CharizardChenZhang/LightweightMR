import logging
import torch.distributed as dist
import numpy as np
import torch
from models.cpplib.libkdtree import KDTree
import torch.nn.functional as F
import random
import open3d as o3d

logger_initialized = {}

def get_root_logger(log_file=None, log_level=logging.INFO, name='main'):
    """Get root logger and add a keyword filter to it.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmdet3d".
    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
        name (str, optional): The name of the root logger, also used as a
            filter keyword. Defaults to 'mmdet3d'.
    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name=name, log_file=log_file, log_level=log_level)
    # add a logging filter
    logging_filter = logging.Filter(name)
    logging_filter.filter = lambda record: record.find(name) != -1

    return logger


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True


    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.
    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')

def get_neighbor_idx_noself(pc, query_pts, k):
    kdtree = KDTree(pc)
    (x, idx) = kdtree.query(query_pts, k + 1)
    idx = idx[:, 1:]
    idx = torch.from_numpy(idx.astype(int)).long()

    return idx.squeeze(-1)

def get_neighbor_idx(pc, query_pts, k):
    kdtree = KDTree(pc)
    (x, idx) = kdtree.query(query_pts, k)
    idx = torch.from_numpy(idx.astype(int)).long()

    return idx

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def PCA(pts, queries, conf):
    knn = conf.get_int('model.loss.knn_nc')
    neigh_idx = get_neighbor_idx(pts.detach().cpu().numpy(), queries.detach().cpu().numpy(), knn)  # n,k
    neigh_pts = pts[neigh_idx]  # n,k,3
    pts_dif = neigh_pts - queries.unsqueeze(1)  # n,k,3
    pts_dif_T = pts_dif.permute(0, 2, 1)  # n,3,k
    co_matrix = pts_dif_T @ pts_dif  # n,3,3
    eigs, vectors = torch.linalg.eig(co_matrix) # eigs:n,3;vectors:n,3,3
    eigs, vectors = eigs.real, vectors.real
    min_eig_index = torch.argmin(eigs, dim=1)
    normal = vectors[torch.arange(vectors.shape[0]), min_eig_index, :]
    normal = F.normalize(normal, dim=-1)

    return normal

def compute_circumsphere_centers(tetrahedra):
    A, B, C, D = tetrahedra[:, 0, :], tetrahedra[:, 1, :], tetrahedra[:, 2, :], tetrahedra[:, 3, :]  # n,3

    # Extract coordinates
    x1, y1, z1 = A[:, 0, None], A[:, 1, None], A[:, 2, None]
    x2, y2, z2 = B[:, 0, None], B[:, 1, None], B[:, 2, None]
    x3, y3, z3 = C[:, 0, None], C[:, 1, None], C[:, 2, None]
    x4, y4, z4 = D[:, 0, None], D[:, 1, None], D[:, 2, None]

    # Matrix for the left-hand side of the equation
    A_matrix = torch.cat([
        torch.cat([x2 - x1, y2 - y1, z2 - z1], dim=-1).unsqueeze(1),
        torch.cat([x3 - x1, y3 - y1, z3 - z1], dim=-1).unsqueeze(1),
        torch.cat([x4 - x1, y4 - y1, z4 - z1], dim=-1).unsqueeze(1)
    ], dim=1)

    # Vector for the right-hand side of the equation
    b_vector = 0.5 * torch.cat([
        x2 ** 2 - x1 ** 2 + y2 ** 2 - y1 ** 2 + z2 ** 2 - z1 ** 2,
        x3 ** 2 - x1 ** 2 + y3 ** 2 - y1 ** 2 + z3 ** 2 - z1 ** 2,
        x4 ** 2 - x1 ** 2 + y4 ** 2 - y1 ** 2 + z4 ** 2 - z1 ** 2
    ], dim=-1)
    center = torch.linalg.solve(A_matrix, b_vector)

    return center