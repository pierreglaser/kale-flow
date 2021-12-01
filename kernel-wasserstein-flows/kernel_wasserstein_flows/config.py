import logging


LOG_LEVELS = {
    "gradient_flow": logging.WARNING,
    "kale.optim.newton": logging.WARNING,
    "kale.optim.online_cd": logging.WARNING,
    "kale.optim.cd": logging.WARNING,
    "kale.line_search": logging.WARNING,
    "mmd.kernel_optimization": logging.WARNING,
    "applications.color_transfer": logging.WARNING
}


def get_logger(name):
    logger = logging.getLogger(name)

    try:
        from distributed import get_worker

        worker = get_worker()
        worker_id_slot = str(worker.id)[:15] + ": "
    except (ImportError, ValueError):
        worker_id_slot = ""

    # formatter = logging.Formatter(
    #     f"{worker_id_slot[:15]}: %(name)-2s: %(message)s"
    # )
    formatter = logging.Formatter(f"{worker_id_slot} %(name)s: %(message)s")

    logger.handlers[:] = []
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(LOG_LEVELS[name])
    return logger
