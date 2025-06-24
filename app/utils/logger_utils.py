import logging

def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a logger with a standardized formatter.
    Args: name (str): Name of the logger (usually __name__ of calling module)
    Returns:logging.Logger: Configured logger with INFO level and console output
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(name)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
