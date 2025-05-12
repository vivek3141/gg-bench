from pathlib import Path
import os


def find_config_file(
    *,
    start_path: str = os.getcwd(),
    config_file: str,
    exception_to_raise: Exception,
    exception_message: str,
) -> str:
    """
    Searches for a configuration file starting from a given directory and moving up the directory tree.

    Args:
        start_path (str): The directory path to start the search from.
        config_file (str): The name of the configuration file to search for.
        exception_to_raise (Exception): The exception to raise if the configuration file is not found.
        exception_message (str): The message to include with the raised exception.

    Returns:
        str: The absolute path to the configuration file if found.

    Raises:
        exception_to_raise: If the configuration file is not found.
    """
    path = Path(start_path).resolve()

    while path != Path("/") and not (path / config_file).exists():
        if (path / ".git").exists():
            break
        path = path.parent

    config_path = path / config_file
    if not config_path.exists():
        raise exception_to_raise(exception_message)

    return str(config_path)
