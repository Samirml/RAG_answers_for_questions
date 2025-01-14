import json


def load_config(config_path="config.json"):
    """
    Loads the configuration settings from a JSON file.

    Args:
        config_path (str): The path to the JSON configuration file. Defaults to "config.json".

    Returns:
        dict: A dictionary containing the configuration settings.
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config
