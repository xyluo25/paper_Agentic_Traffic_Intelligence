'''
##############################################################
# Created Date: Tuesday, June 17th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''


from pathlib import Path
from typing import Optional, Dict
import pyufunc as pf
from langchain_core.tools import tool
import yaml
import shutil


config_file = Path(__file__).parent.parent / "proj_config" / "realtwin_config.yaml"
# config_file = pf.path2linux(config_file)

with open(config_file, 'r') as config_file:
    config_content = yaml.safe_load(config_file)


@tool
def realtwin_show_default_config(input_msg: str = "") -> str:
    """Displays the contents of the default configuration file located at `./proj_config/realtwin_config_default.yaml`.
    This tool reads and returns the configuration settings in a dictionary format,
    allowing users to view the current configuration values.

    Args:
        input_msg (str, optional): Reserved for future use. Not required for viewing the config.

    Returns:
        str: A string representation of the configuration dictionary, or an message if the file not found.
    """

    config_file = Path(__file__).parent.parent / \
        "proj_config" / "realtwin_config_default.yaml"

    with open(config_file, 'r') as config_file:
        config_content = yaml.safe_load(config_file)
    if not config_content:
        return "Config file not found."
    return f"Real-Twin default configuration (Show configuration (all records) in Python code with indentation and line breaks): {config_content}"


@tool
def realtwin_show_config(input_msg: str = "") -> str:
    """Displays the contents of the default configuration file located at `./proj_config/realtwin_config.yaml`.
    This tool reads and returns the configuration settings in a dictionary format,
    allowing users to view the current configuration values.

    Args:
        input_msg (str, optional): Reserved for future use. Not required for viewing the config.

    Returns:
        str: A string representation of the configuration dictionary, or an message if the file not found.
    """

    config_file = Path(__file__).parent.parent / "proj_config" / "realtwin_config.yaml"

    with open(config_file, 'r') as config_file:
        config_content = yaml.safe_load(config_file)
    if not config_content:
        return "Config file not found."
    return f"Real-Twin configuration (Show configuration (all records) in Python code with indentation and line breaks): {config_content}"


def update_nested_dict(a: dict, b: dict) -> None:
    """
    Recursively updates dict `a` in-place using values from dict `b`:
      - If a key k exists in b at this level:
          * and both a[k] and b[k] are dicts, recurse into them
          * otherwise, replace a[k] with b[k]
      - For any other dict-valued entries in a, recurse with the same b
    Keys in b that aren't in a are ignored.
    """
    def _recurse(d: dict, b_sub: dict):
        for k, v in d.items():
            if k in b_sub:
                # if both sides are dict: dive deeper
                if isinstance(v, dict) and isinstance(b_sub[k], dict):
                    _recurse(v, b_sub[k])
                else:
                    d[k] = b_sub[k]
            # even if k not in b_sub, still recurse into nested dicts
            elif isinstance(v, dict):
                _recurse(v, b_sub)

    _recurse(a, b)


@tool
def realtwin_edit_config(input_msg: Optional[Dict] = None) -> str:
    """
    Edits the default configuration file at './proj_config/realtwin_config.yaml'.
    It extracts keys and values from the provided dictionary and updates the config file accordingly.
    Please ensure that the input is a dictionary with valid keys and values, and NO nested dictionaries.
    Please make sure show dictionary/json output in pretty format, such as using indentation and line breaks.

    Args:
        input_msg (dict): A dictionary of configuration settings to update.

    Returns:
        str: Message indicating success or failure of the update and showing updated config in dictionary format.
    """

    print(f"edit_config input_msg: {input_msg}")

    if not isinstance(input_msg, dict):
        return "Input message is not in the correct format, please provide a dictionary."

    config_path = Path(__file__).parent.parent / "proj_config" / "realtwin_config.yaml"

    try:
        with open(config_path, 'r') as f:
            config_content = yaml.safe_load(f)

        if not isinstance(config_content, dict):
            # use the backup config content
            config_path_backup = Path(__file__).parent.parent / "proj_config" / "realtwin_config_backup.yaml"
            with open(config_path_backup, 'r') as f_backup:
                config_content = yaml.safe_load(f_backup)

        if not isinstance(config_content, dict):
            config_content = {}

        # update the config content with the input message
        update_nested_dict(config_content, input_msg)

        with open(config_path, 'w') as f:
            yaml.safe_dump(config_content, f, default_flow_style=False, sort_keys=False)

        return f"Config file updated. Show configuration dictionary (all records) in Python code with indentation and line breaks: {config_content}"
    except Exception as e:
        return f"Failed to update config file: {e}"


@tool
def realtwin_save_config(input_msg: str = None) -> str:
    """This tool saves the current configuration to the folder that user can modify.

    Args:
        input_msg (dict, optional): A dictionary of configuration settings to update. Not used in this function.

    Returns:
        str: Message indicating success or failure of the save operation. You must return exactly result as a string.
    """
    try:
        config_file = Path(__file__).parent.parent / "proj_config" / "realtwin_config.yaml"
        user_dir = Path(__file__).parent.parent / "proj_tmp_output" / "RealTwin_User"

        print("config_file, user_dir:", config_file, user_dir)

        if not user_dir.exists():
            user_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(config_file, user_dir)

        return f"Config file saved to {Path(user_dir) / "realtwin_config.yaml"}."

    except Exception as e:
        print(f"Error saving config file: {e}")
        return f"Failed to save config file: {e}"
