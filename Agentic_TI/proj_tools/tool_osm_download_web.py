'''
##############################################################
# Created Date: Thursday, May 1st 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

import subprocess
import os
import sys
from pathlib import Path
import pyufunc as pf
from loguru import logger
import pyufunc as pf

from .tool_util import func_prompt

try:
    from .tool_sumo_check import check_sumo_installed_NotTool
except Exception:
    from tool_sumo_check import check_sumo_installed_NotTool
from langchain_core.tools import tool


# @logger.catch
# @func_prompt(name="get_osm_from_web",
#              description="""This tool is used to open osm WebWizard.
#              Consider using this tool when asked to get osm data from the web.
#              The output will tell whether you have finished this command successfully.""")


@tool
def get_osm_from_web(input_msg: str) -> str:
    """This tool is used to open osm WebWizard.
    Consider using this tool when asked to get osm data from the web.
    The output will tell whether you have finished this command successfully. """

    # Check if SUMO is installed
    if not check_sumo_installed_NotTool():
        return ("  :SUMO is not installed. Please install SUMO first.")

    # use sumo to download the network from the bounding box
    # print("input:", input_msg)

    # check if osmGet.py exists in the current os
    if pf.is_windows():
        res_get = subprocess.run(["where", "-a", "osmWebWizard.py"], check=False, capture_output=True, text=True)
    elif pf.is_linux() or pf.is_macos():
        res_get = subprocess.run(["which", "-a", "osmWebWizard.py"], check=False, capture_output=True, text=True)

    # print("res_get:", res_get)

    # get path of osmGet.py
    if res_get.stdout:
        osmGet_path = res_get.stdout.strip()
        print("osmGet_path:", osmGet_path)
        path_sumo_tools = Path(osmGet_path).parent

        # add sumo tools to the system PATH
        if str(path_sumo_tools) not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + str(path_sumo_tools)
            print(f"  :Added SUMO tools path to the system PATH: {path_sumo_tools}")
        sys.path.append(str(path_sumo_tools))
    else:
        return ("  :osmWebWizard not found in the current OS. Please check your SUMO installation.")

    # download the network from the bounding box
    if pf.is_windows():
        cmd_get = ["python", "osmWebWizard.py",
                   f"-o {pf.path2linux(Path(__file__).parent.parent / 'datasets/')}",
                   #    "--remote",
                   ]

    elif pf.is_linux() or pf.is_macos():
        cmd_get = ["python3", "osmWebWizard.py",
                   f" -o {pf.path2linux(Path(__file__).parent.parent / '/datasets/')}",
                   #    "--remote",
                   ]
    print("cmd_get:", cmd_get)
    os.chdir(path_sumo_tools)
    res_download = subprocess.run(cmd_get, check=True, capture_output=True, text=True,)

    if res_download.returncode != 0:
        return (f"  :Failed to download network: {res_download.stderr.strip()}")

    print("res_download:", res_download)
    return (f"  :Network downloaded successfully: {res_download.stdout.strip()}")


if __name__ == "__main__":

    success = get_osm_from_web()
    if success:
        print("Network downloaded successfully.")
    else:
        print("Failed to download network.")