'''
##############################################################
# Created Date: Thursday, May 1st 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

from typing import List, Optional
import subprocess
import os
import re
import pyufunc as pf
from loguru import logger

try:
    from .tool_util import func_prompt
except ImportError:
    from tool_util import func_prompt

from langchain_core.tools import tool


def find_executable_from_PATH_on_win(exe_name: str,
                                     sel_dir: list = None,
                                     verbose: bool = True) -> list | None:
    """Find the executable from the system PATH.

    Args:
        exe_name (str): The executable name to search for.
        ext (str): The extension of the executable. Defaults to "exe" for executable files.
        sel_dir (list): A list of directories to search for the executable. Defaults to [].
        verbose (bool): Whether to print the process info. Defaults to True.

    Returns:
        list or None: A list of full path of the executable if found, otherwise None.
    """

    # check if the executable name is a string
    if not isinstance(exe_name, str):
        raise ValueError("exe_name should be a string.")

    # Add the selected directories to the PATH environment
    if sel_dir:
        for path in sel_dir:
            if os.path.isdir(path):
                os.environ["PATH"] += os.pathsep + path
            elif verbose:
                print(f"  :The directory: {path} does not exist. Skipped.")

    # check if exe_name has the extension
    _, ext_str = os.path.splitext(exe_name)
    if not ext_str:
        if verbose:
            print(f"  :The executable: {exe_name} has no extension. Added exe as the extension.")
        exe_name = f"{exe_name}.exe"

    # get the full environment PATH
    env_paths = os.getenv("PATH").split(os.pathsep)

    res = []
    for path in env_paths:
        abs_path = pf.path2linux(os.path.join(path, exe_name))

        # check if the file exists and is executable
        if os.path.isfile(abs_path) and os.access(abs_path, os.X_OK):
            res.append(abs_path)

    if not res:
        if verbose:
            print(f"  :Could not find {exe_name} in the system PATH."
                  " Please make sure the executable is installed."
                  " please include executable extension in the name.")
        return None

    if verbose:
        print(f"  :Found {exe_name} in the system PATH:")
        for path in res:
            print(f"    :{path}")
    return res


def find_executable_from_PATH_on_linux(exe_name: str,
                                       verbose: bool = True) -> Optional[List[str]]:
    """
    Use the system `which -a` to list all matches for exe_name on Linux.

    Args:
        exe_name (str): The name of the executable to search for.
        verbose (bool): Whether to print the process info. Defaults to True.

    Example:
        >>> find_executable_from_PATH_on_linux("python3", verbose=True)
        >>> ['/usr/bin/python3', '/usr/local/bin/python3']

    Returns:
        Optional[List[str]]: A list of paths where the executable is found, or None if not found.
    """

    try:
        # -a: list all matches, not just the first
        proc = subprocess.run(
            ["which", "-a", exe_name],
            capture_output=True,
            text=True,
            check=False)
        paths = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if verbose:
            if paths:
                for p in paths:
                    print(f"[which] Found: {p}")
            else:
                print(f"[which] No matches for {exe_name}")
        return paths or None

    except FileNotFoundError:
        if verbose:
            print("[which] `which` command not found on this system.")
        return None


# Check required simulation environments
# @logger.catch
# @func_prompt(name="check_sumo_installed",
#              description="""This tool checks if SUMO (Simulation of Urban MObility) is installed on the system.
#              It searches for the SUMO executable in the system PATH and returns the version if found, if not found, it returns an empty list.
#              the output will tell user exactly SUMO version if it's installed, else, please tell user that you cannot find SUMO executable in the system PATH.
#              and recommend user to install SUMO from https://sumo.dlr.de/docs/Downloads.php
#              """)


@tool
def check_sumo_installed(sel_dir: list = None, verbose: bool = True) -> list:
    """Check if SUMO is installed on the system.
    This tool checks if SUMO (Simulation of Urban MObility) is installed on the system.
    It searches for the SUMO executable in the system PATH and returns the version if found, if not found, it returns an empty list.
    the output will tell user exactly SUMO version if it's installed, else, please tell user that you cannot find SUMO executable in the system PATH.
    and recommend user to install SUMO from https://sumo.dlr.de/docs/Downloads.php

    Args:
        ext (str): The extension of the executable. Defaults to "exe" for executable files.
        sel_dir (list): A list of directories to search for the SUMO executable. Defaults to None.
        verbose (bool): Whether to print the process info. Defaults to True.

    Raises:
        Exception: Unsupported OS, could not find SUMO executable

    Returns:
        bool or list: a list of installed versions, False otherwise.
    """

    # check the operation system
    if pf.is_windows():
        print("  :Checking SUMO installation on Windows.")
        sumo_executable = "sumo.exe"  # For Windows
        sumo_exe_lst = find_executable_from_PATH_on_win(sumo_executable, sel_dir=sel_dir, verbose=False)

    elif pf.is_linux():
        print("  :Checking SUMO installation on Linux.")
        sumo_executable = "sumo"
        sumo_exe_lst = find_executable_from_PATH_on_linux(sumo_executable, verbose=False)

    elif pf.is_mac():
        print("  :Checking SUMO installation on MacOS.")
        sumo_executable = None  # TODO: Check name of the executable
        sumo_exe_lst = []

    else:
        raise Exception("  :Unsupported OS, could not find SUMO executable.")

    if sumo_exe_lst:
        # remove duplicates
        sumo_exe_lst = list(set(sumo_exe_lst))

        # print out the version of SUMO if more than one path is found
        if len(sumo_exe_lst) > 1:
            print("  :Multiple SUMO executables found in the system PATH:")

        # run SUMO to check the version
        version_lst = []
        for exe_path in sumo_exe_lst:
            try:
                version_check = subprocess.run([exe_path],
                                               capture_output=True,
                                               text=True,
                                               check=True)
                # find the version number from the output
                if version_check.returncode == 0:

                    # Define the pattern to match the version number
                    pattern = r'Version (\d+\.\d+\.\d+)'

                    # Search for the pattern in the text
                    match = re.search(pattern, version_check.stdout)

                    # Extract and print the version number if a match is found
                    if match:
                        version = match.group(1)
                        version_lst.append(str(version))
                        # version_lst.append(f"{version} found: {exe_path}")
                        print(f"  :SUMO version: {version} found: '{exe_path}'")
                    else:
                        pass

            except Exception as e:
                if verbose:
                    print(f"  :Error running SUMO: {e}")
        if version_lst:
            return list(set(version_lst))

    print("  :SUMO not found in the system PATH.")
    return []


def check_sumo_installed_NotTool(sel_dir: list = None, verbose: bool = True) -> list:
    """Check if SUMO is installed on the system.
    This tool checks if SUMO (Simulation of Urban MObility) is installed on the system.
    It searches for the SUMO executable in the system PATH and returns the version if found, if not found, it returns an empty list.
    the output will tell user exactly SUMO version if it's installed, else, please tell user that you cannot find SUMO executable in the system PATH.
    and recommend user to install SUMO from https://sumo.dlr.de/docs/Downloads.php

    Args:
        ext (str): The extension of the executable. Defaults to "exe" for executable files.
        sel_dir (list): A list of directories to search for the SUMO executable. Defaults to None.
        verbose (bool): Whether to print the process info. Defaults to True.

    Raises:
        Exception: Unsupported OS, could not find SUMO executable

    Returns:
        bool or list: a list of installed versions, False otherwise.
    """

    # check the operation system
    if pf.is_windows():
        print("  :Checking SUMO installation on Windows.")
        sumo_executable = "sumo.exe"  # For Windows
        sumo_exe_lst = find_executable_from_PATH_on_win(sumo_executable, sel_dir=sel_dir, verbose=False)

    elif pf.is_linux():
        print("  :Checking SUMO installation on Linux.")
        sumo_executable = "sumo"
        sumo_exe_lst = find_executable_from_PATH_on_linux(sumo_executable, verbose=False)

    elif pf.is_mac():
        print("  :Checking SUMO installation on MacOS.")
        sumo_executable = None  # TODO: Check name of the executable
        sumo_exe_lst = []

    else:
        raise Exception("  :Unsupported OS, could not find SUMO executable.")

    if sumo_exe_lst:
        # remove duplicates
        sumo_exe_lst = list(set(sumo_exe_lst))

        # print out the version of SUMO if more than one path is found
        if len(sumo_exe_lst) > 1:
            print("  :Multiple SUMO executables found in the system PATH:")

        # run SUMO to check the version
        version_lst = []
        for exe_path in sumo_exe_lst:
            try:
                version_check = subprocess.run([exe_path],
                                               capture_output=True,
                                               text=True,
                                               check=True)
                # find the version number from the output
                if version_check.returncode == 0:

                    # Define the pattern to match the version number
                    pattern = r'Version (\d+\.\d+\.\d+)'

                    # Search for the pattern in the text
                    match = re.search(pattern, version_check.stdout)

                    # Extract and print the version number if a match is found
                    if match:
                        version = match.group(1)
                        version_lst.append(str(version))
                        print(f"  :SUMO version: {version} found: {exe_path}")
                    else:
                        pass

            except Exception as e:
                if verbose:
                    print(f"  :Error running SUMO: {e}")
        if version_lst:
            return list(set(version_lst))

    print("  :SUMO not found in the system PATH.")
    return []
