'''
##############################################################
# Created Date: Thursday, May 1st 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''


import urllib.request
import shutil
import zipfile
from typing import List, Optional
import subprocess
import os
import re
import pyufunc as pf
from loguru import logger
from .tool_util import func_prompt
from .tool_sumo_check import check_sumo_installed_NotTool

from langchain_core.tools import tool


def download_single_file_from_web(url: str, dest_filename: str, chunk_size=1024) -> bool:
    """
    Downloads a large file from a URL in chunks and saves it to the specified destination.

    Args:
        url (str): The URL of the file to download.
        dest_filename (str): filename or path to the filename to save the downloaded file.
        chunk_size (int): Size of each chunk to read in bytes (default: 1024).

    Returns:
        bool: True if the download is successful, False otherwise.
    """
    try:
        with urllib.request.urlopen(url) as response, open(dest_filename, 'wb') as out_file:
            total_size = int(response.getheader('Content-Length', 0))

            if total_size == 0:
                print("  :An error occurred: File size is 0.")
                return False

            downloaded = 0

            print(f"  :Starting download: {url}")
            # print(f"  :Total size: {total_size / (1024 * 1024):.2f} MB")

            while chunk := response.read(chunk_size):
                out_file.write(chunk)
                downloaded += len(chunk)
                print(f"\r  :Downloaded: {downloaded / (1024 * 1024):.2f} / {total_size / (1024 * 1024):.2f}  MB",
                      end="")

    except Exception as e:
        print(f"  :An error occurred: {e}")
        return False
    return True


# @logger.catch
# @func_prompt(name="install_sumo",
#              description="""This tool is used to install the SUMO simulator on your system.
#              Consider using this tool when asked to install the SUMO simulator.
#              The output will tell whether you have finished this command successfully, if output is True,
#              it means SUMO is installed successfully, if output is False, it means SUMO installation failed.
#              Please tell user that SUMO is installed successfully,
#              or failed to install SUMO and recommend user to install SUMO manually.
#              """)


@tool
def install_sumo(input_msg: str = "",
                 sel_dir: list = None,
                 strict_sumo_version: str = "1.21.0",
                 verbose: bool = True, **kwargs) -> str:
    """This tool is used to install the SUMO simulator on your system.
    Consider using this tool when asked to install the SUMO simulator.
    The output will tell whether you have finished this command successfully,
    if successfully run, the output is exactly sumo version installed,
    if not, tell the user that SUMO installation failed with the error message.
    Please note, you need to extract strict_sumo_version from input message,
    if not extracted, use None as default value.
    Please tell user that SUMO is installed successfully,
    or failed to install SUMO and recommend user to install SUMO manually.

    Args:
        sumo_version (str): The version of SUMO to be installed. Default is "1.21.0".
        sel_dir (list): A list of directories to search for the SUMO executable. Defaults to None.
        strict_sumo_version (bool): If True, check and install the exact version of SUMO. Default is False.
        verbose (bool): If True, print the installation process. Default is True.
        kwargs: Additional keyword arguments.

    Returns:
        str: A message indicating the installation status of SUMO.
    """

    print(f"  :Installing SUMO {strict_sumo_version}...")
    print(f" {input_msg}, sel_dir: {sel_dir}, strict_sumo_version: {strict_sumo_version}")

    # Check if SUMO is already installed
    version_lst = check_sumo_installed_NotTool(sel_dir=sel_dir, verbose=verbose)
    print(f"  :SUMO version list: {version_lst}")
    if version_lst:
        # Check if the exact version of SUMO is installed
        if strict_sumo_version is None or strict_sumo_version in version_lst:
            return (f"  :SUMO is already installed, available versions: {version_lst}")

    # If SUMO not installed,
    # Or strict_sumo_version is True and the version is not installed
    # Install the SUMO
    if pf.is_windows():
        return install_sumo_windows(strict_sumo_version, verbose=verbose)

    if pf.is_linux():
        return install_sumo_linux(verbose=verbose)

    if pf.is_mac():
        return ("  :Error: MacOS is not supported yet.")

    return ("  :Error: Unsupported operating system.")


def install_sumo_windows(sumo_version: str = "1.21.0", verbose: bool = True) -> bool:
    """Install SUMO onto the windows system.

    Args:
        sumo_version (str): The version of SUMO to be installed. Default is "1.21.0".
        verbose (bool): If True, print the installation process. Default is True.

    Returns:
        bool: True if the SUMO is installed successfully, False otherwise
    """

    print(f"  :Installing SUMO {sumo_version} for Windows...")

    # Download SUMO from the official website
    sumo_release_url = "https://sumo.dlr.de/releases/"
    sumo_version_win = f"sumo-win64-{sumo_version}.zip"

    download_path = pf.path2linux(os.path.join(os.getcwd(), "sumo.zip"))
    extract_path = pf.path2linux(os.path.join(os.getcwd(), "SUMO"))

    # Download the SUMO zip file from the official website
    sumo_zip_url = f"{sumo_release_url}{sumo_version}/{sumo_version_win}"
    id_download = download_single_file_from_web(sumo_zip_url, download_path)

    if not id_download:
        return False

    # Extract the SUMO zip file
    if verbose:
        print(
            f"  :Extracting SUMO {sumo_version} for Windows at: {extract_path}...")

    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Clean up the downloaded zip file
    os.remove(download_path)

    # check if SUMO bin folder exists
    sumo_bin_path = os.path.join(extract_path, f"sumo-{sumo_version}", "bin")
    sumo_bin_path = pf.path2linux(sumo_bin_path)
    if not os.path.exists(sumo_bin_path):
        print(
            f"  :Error: bin folder not found in extracted SUMO directory: {sumo_bin_path}")
        return False

    # Add the SUMO bin folder to the system PATH
    if sumo_bin_path not in os.environ['PATH']:

        os.environ["PATH"] += os.pathsep + \
            sumo_bin_path[0].upper() + sumo_bin_path[1:]
        os.environ["PATH"] += os.pathsep + \
            sumo_bin_path[0].lower() + sumo_bin_path[1:]

        # add_path = subprocess.run(["setx", "PATH", f"%PATH%;{sumo_bin_path}"], shell=True, check=True)
        # if add_path.returncode == 0:
        #     print("  :SUMO is installed successfully.")
        # else:
        #     print("  :Error: Failed to add SUMO bin folder to system PATH.")

    return True


def install_sumo_linux(verbose: bool = True) -> bool:
    """Install SUMO onto the linux system.

    Args:
        sumo_version (str): The version of SUMO to be installed. Default is "1.21.0".
        verbose (bool): If True, print the installation process. Default is True.

    Note:
        The installation of SUMO on Linux is not supported yet.
        upcoming feature.
    Returns:
        bool: True if the SUMO is installed successfully on Linux, False otherwise
    """
    # detect package manager
    pkg_mgr = None
    if shutil.which("apt-get"):
        pkg_mgr = "apt-get"
    elif shutil.which("yum"):
        pkg_mgr = "yum"
    else:
        if verbose:
            print("[install_sumo] No supported package manager found (apt-get or yum).")
        return False

    try:
        # update repos
        cmd_update = ["sudo", pkg_mgr, "update",
                      "-y"] if pkg_mgr == "yum" else ["sudo", pkg_mgr, "update"]
        if verbose:
            print(f"[install_sumo] Running: {' '.join(cmd_update)}")
        subprocess.run(cmd_update, check=True)

        # install SUMO package (version pinning may not be supported by all repos)
        install_pkgs = ["sumo", "sumo-tools"]
        cmd_install = ["sudo", pkg_mgr, "install", "-y"] + install_pkgs
        if verbose:
            print(f"[install_sumo] Running: {' '.join(cmd_install)}")
        subprocess.run(cmd_install, check=True)

        if verbose:
            print("[install_sumo] SUMO installed successfully via package manager.")
        return True

    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"[install_sumo] Installation command failed: {e}")
        return False
    except Exception as e:
        if verbose:
            print(f"[install_sumo] Unexpected error: {e}")
        return False
