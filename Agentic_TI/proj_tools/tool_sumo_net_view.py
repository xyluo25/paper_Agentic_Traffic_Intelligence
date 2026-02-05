'''
##############################################################
# Created Date: Tuesday, June 17th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

import shutil
from pathlib import Path
import subprocess
from xml.dom import minidom
import xml.etree.ElementTree as ET
import pyufunc as pf
from langchain_core.tools import tool

path_net_example = Path(__file__).parent.parent / "datasets" / "example2" / "output/SUMO/chat.net.xml"


def xml_prettify(element: str) -> str:
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(element, 'utf-8')
    re_parsed = minidom.parseString(rough_string)
    return re_parsed.toprettyxml(indent="    ")


def create_sumo_gui_view_settings_xml(img_name: str = "sumo_net_view.png",
                                      img_dir: str = None,
                                      output_dir: str = None) -> bool:

    if not img_dir:
        img_dir = Path.cwd()

    if not output_dir:
        output_dir = Path.cwd()

    root = ET.Element("viewsettings")

    # scene
    scene = ET.SubElement(root, "scheme")
    scene.set("name", "real world")  # standard, real world, faster standard, rail, selection

    # add new line in xml for better readability
    root.tail = "\n"

    # delay
    delay = ET.SubElement(root, "delay")
    delay.set("value", "12")  # seconds

    root.tail = "\n"

    # add snapshot
    snapshot = ET.SubElement(root, "snapshot")
    snapshot.set("file", img_name)
    snapshot.set("time", "0")

    # save to viewsettings.xml
    xml_str = xml_prettify(root)
    output_viewsettings = output_dir / "viewsettings.xml"
    with open(output_viewsettings, "w") as f:
        f.write(xml_str)
    return True


@tool
def sumo_net_snapshot(path_net: str = "", path_viewsettings: str = None) -> str:
    """This tool is used to take the snapshot of the sumo network file (.net.xml) using sumo-gui. The user input should include the path to the .net.xml file. Optionally, the user can also provide a custom viewsettings.xml file path. If not provided, a default viewsettings.xml located in the proj_config directory will be used. The tool will return the path to the generated snapshot image (sumo_net_view.png). You must show the png to the user, not the png path or link.

    Args:
        path_net (str): the path to the .net.xml file.
        path_viewsettings (str, optional): the path to the user defined viewsettings.xml file. Defaults to None.

    Returns:
        str: the path to the generated snapshot image (sumo_net_view.png) and show the png to the user in raw, not the png path or link.
    """
    print(f"sumo_net_snapshot path_net: {path_net}")

    if not path_net:
        path_net = Path(path_net_example)

    if not Path(path_net).is_file():
        path_net = Path(path_net_example)

    if not path_viewsettings:
        path_viewsettings = Path(__file__).parent.parent / "proj_config" / "viewsettings.xml"

    cmd = f'cmd /c "sumo-gui -n {path_net} --gui-settings-file {pf.path2linux(path_viewsettings)} --start true --quit-on-end true"'

    process = subprocess.run(cmd, shell=True)

    path_snapshot = Path(path_viewsettings).parent / "sumo_net_view.png"

    path_tmp_snapshot = Path(__file__).parent.parent / "proj_tmp_output" / "sumo_net_view.png"

    shutil.copy(path_snapshot, path_tmp_snapshot)

    return f"Snapshot image created at: {path_tmp_snapshot}. Please show the png to the user in raw, not the png path or link."

#
# path_viewsettings = Path(__file__).parent.parent / \
#     "proj_config" / "viewsettings.xml"
#
# cmd = f'cmd /c "sumo-gui -n {path_net} --gui-settings-file {pf.path2linux(path_viewsettings)} --start true --quit-on-end true"'
#
# process = subprocess.run(cmd, shell=True)
