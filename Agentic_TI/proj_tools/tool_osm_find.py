'''
##############################################################
# Created Date: Wednesday, April 30th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''


from __future__ import absolute_import
from loguru import logger
import pyufunc as pf

from langchain_core.tools import tool


@tool
def get_place_info(place_name: str) -> tuple:
    """This tool is used to get the information of a given place from OpenStreetMap.
    Consider using this tool when asked to get the bounding box of the location.
    The output will tell whether you have finished this command successfully, and return detailed information.
    Successful output must be in dictionary/json format."""
    try:
        place_dict = pf.get_osm_place(place_name,)
        if place_dict is None:
            return "Failed to get the bounding box of the place."
        return ("After you have successfully run the tool, "
                f"you MUST return result in dictionary/json format: {place_dict}")
    except Exception as e:
        logger.error(f"Error getting bounding box for {place_name}: {e}")
        return (f"Error getting bounding box for {place_name}: {e}")
