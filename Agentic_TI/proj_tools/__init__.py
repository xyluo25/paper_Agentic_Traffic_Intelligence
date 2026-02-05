'''
##############################################################
# Created Date: Wednesday, April 30th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

from .tool_osm_find import get_place_info
from .tool_osm_download import get_osm_from_relation_id
from .tool_osm_download_web import get_osm_from_web
from .tool_osm_vis import vis_osm

from .tool_sumo_check import check_sumo_installed
from .tool_sumo_install import install_sumo
from .tool_sumo_net_view import sumo_net_snapshot

from .tool_rt_edit_config import realtwin_show_config, realtwin_edit_config, realtwin_save_config, realtwin_show_default_config
from .tool_rt_sample import realtwin_sample_run
from .tool_realtwin import realtwin_inputs_generation, realtwin_simulation

sumo_tools = [
    check_sumo_installed,
    install_sumo,
    sumo_net_snapshot,
]
osm_tools = [
    get_place_info,
    get_osm_from_relation_id,
    get_osm_from_web,
    vis_osm,
]
realtwin_tools = [
    realtwin_show_config,
    realtwin_edit_config,
    realtwin_save_config,
    realtwin_sample_run,
    realtwin_inputs_generation,
    realtwin_simulation,
    realtwin_show_default_config,
]
rag_tools = []

usr_defined_tools = {"sumo_tools": sumo_tools,
                     "osm_tools": osm_tools,
                     "realtwin_tools": realtwin_tools,
                     "rag_tools": rag_tools}

HIL_Tools = [
    # SUMO Tools
    "install_sumo",

    # OSM Tools
    "get_place_info",
    "get_osm_from_relation_id",

    # RealTwin Tools
    # "realtwin_sample_run",
    "realtwin_inputs_generation",
    "realtwin_simulation",

    # RAG Tools
]
