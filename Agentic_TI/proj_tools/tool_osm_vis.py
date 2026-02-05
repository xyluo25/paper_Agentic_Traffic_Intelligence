'''
##############################################################
# Created Date: Tuesday, July 15th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

from pathlib import Path
import pyufunc as pf
import osm2gmns as og
import plot4gmns as pg
import grid2demand as gd
from langchain_core.tools import tool

path_tmp_output = Path(__file__).parent.parent / 'proj_tmp_output'


def convert_osm_to_gmns(dir_osm: str = "") -> bool:
    """Convert OSM data to NODE, LINK AND POI CSV."""

    if not dir_osm:
        dir_osm = path_tmp_output

    if not Path(dir_osm).exists():
        return False

    # check end with .osm
    path_osm = dir_osm / "map.osm"
    if not path_osm.exists():
        return False

    net = og.getNetFromFile(str(path_osm), POI=True, mode_types=["auto", "bike", "walk", "railway"])
    og.outputNetToCSV(net, output_folder=str(path_tmp_output))
    return True


@tool
def vis_osm(dir_osm: str = "") -> str:
    """This tool is used to visualize OpenStreetMap data. Consider using this tool when asked to visualize the OSM data.
    The output will return the paths to the generated images and HTML map file. And you need to check these files to get the visualization results, including the HTML map file.

    Args:
        dir_osm (str): Directory containing OSM data files. If not specified, defaults to 'proj_tmp_output'.

    Returns:
        str: Comma-separated list of paths to generated images and HTML map file.
    """

    print(f"Visualizing OSM data from path: {dir_osm}...")

    if not dir_osm:
        dir_osm = path_tmp_output

    if dir_osm is None:
        dir_osm = path_tmp_output

    dir_osm = Path(dir_osm)
    if not dir_osm.exists():
        dir_osm = path_tmp_output

    # generate node, link and poi csv files from the OSM data
    check_cvt = convert_osm_to_gmns(dir_osm=dir_osm)
    if not check_cvt:
        return f"Could not convert OSM data to GMNS format. Please check {dir_osm} folder."

    # demand generation
    try:
        net = gd.GRID2DEMAND(input_dir=pf.path2linux(dir_osm), output_dir=path_tmp_output)
        net.load_network()
        net.net2grid(num_x_blocks=10, num_y_blocks=10)
        net.taz2zone()
        net.map_zone_node_poi()
        net.calc_zone_od_distance()
        net.run_gravity_model()
        net.save_results_to_csv(output_dir=path_tmp_output, demand=True, zone=True, overwrite_file=True)
    except Exception:
        print("Demand generation failed. Skipping demand visualization.")

    # visualize the OSM data
    mnet = pg.generate_multi_network_from_csv(input_dir=pf.path2linux(dir_osm),
                                              output_dir=path_tmp_output)

    _ = pg.show_network_by_modes(mnet=mnet, modes=['auto'],
                                 output_dir=path_tmp_output)

    _ = pg.show_network_by_link_types(mnet=mnet, link_types=['motorway', 'primary', 'secondary'],
                                      output_dir=path_tmp_output)

    _ = pg.show_network_by_link_length(mnet=mnet, min_length=10, max_length=50,
                                       output_dir=path_tmp_output)

    _ = pg.show_network_by_link_free_speed(mnet=mnet, min_free_speed=10, max_free_speed=40,
                                           output_dir=path_tmp_output)

    _ = pg.show_network_by_link_lanes(mnet=mnet, min_lanes=2, max_lanes=4,
                                      output_dir=path_tmp_output)

    _ = pg.show_network_by_poi_types(mnet=mnet, poi_type=['apartments', 'industrial'],
                                     output_dir=path_tmp_output)
    try:
        _ = pg.show_network_demand_matrix_heatmap(mnet=mnet, output_dir=path_tmp_output)
    except Exception:
        print("Demand matrix heatmap generation failed. Skipping demand matrix visualization.")
    # return the list of generated images

    html_map = path_tmp_output / "p4g_fig_results/plot4gmns_vis_map.html"

    png_list = list((path_tmp_output / "p4g_fig_results").glob("*.png"))

    png_list_str = [str(png) for png in png_list]

    return "Generated OSM figures are: " + ",".join(png_list_str) + f", {html_map}"
