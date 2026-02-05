'''
##############################################################
# Created Date: Thursday, May 1st 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

import os
import http.client as httplib
import urllib.parse as urlparse
import base64
from pathlib import Path
import pyufunc as pf
from langchain_core.tools import tool


_url = "www.overpass-api.de/api/interpreter"
# alternatives: overpass.kumi.systems/api/interpreter, sumo.dlr.de/osm/api/interpreter

path_osm_map = Path(__file__).parent.parent / 'proj_tmp_output/map.osm'


def _readCompressed(conn, urlpath, query, filename):
    conn.request("POST", f"/{urlpath}", f"""
    <osm-script timeout="240" element-limit="1073741824">
    <union>
       {query}
       <recurse type="node-relation" into="rels"/>
       <recurse type="node-way"/>
       <recurse type="way-relation"/>
    </union>
    <union>
       <item/>
       <recurse type="way-node"/>
    </union>
    <print mode="body"/>
    </osm-script>""")
    response = conn.getresponse()
    # print(response.status, response.reason)
    if response.status == 200:
        print('valid responses got from API server.')
        print('receiving data...')
        with open(filename, "wb") as out:
            out.write(response.read())
        return (f'map data has been written to {filename}')
    else:
        return (f"Error: {response.status} {response.reason}"
                "Please check the relation id and try again.")


@tool
def get_osm_from_relation_id(relation_id, output_filepath=path_osm_map, url=_url) -> str:
    """Downloads OpenStreetMap (OSM) data for a specified region using the Overpass API.

    This tool is used to download OpenStreetMap (OSM) data for a specified region using the Overpass API.
    Consider using this tool when asked to download OSM data for a specific relation id.
    please note, the relation id must be integer value, and you have the ability to get the relation id from user and convert it to integer.
    if the output_filepath is not specified, it will be saved to 'datasets/map.osm'.
    The output will tell whether you have finished this command successfully.

    Args:
        relation_id (int): The OSM relation ID for the area to be queried.
        output_filepath (str): The path where the downloaded OSM data will be saved.
            Defaults to 'map.osm'.
        url (str): The URL of the Overpass API endpoint. Defaults to 'www.overpass-api.de/api/interpreter'.

    See Also:
        SUMO Documentation: https://github.com/eclipse-sumo/sumo/blob/main/tools/osmGet.py
        osm2gmns Documentation: https://github.com/jiawlu/OSM2GMNS/blob/main/osm2gmns/downloader.py

    Example:
        >>> import pyufunc as pf
        >>> relation_id = 196150  # Example relation ID (Get this from OSM)
        >>> pf.get_osm_by_relation_id(relation_id, output_filepath='my_map.osm')

    Returns:
        bool: True if the data was successfully downloaded and saved, False otherwise.
    """

    if not isinstance(relation_id, int):
        relation_id = int(relation_id)

    file_name, file_extension = os.path.splitext(output_filepath)
    if not file_extension:
        print(f'WARNING: no file extension in output_filepath {output_filepath}, '
              f'output_filepath is changed to {file_name}.osm')
        output_filepath = f'{file_name}.osm'
    elif file_extension not in ['.osm', '.xml']:
        print(f'WARNING: the file extension in output_filepath {output_filepath} is not supported, '
              f'output_filepath is changed to {file_name}.osm')
        output_filepath = f'{file_name}.osm'

    if "http" in url:
        url = urlparse.urlparse(url)
    else:
        url = urlparse.urlparse(f"https://{url}")
    if os.environ.get("https_proxy") is not None:
        headers = {}
        proxy_url = urlparse.urlparse(os.environ.get("https_proxy"))
        if proxy_url.username and proxy_url.password:
            auth = f'{proxy_url.username}:{proxy_url.password}'
            headers['Proxy-Authorization'] = f'Basic {base64.b64encode(auth)}'
        conn = httplib.HTTPSConnection(proxy_url.hostname, proxy_url.port)
        conn.set_tunnel(url.hostname, 443, headers)
    else:
        if url.scheme == "https":
            conn = httplib.HTTPSConnection(url.hostname, url.port)
        else:
            conn = httplib.HTTPConnection(url.hostname, url.port)

    if relation_id < 3600000000:
        relation_id += 3600000000
    return_message = _readCompressed(
        conn, url.path, f'<area-query ref="{relation_id}"/>', output_filepath)
    conn.close()
    return return_message
