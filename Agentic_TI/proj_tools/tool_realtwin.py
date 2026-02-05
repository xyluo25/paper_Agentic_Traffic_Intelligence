'''
##############################################################
# Created Date: Tuesday, June 24th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

from __future__ import absolute_import
from pathlib import Path
import os
import sys

from langchain_core.tools import tool
import pyufunc as pf

try:
    import realtwin as rt
except ImportError:

    # Add current directory to sys.path
    current_dir = Path(__file__).parent.resolve()
    sys.path.append(str(current_dir))

    import realtwin as rt


@tool
def realtwin_inputs_generation(input_msg: str = "") -> str:
    """Load the RealTwin configuration and environment checking.

    Args:
        input_msg (str, optional): the path to the RealTwin configuration file. If not provided, it will use the default configuration file.

    Returns:
        str: A message indicating the success of the inputs generation and tell the user where the generated inputs are saved.
    """

    # check if input_msg is a path with .yaml extension
    if input_msg and Path(input_msg).suffix == '.yaml':
        config_path = Path(input_msg)
    else:
        # If no valid path is provided, use the default configuration file
        config_path = Path(__file__).parent.parent / "proj_config" / "realtwin_config.yaml"

    twin = rt.RealTwin(input_config_file=pf.path2linux(config_path), input_confirm=False)

    # check if the SUMO simulator is installed
    twin.env_setup(sel_sim=["SUMO"])

    # Get the dir of the user input
    input_dir = Path(twin.input_config["input_dir"])
    updated_net = input_dir / "updated.net.xml"

    # check if the updated.net.xml file exists
    if not updated_net.exists():
        # If not, generate the SUMO network
        updated_net = None
    else:
        # If it exists, use the existing SUMO network
        updated_net = pf.path2linux(updated_net)

    twin.generate_inputs(incl_sumo_net=updated_net)

    return ("RealTwin inputs generated successfully. "
            f"The inputs are saved in the directory: {twin.input_config['input_dir']}.\n")


@tool
def realtwin_simulation(input_msg: str = "") -> str:
    """Perform the RealTwin Simulation and Calibration. You need to extract the configuration file path from the input message.

    Args: input_msg (str), the path to the RealTwin configuration file. If not provided, it will use the default configuration file.

    Returns:
        str: A message indicating the success of the simulation and calibration and the generated images.
    """

    try:
        # check if input_msg is a path with .yaml extension
        if input_msg and Path(input_msg).suffix == '.yaml':
            config_path = Path(input_msg)
        else:
            # If no valid path is provided, use the default configuration file
            config_path = Path(__file__).parent.parent / "proj_config" / "realtwin_config.yaml"

        twin = rt.RealTwin(input_config_file=pf.path2linux(config_path), input_confirm=False)

        # check if the SUMO simulator is installed
        twin.env_setup(sel_sim=["SUMO"])

        # Get the dir of the user input
        input_dir = Path(twin.input_config["input_dir"])
        updated_net = input_dir / "updated.net.xml"

        # check if the updated.net.xml file exists
        if not updated_net.exists():
            # If not, generate the SUMO network
            updated_net = None
        else:
            # If it exists, use the existing SUMO network
            updated_net = pf.path2linux(updated_net)

        twin.generate_inputs(incl_sumo_net=updated_net)

        # Step 5: generate abstract scenario
        twin.generate_abstract_scenario()

        # AFTER step 5, Double-check the Matchup Table in the input directory to ensure it is correct.

        # Step 6: generate scenarios
        twin.generate_concrete_scenario()

        # Step 7: simulate the scenario
        twin.prepare_simulation()

        # Step 8: perform calibration, Available algorithms: GA: Genetic Algorithm, SA: Simulated Annealing, TS: Tabu Search
        twin.calibrate(sel_algo={"turn_inflow": "GA", "behavior": "GA"})

        png_list = list(Path(twin.input_config["output_dir"]).glob("*.png"))

        return f"RealTwin simulation and calibration completed successfully. Generated images: {', '.join(str(p) for p in png_list)}"
    except Exception as e:
        return f"An error occurred during the RealTwin simulation and calibration: {str(e)}"
