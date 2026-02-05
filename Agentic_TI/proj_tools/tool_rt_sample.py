'''
##############################################################
# Created Date: Wednesday, July 16th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''
from pathlib import Path
import sys
from langchain_core.tools import tool
try:
    import realtwin as rt
except ImportError:
    # Add current directory to sys.path
    current_dir = Path(__file__).parent.resolve()
    sys.path.append(str(current_dir))

    import realtwin as rt


@tool
def realtwin_sample_run(input_msg: str = "") -> str:
    """This tool is used to run a sample RealTwin simulation.

    Args:
        input_msg (str): Reserved for future use. Not required for running the sample.

    Returns:
        str: A message indicating the success of the sample run.
    """
    # Here you would implement the logic to run a sample RealTwin simulation
    # For now, we will just return a success message

    try:
        path_config = Path(__file__).parent.parent / "proj_config" / "realtwin_config_default.yaml"

        twin = rt.RealTwin(input_config_file=path_config, input_confirm=False)
        twin.input_config["input_dir"] = str(Path(__file__).parent.parent / "datasets" / "example2")
        twin.input_config["output_dir"] = str(Path(__file__).parent.parent / "datasets" / "example2" / "output")

        print("  :Run Sample - input_dir:", twin.input_config["input_dir"])

        twin.env_setup(sel_sim=["SUMO"])

        path_net_updated = Path(twin.input_config["input_dir"]) / "updated.net.xml"
        if not path_net_updated.exists():
            path_net_updated = None
        print("  :Run Sample - path_net_updated:", path_net_updated)
        twin.generate_inputs(incl_sumo_net=str(path_net_updated))

        twin.generate_abstract_scenario()
        twin.generate_concrete_scenario()
        twin.prepare_simulation()
        twin.calibrate(sel_algo={"turn_inflow": "GA", "behavior": "GA"})

        output_dir = twin.input_config["output_dir"]

        png_list = list(Path(output_dir).glob("*.png"))

        return f"Sample RealTwin simulation run successfully and outputs are saved in {output_dir}. Generated images: {', '.join(str(p) for p in png_list)}"

    except Exception as e:
        return f"An error occurred while running the RealTwin sample: {str(e)}"


if __name__ == "__main__":
    path_config = Path(__file__).parent.parent / "proj_config" / "realtwin_config_default.yaml"

    twin = rt.RealTwin(input_config_file=path_config, input_confirm=False)
    twin.input_config["input_dir"] = str(
        Path(__file__).parent.parent / "datasets" / "example2")
    twin.input_config["output_dir"] = str(
        Path(__file__).parent.parent / "datasets" / "example2" / "output")

    print("  :Run Sample - input_dir:", twin.input_config["input_dir"])

    twin.env_setup(sel_sim=["SUMO"])

    path_net_updated = Path(twin.input_config["input_dir"]) / "updated.net.xml"
    if not path_net_updated.exists():
        path_net_updated = None
    print("  :Run Sample - path_net_updated:", path_net_updated)
    twin.generate_inputs(incl_sumo_net=str(path_net_updated))

    twin.generate_abstract_scenario()
    twin.generate_concrete_scenario()
    twin.prepare_simulation()
    twin.calibrate(sel_algo={"turn_inflow": "GA", "behavior": "GA"})
