'''
##############################################################
# Created Date: Monday, June 30th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''


import sys
import time
import warnings
import os
import re
import shutil
from pathlib import Path
import json
import gradio as gr
from gradio.processing_utils import audio_from_file
from transformers import pipeline
import numpy as np
import uuid

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages import HumanMessage

from loguru import logger
import pyufunc as pf

os.chdir(Path(__file__).parent)

path_tmp = Path(__file__).parent / "proj_tmp_gradio"
if path_tmp.exists():
    shutil.rmtree(path_tmp)
path_tmp.mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(path_tmp)

from chat_bot_supervisor_HIL import HIL_Agent, catch_tool_calls, generate_verification_message

from proj_tools import HIL_Tools
from proj_util import update_mealpy_optimizer

# Update mealpy optimizer
update_mealpy_optimizer()

warnings.filterwarnings("ignore")
path_log = Path(__file__).parent.parent / "proj_log/real_twin_sim.log"

logger.add("./proj_log/real_twin_sim.log",
           format="{time} {level} {message}",
           level="INFO",
           rotation="10 MB")

logger.info("Agentic Real-Twin is starting...")

path_css = ["./chat_css.css",]
gr.set_static_paths(paths=[Path(__file__).parent / "proj_tmp_output",
                           Path(__file__).parent / "assets",
                           Path(__file__).parent / "cache",
                           Path(__file__).parent / "proj_config",
                           Path(__file__).parent / "proj_llm",
                           Path(__file__).parent / "proj_tmp_gradio",
                           ])

CHAIN_THOUGHT = True


def process_result_message(msg: str) -> str:
    """ Process the result message to ensure it is a string.

    input is: result["messages"][-1].content

    """

    if not isinstance(msg, str):
        try:
            return json.dumps(msg, ensure_ascii=False)
        except Exception:
            return str(msg)
    return str(msg)


def reset(chatbot: list, thoughts_msg: gr.Text, hil_selection: gr.Text):
    """ Reset the chat history and thoughts message."""

    thoughts_msg = "To be responded..."
    hil_selection = ""

    return "", thoughts_msg, hil_selection


def process_user_input(msg: str, chat_history: list, thoughts_msg: gr.Text,
                       progress: gr.Progress = gr.Progress(track_tqdm=True)):
    print("new round...")
    print("msg............")
    print(f"input:{msg}")
    # start a dialogue here:

    msg_res = "I am Agentic Real-Twin AI, how can I help you?"
    if isinstance(msg, str):
        msg_res = msg.strip()

    elif isinstance(msg, dict):
        # load message directly from input text box
        if msg.get("text"):
            msg_res = msg["text"].strip()

        # load message from uploaded file
        if msg.get("files"):
            file_names = msg["files"]
            for file in file_names:
                # check input is audio.wav or audio.mp3
                if file.endswith(".wav") or file.endswith(".mp3"):
                    # read the audio file and convert to text
                    try:
                        # load local model if available
                        transcriber = pipeline("automatic-speech-recognition", model="./proj_llm/whisper-base-en")
                    except Exception:
                        # download the  model if not available
                        transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
                        transcriber.save_pretrained(Path(__file__).parent / "proj_llm/whisper-base-en")

                    def transcribe(audio_file):
                        print(f"  INFO:audio file: {audio_file}")
                        # audio = gr.Audio(value=Path(audio_file), type="filepath")
                        sr, y = audio_from_file(audio_file)

                        # Convert to mono if stereo
                        if y.ndim > 1:
                            y = y.mean(axis=1)

                        y = y.astype(np.float32)
                        y /= np.max(np.abs(y))

                        return transcriber({"sampling_rate": sr, "raw": y})["text"]

                    msg_res = transcribe(file)
                    print(f"  INFO:transcribe: {msg_res}")
                # check input is image.png or image.jpg
                elif file.endswith(".png") or file.endswith(".jpg"):
                    # read the image file and convert to text
                    # image_text = DTSBot.image_to_text(file)
                    msg_res += f" (Image input received at {file})"
                # check input is video.mp4 or video.avi
                elif file.endswith(".mp4") or file.endswith(".avi"):
                    # read the video file and convert to text
                    # video_text = DTSBot.video_to_text(file)
                    pass
                # check input is csv, json, xml, text, xlsx, or xls
                elif file.endswith(".csv") or file.endswith(".json") or file.endswith(".xml") or file.endswith(".txt") or file.endswith(".xlsx") or file.endswith(".xls"):
                    # read the file and convert to text
                    # file_text = DTSBot.file_to_text(file)
                    print("Invalid audio file format. Only .wav and .mp3 are supported.")
                else:
                    print("Invalid file format. Only .wav, .mp3, .png, .jpg, .mp4, .avi, .csv, .json, .xml, .txt, .xlsx, and .xls are supported.")

    print(f"  INFO:msg_res: {msg_res}")

    global thread
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    user_message = [HumanMessage(content=msg_res)]

    global bot_response
    for event in HIL_Agent.stream({"messages": user_message}, thread, stream_mode="values"):
        # print("stream event: ", event)
        if "messages" in event:
            bot_result = event["messages"]

    if CHAIN_THOUGHT:
        logger.info("Chain of Thought enabled.")
        logger.info(f"{bot_result}")

    try:
        # Get Tool info to update on HIL section
        hil_selection_C = ""
        hil_confirm_C = ""
        for Message in bot_result[::-1]:
            if isinstance(Message, AIMessage):
                try:
                    tool_calls = Message.tool_calls
                    # not a transfer tool call
                    if tool_calls and "transfer" not in tool_calls[-1]["name"].lower():
                        tool_name = Message.tool_calls[-1]["name"]
                        tool_args = Message.tool_calls[-1]["args"]

                        tool_selection_text = f"Tool invoked: {tool_name}, with arguments: {tool_args}"

                        hil_selection_C = gr.Textbox(elem_id="hil_selection",
                                                     value=tool_selection_text,
                                                     visible=True,
                                                     interactive=True,
                                                     label="",
                                                     info=("1. Wrong Tool, select No. "
                                                           "2. Wrong args, edit below and select No."))
                        # print(f"  INFO:Tool name: {tool_name} and HIL_Tools: {HIL_Tools}")
                        hil_confirm_C = gr.Radio(
                            choices=["Yes", "No"],
                            info=("Please confirm tool and arguments,"
                                  " click yes or no"),
                            visible=True,
                            elem_id="hil_confirm",
                            interactive=True,
                            label="")

                        break
                except Exception:
                    print(
                        f"  ERROR:Failed to process tool calls from {Message}.")
                    continue
    except Exception as e:
        print(f"  ERROR:Failed to process tool calls: {e}")
        hil_selection_C = ""
        hil_confirm_C = ""

    try:
        # Get the last AIMessage (Not Transfer Message) from the bot result
        for Message in bot_result[::-1]:
            if isinstance(Message, AIMessage):
                content_message = Message.content
                # content_message is not empty and does not contain "transfer"
                if content_message and "transfer" not in content_message.lower():
                    bot_response = Message.content
                    bot_response_Message = Message
                    break

    except Exception:
        print("  ERROR:Failed to process bot response, using last message content.")
        bot_response = process_result_message(bot_result[-1].content)

    print(f"Dialog response: {bot_response}")

    # extract filename from the response
    regex_lst = [re.compile(r'`([^`]+)`'),
                 re.compile(r'\((.*?)\)'),]

    filenames = None

    try:
        filenames_extract = [regex.findall(bot_response) for regex in regex_lst]
        # print(f"  INFO:filenames before: {filenames_extract}")
        for fname_lst in filenames_extract:
            # make sure the filename is a list of strings and not empty
            if fname_lst:
                try:
                    if Path(fname_lst[0]).exists():
                        filenames = fname_lst
                        break
                except Exception as e:
                    print(f"  ERROR:Failed to process filename {fname_lst[0]}: {e}")
                    filenames = None

    except AttributeError:
        filenames = None

    # print(f"  INFO:filenames after: {filenames}")
    global chat_history_updated
    chat_history_updated = []
    if filenames:
        chat_history_updated += [(msg_res, None)]
        for fname in filenames:
            chat_history_updated += [(None, (str(pf.path2linux(fname)),))]
        chat_history_updated += [(None, bot_response)]
    else:
        chat_history_updated += [(msg_res, bot_response)]
    chat_history_response = chat_history + chat_history_updated

    # update the thoughts message from memory from langgraph
    global thoughts_msg_updated
    thoughts_msg_updated = []

    try:
        for thought in bot_result["messages"]:
            thoughts_msg_updated.append(f"Thoughts: {thought.content}\n\n")
    except Exception:
        for thought in bot_result:
            thoughts_msg_updated.append(f"Thoughts: {thought}\n\n")
            # thoughts_msg_list += f"Thoughts: {thought.content}\n\n"
    thoughts_msg = thoughts_msg or "To be responded..."
    thoughts_msg_response = thoughts_msg + "".join(thoughts_msg_updated)

    try:
        global verification_message
        verification_message = generate_verification_message(bot_response_Message)
        verification_message.pretty_print()
    except Exception as e:
        print(f"  ERROR:Failed to generate verification message: {e}")
        verification_message = HumanMessage(content="Please confirm the tool and arguments.")

    print("hil_selection_C: ", hil_selection_C)

    # Tool have been called, with hil implemented
    if hil_selection_C:

        if tool_name in HIL_Tools:
            chat_history_response = chat_history
            thoughts_msg_response = thoughts_msg

            if tool_name in ["realtwin_inputs_generation", "realtwin_simulation"]:
                path_user_input = Path(__file__).parent / "datasets/User_Input"
                # check if the folder exists
                if path_user_input.exists():
                    # check if the folder is empty
                    if not any(path_user_input.iterdir()):
                        pass
                else:
                    path_user_input = Path(__file__).parent / "datasets/example2"

                gr.Warning(
                    message=f"""<h1 style="color: blue;">Please prepare the <b>Control</b> and <b>Traffic</b>
                    data and fill in the <b>Matchup Table</b> in folder: <i>{path_user_input}</i>."</h1>
                            <br>
                            <p style="color: red;">Please confirm the tool and arguments in <b>Human-In-The-Loop</b> Section</p>
                            <br>
                            <p style="color: green;">For more information, please refer to:
                            <a href="https://real-twin.readthedocs.io/en/latest/index.html#">Real-Twin Documentation</a></p>
                            """,
                    duration=None)

            else:
                gr.Info(
                    message=f"""<p style="opacity: 1; background-color: white;">Tool {tool_name} is called,
                    please confirm the tool and arguments from <b>Human-In-Loop</b> Section.</p>
                    """,
                    duration=15)

        else:
            hil_confirm_C = gr.Radio(choices=["Yes", "No"],
                                     info=("Please confirm tool and arguments,"
                                           " click yes or no"),
                                     visible=True,
                                     elem_id="hil_confirm",
                                     interactive=False,
                                     label="")

        return (
            "",
            chat_history_response,
            thoughts_msg_response,

            # show the selection for user
            hil_selection_C,

            # show confirmation radio
            hil_confirm_C,
        )

    # No HIL Needed
    else:
        return (
            "",
            chat_history_response,
            thoughts_msg_response,

            # show the selection for user
            gr.Textbox(elem_id="hil_selection",
                       value=None,
                       visible=True,
                       interactive=False,
                       label="",
                       info=("Notes: 1. Select No for Wrong Tool. "
                             "2. Select No and edit args for Wrong args")),

            # show confirmation radio
            gr.Radio(choices=["Yes", "No"],
                     info=("Please confirm tool and arguments,"
                           " click yes or no"),
                     visible=True,
                     elem_id="hil_confirm",
                     interactive=False,
                     label=""),
        )


def process_hil_input(tool_selection, tool_confirm, chat_history, thoughts_msg,
                      progress: gr.Progress = gr.Progress(track_tqdm=True)):
    confirm_message = HumanMessage(tool_confirm)

    global tool_call_message

    if confirm_message.content == "Yes":

        global chat_history_updated
        global thoughts_msg_updated

        chat_history_response = chat_history + chat_history_updated
        thoughts_msg_response = thoughts_msg + "".join(thoughts_msg_updated)

        return (
            # show the selection for user
            gr.Radio(choices=["Yes", "No"],
                     info=("Please confirm tool and arguments,"
                           " click yes or no"),
                     visible=True,
                     elem_id="hil_confirm",
                     interactive=False,
                     label=""),

            # chatbot
            chat_history_response,

            # thoughts message
            thoughts_msg_response,
        )
    else:
        for event in HIL_Agent.stream({"messages": "Please use the following tool and arguments: " + tool_selection},
                                      thread, stream_mode="values"):

            if "messages" in event:
                bot_result = event["messages"]

        try:
            # Get the last AIMessage (Not Transfer Message) from the bot result
            for Message in bot_result[::-1]:
                if isinstance(Message, AIMessage):
                    content_message = Message.content
                    # content_message is not empty and does not contain "transfer"
                    if content_message and "transfer" not in content_message.lower():
                        bot_response = Message.content
                        break

        except Exception:
            print("  ERROR:Failed to process bot response, using last message content.")
            bot_response = process_result_message(bot_result[-1].content)
        chat_history_response = chat_history + [(bot_response, None)]

        thoughts_msg_hil = []
        try:
            for thought in bot_result["messages"]:
                thoughts_msg_hil.append(f"Thoughts: {thought.content}\n\n")
        except Exception:
            for thought in bot_result:
                thoughts_msg_hil.append(f"Thoughts: {thought}\n\n")
                # thoughts_msg_list += f"Thoughts: {thought.content}\n\n"
        thoughts_msg_response = thoughts_msg + "".join(thoughts_msg_hil)

        return (
            # hide radio
            gr.Radio(choices=["Yes", "No"],
                     info=("Please confirm tool and arguments,"
                           " click yes or no"),
                     value=tool_confirm,
                     visible=True,
                     elem_id="hil_confirm",
                     interactive=False,
                     label=""),

            # hide tool select message
            chat_history_response,

            # hide result message
            thoughts_msg_response,
        )


def save_uploaded_file(file):

    print("file type: ", type(file))
    print("file uploaded: ", file.name)
    shutil.copyfile(file.name, "./temp_input/temp_uploaded_file.csv")
    print("file saved as: ", "./temp_input/temp_uploaded_file.csv")
    return gr.UploadButton(label="File Uploaded",
                           scale=None,
                           elem_id="file_layout",
                           file_count="single",
                           file_types=[".csv"])


# Accept the event argument, even if not used
def update_thoughts_visibility(t_button, thoughts_msg):
    """ Update the visibility of the thoughts message box based on the button state. """

    if t_button == "Show Agentic Thoughts":
        print("Show thoughts button clicked")
        return [
            "Hide Agentic Thoughts",
            gr.Textbox(
                thoughts_msg,
                label="Thoughts and Actions",
                interactive=False,
                lines=8,
                max_lines=8,
                placeholder="To be responded...",
                visible=True,
                scale=2,
                elem_id="thoughts_message",
                show_copy_button=True,
            )
        ]

    elif t_button == "Hide Agentic Thoughts":
        print("Hide thoughts button clicked")
        return [
            "Show Agentic Thoughts",
            gr.Textbox(
                thoughts_msg,
                label="Thoughts and Actions",
                interactive=False,
                lines=8,
                max_lines=8,
                placeholder="To be responded...",
                visible=False,
                scale=2,
                elem_id="thoughts_message",
                show_copy_button=True,
            )
        ]

    else:
        print("No button clicked")


with gr.Blocks(title="RealTwin AI",
               theme=gr.themes.Soft(primary_hue=gr.themes.colors.green,
                                    secondary_hue=gr.themes.colors.pink),
               css_paths=path_css) as demo:

    # define the GUI framework:
    with gr.Row(variant="panel", height="6vh", elem_id="title_header_layout"):
        gr.Label(label="",
                 value=("Agentic Traffic Intelligence: Augmented Human-In-The-Loop Scenario Generation"
                        " for Microscopic Traffic Simulation"),
                 elem_id="title_header")

    with gr.Row(visible=True, variant="panel"):
        with gr.Column(visible=True, variant='default', min_width=120, scale=1, elem_id="left_layout"):
            with gr.Column(visible=True,
                           variant='panel',
                           scale=2,
                           elem_id="input_container",
                           min_width=120):

                chat_input = gr.MultimodalTextbox(
                    interactive=True,
                    label="Prompt or question",
                    file_count="multiple",
                    placeholder="Enter message or upload file...",
                    show_label=False,
                    sources=["microphone", "upload"],
                    lines=1,
                    scale=1,
                    autofocus=True,
                    # show_copy_button=True,
                    elem_id="input_text_layout",
                    # stop_btn="stop"
                )

                clearBtn = gr.ClearButton(scale=1)

            gr.Markdown("### Human-In-The-Loop: Tool Selection and Actions")
            # For Human-In-Loop (HIL) confirmation and feedback
            # with gr.Row(variant="panel", elem_id="hil_layout", scale=1):
            with gr.Column(variant="panel", elem_id="hil_layout", scale=1):
                hil_selection = gr.Textbox(elem_id="hil_selection",
                                           value=None,
                                           visible=True,
                                           interactive=False,
                                           label="",
                                           info=("Notes: 1. Select No for Wrong Tool. "
                                                 "2. Select No and edit args for Wrong args"))
                hil_confirm = gr.Radio(choices=["Yes", "No"],
                                       info=("Please confirm tool and arguments,"
                                             " click yes or no"),
                                       visible=True,
                                       elem_id="hil_confirm",
                                       interactive=False,
                                       label="")


            gr.Examples(
                label='Question Hints',
                elem_id="example_layout",
                examples=[
                    "What is traffic digital twin",
                    "What is Real-Twin, who are developers?",
                    "Get behavior parameter suggested values and ranges",
                    "Get place information of a city (e.g., 'Knoxville, TN')",
                    "Get osm data from web wizard",
                    "Get osm data from relation id (e.g., '196150')",
                    "Get osm data from place name (e.g., 'Knoxville, TN')",
                    "Visualize osm network from downloaded osm data in default folder",
                    "Check SUMO installation status",
                    "Install SUMO on User Operating System",
                    "Install SUMO with specific version (e.g., '1.21.0')",
                    "Visualize SUMO network from path: temp.net.xml",
                    "Show Real-Twin default settings",
                    "Show Real-Twin configurations",
                    'Update Real-Twin configurations (e.g., set "Traffic" to "VERY Well")',
                    "Save Real-Twin configurations",
                    'RealTwin sample run with default configurations',
                    "RealTwin inputs generation using: path_config.yaml",
                    "RealTwin Simulation and Calibration"
                ],
                inputs=[chat_input],
                examples_per_page=4,
            )

            thoughtsBtn = gr.Button(
                "Hide Agentic Thoughts",
                elem_id="thoughts_button"
            )

            thoughtsMsg = gr.Textbox(
                label="Thoughts and Actions",
                interactive=False,
                lines=8,
                max_lines=8,
                placeholder="To be responded...",
                visible=True,
                scale=2,
                elem_id="thoughts_message",
                show_copy_button=True,
            )

        chatbot = gr.Chatbot(elem_id="right_layout",
                             label="Agentic Response",
                             scale=2,
                             min_height="85vh",
                             layout='bubble',
                             show_copy_button=False,
                             show_share_button=False,
                             editable=None,
                             resizable=True,
                             avatar_images=("assets/user.png",
                                            "assets/ai_assistant.png"),
                             watermark="Agentic Real-Twin Developed by ORNL ARMS Team",
                             placeholder="To be responded...",
                             )

    # Event handlers
    chat_msg = chat_input.submit(
        process_user_input,
        [chat_input, chatbot, thoughtsMsg],
        [chat_input, chatbot, thoughtsMsg, hil_selection, hil_confirm],
        show_progress_on=[chatbot, thoughtsMsg, hil_selection],  # chat_input,
        show_progress="minimal"
    )

    clearBtn.click(reset,
                   [chatbot, thoughtsMsg, hil_selection],
                   [chatbot, thoughtsMsg, hil_selection],
                   cancels=[chat_msg],  # cancel ongoing processing
                   )

    thoughtsBtn.click(update_thoughts_visibility,
                      [thoughtsBtn, thoughtsMsg],
                      [thoughtsBtn, thoughtsMsg],
                      show_progress="hidden",
                      )

    hil_confirm.change(fn=process_hil_input,
                       inputs=[hil_selection, hil_confirm, chatbot, thoughtsMsg],
                       outputs=[hil_confirm, chatbot, thoughtsMsg],
                       show_progress_on=[hil_confirm],)

if __name__ == '__main__':

    # Add C: to allowed_paths
    demo.launch(
        inbrowser=True,
        debug=True,
        allowed_paths=[
            str((Path(__file__).parent.parent).resolve()),
            Path(__file__).parent.parent,
        ],
    )
