from typing import Annotated
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
import getpass
import os
# from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.graph import MessagesState, END, START, StateGraph
from langgraph.types import Command
# from typing import Literal
from typing_extensions import TypedDict, Literal
from IPython.display import display, Image
import argparse
import warnings
warnings.filterwarnings('ignore')

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = 'fill in your api key.'
os.environ['LANGSMITH_PROJECT'] = 'project name'
os.environ['ANTHROPIC_API_KEY'] = "fill in your api key"
os.environ['ANTHROPIC_BASE_URL'] = "fill in your api key"

from langchainTool import traverse_dirs, read_files, copy_directory, copy_files, write_file, edit_file, run_script, list_files_in_second_level, preview_file_content
from langchainTool_llama import read_file

def create_selector_prompt(description_path):
    system_prompt = f"""
    You are acting as an agent for selecting a dataset that best matches a human user's requirements.
    I provide you with a list of dataset descriptions: {description_path}, which is a json that contains a list of dictionaries.
    Every dictionary contains following entries: ["dataset name", "dataset description", "dataset_path"].
    
    You have access to the tools:
    read_files: This function reads a script file (such as a Python file) so you can understand its content. Use it to read the dataloader template file to grasp the expected format for the dataloader class. Pay attention, you can not apply this tool to read train/test.json.
    
    Here is the typical workflow you should follow:
    1. Use read_files to read {description_path}, understand its content.
    2. choose exactly one dataset that best matches the user's requirements. Remember, your choice should be mainly based on "dataset description" entry.
    3. Return the chosen dataset's name, dataset description, and dataset_path, so a downstream peer agent can know these information accurately.
    4. include <end> to end the conversation.

    IMPORTANT NOTE: If you think there realy is no dataset that meets the user's requirements, then return no dataset, but you must return your reasons before ending the conversation.
    """
    return system_prompt

def create_processor_prompt(selector_content, save_path, examples_path):
    system_prompt = f"""
    You are acting as an agent for preparing training and testing data in a clinical radiology context. I provide you with a raw, unprocessed dataset and its corresponding description, which can be found in {selector_content}. Your mission is to generate three files—train.json, test.json, and label_dict.json(if needed)—and save them to the working directory: {save_path}. And you must make sure that the format of the json files matches some example files which will be mentioned below. Do not modify the original data files directly.
    IMPORTANT NOTE: In selector content, you should be able to identify the dataset's name, the dataset description, and the dataset's root path.

    You have access to the following tools:

    list_files_in_second_level:
    Use this tool to inspect the dataset's structure and determine where the image data files are located (typically large files such as png, jpg, nii, pck, pt, npy, etc.) along with any metadata or label files (usually csv, json, txt, etc.). This will help you decide which files to read further.
    preview_file_content:
    Some metadata or label files might be very large and exceed the context length. Use this tool to preview a portion of the file so that you understand its structure well enough to plan how to parse it using code.
    write_file:
    Your final goal is to generate train.json, test.json, and label_dict.json(if needed). Therefore, you should write a Python script that creates these files. Once you fully understand the folder structure and the content of the metadata/label files, write code (possibly using libraries such as os, re, pandas, etc.) to traverse or read the metadata files and extract the needed entries for each data item.
    read_files: 
    This function reads a script file (such as a Python file) so you can understand its content. Use it to read the dataloader template file to grasp the expected format for the dataloader class. Pay attention, you can not apply this tool to read train/test.json.
    edit_file:
    Since errors might occur during your first code attempt, you can use this tool to modify your files based on error messages or feedback.
    run_script:
    Use this tool to execute your Python script from the command line. Remember that this is the only tool available to run code.
    Here is the typical workflow you should follow:

    Based on the dataset's description, use the list_files_in_second_level tool to understand the organization and structure of the dataset. Identify files that are likely to contain metadata or labels.
    Use the preview_file_content tool to read a portion of these files so that you can understand their structure and determine how to parse them with your code.
    Based on the dataset's description, You Must Use the traverse_dirs tool and read_files tool to read the directory structure of {examples_path}, and find the example output jsons based on the medical task, for the next step to refer to.
    Once you feel that you have a sufficient understanding, write a Python script under director {save_path} (using the write_file tool) that generates the following:
    [train.json ,test.json and label_dict.json(if needed)]
    Remember, If label_dict.json is not provided by chosen example files, then you must not generate it!!!

    IMPORTANT: You Must Make sure that the json files you output matches the format of your chosen example files! Especially the dictionary structure!
    If you wan to read a file named 'labels.json', use read_files instead of preview_file_content!

    train/test split: Ensure that the data is split into training and testing sets in a reasonable ratio (e.g., 80/20) and that the split is random. If train/test split is already presented, you don't need to split, but you still need to generate the json files.
    Besides, ensures that for each training and testing sample the key-value pairs in the dictionary are internally shuffled.
    Use the run_script tool to execute your script. If errors occur during execution, you can use the edit_file tool to modify your code until the script runs successfully and produces the three JSON files.
    Remember, your objective is to automate the creation of shuffled train.json, test.json, and label_dict.json(if needed) without altering the raw data files directly.
    Remember, the formats of train.json, test.json and label_dict.json(if exists) must follow the example files.
    If you think there is no error anymore and all the json files are generated, please conclude your work and include <end> to end the conversations.
    """
    return system_prompt

def create_dataloader_prompt(processor_msg, dataindex_path, template_path, description):
    system_prompt = f"""
    You are acting as an agent responsible for writing a dataloader for a dataset. Your ultimate goal is to create a 'dataloader.py' script that will be used to feed data into the training process under the {dataindex_path} . The dataset's index files are located at dataindex_path: {dataindex_path} and contain train.json, test.json, and label_dict.json(may no exist). You must also choose a template file located at {template_path} and refer to it.

    A peer dataset processor has already generated these index files, (informations can be found in {processor_msg}) so your task is to write a dataloader class that can read these files and load the data into the training process. The dataloader should be able to handle the training and testing data, as well as the label dictionary.

    Datast description is also provided in: {description}

    You have access to a series of utility functions, which are as follows:

    traverse_dirs: 
    This tool traverses a given directory and returns all the file paths under that directory. It helps you understand the folder structure—in this case, to inspect the various files in the index folder.
    preview_file_content: 
    Often, JSON files have extensive content that might exceed the context length. Use this function to quickly understand the structure of the JSON files (train.json, test.json, label_dict.json) so you know how to structure your dataloader class.
    read_file: 
    This function reads a script file (such as a Python file) so you can understand its content. Use it to read the dataloader template file to grasp the expected format for the dataloader class. Pay attention, you MUST NOT apply this tool to read train/test.json.
    write_file: 
    This utility writes a file, typically a Python script. After reviewing the template and understanding the structures of train, test, and label_dict(if exists), use this tool to create your dataloader class and save it as dataloader.py for training data ingestion.
    edit_file: 
    If the initial write_file output encounters issues, you can use edit_file to modify the script based on error messages or additional prompts.
    run_script: 
    Once the dataloader is written, use this tool to execute the script and verify that it correctly loads the data without errors.
    A sample workflow might be:

    Directory Inspection: Use traverse_dirs to read the directory structure of the given path {dataindex_path}, identifying the presence of the train, test, and label_dict(may not exist) JSON index files.
    Preview JSON Content: Employ preview_file_content to inspect these JSON files and understand their structures.
    Choose Template: Based on the medical task, which can be found in dataset description, choose a dataloader code template from {template_path} for reference.
    Review Template: Utilize read_file to examine your chosen dataloader template file and understand the proper format for writing the dataloader class. Remember that this is not the end! you must go on to write dataloader.py
    Code Development: Based on the insights from the JSON structures and the template, use write_file to write your dataloader class to '{dataindex_path}/dataloader.py' (Include a main function if necessary).
    Save and Test: After writing dataloader.py, you must use run_script to test and verify that the script runs correctly!!! You must put your dataloader.py under {dataindex_path}!!!
    Debug and Validate: If errors occur, use edit_file and run_script as needed to debug the script until it fully processes the entire dataset.
    
    Remember: Your task is to write and validate a dataloader that successfully iterates over the dataset and verifies that it runs correctly during training. Always refer to the template for guidance on the expected format.
    
    You MUST use write_file to create a dataloader.py under {dataindex_path} and verify that it runs correctly!!!
    You MUST tell where you place dataloader.py!!!
    You should write dataloader according only to the json files and the template. And try not to modify too much of the template.
    If you see comments in the template like "you must not modify this line", then do not modify it.
    
    If you think your dataloader.py is ready, and the dataloader is already validated, please conclude your work and include <end> to end the conversation.
    
    Important: When you use write_file tool, print the parameters you pass to the tool function!!!
    """
    return system_prompt

def create_trainer_prompt(processor_msg, dataloader_msg, work_path, train_script_path):
    system_prompt = f"""
    You are an AI assistant specialized in radiology tasks, capable of writing training code, executing training processes, and debugging. Your primary focus areas include disease diagnosis, organ segmentation, anomaly detection, and report generation tasks. You handle end-to-end code writing, debugging, and training.

    peer processor and dataloader agents have completed preliminary tasks of dataset preparation and dataloader class writing, messages documented in {processor_msg} and {dataloader_msg}. You will build upon this groundwork.

    Your working directory is {work_path}, and all operations must be strictly confined to this directory. To accomplish training tasks, you have access to the following tools:

    1. traverse_dirs: Used for recursively traversing file paths in the workspace to understand directory structure and infer file purposes from their names.
    2. read_files: Used to read content from one or multiple files to understand implementation details and determine if changes or operations are needed. Please avoid read datapath files such as the train.json, test.json and label_dict.json.
    3. write_file: Used for implementing new functional code.
    4. edit_file: Used for modifying files, including adding data and model information to template files, adding new features, or fixing errors. Note that when using this tool to edit a file, please always firstly read the content before.
    5. run_script: Used for executing training through sh scripts.
    6. copy_files: Used to copy a file to a new path, typically used when copying train.py and train.sh from ReferenceFiles to workspace
    
    You can also access {train_script_path} to choose and copy the best matching train.py and train.sh to workspace. But you cannot edit files under {train_script_path}.

    Important notes:
    - The Datapath, Loss, and Utils directories respectively contain JSON/csv/JSONL data indices for training/validation and dataset class you will need, loss functions, and utility packages. While these shouldn't be modified, you must understand their relationships and functions.
    - The Logout directory stores training results and should not be manually written to.
    - The Model directory contains training code modules for different tasks. Generally, these shouldn't be modified, but you should read them to understand their functionality and usage. Remember that if the medical task is Organ Segmnentation, you do not have to read Model directory, because model is provided in train.py already.
    - The directory {train_script_path} contains different medical tasks' respective train.sh and train.py files, you should choose the best matching train.sh and train.py based on medical task, and copy them to workspace.
    - train.py contains the main training code template using the transformers trainer framework. You need to carefully read and modify its contents as needed.
    - train.sh is the script for running the main code, containing parameter settings that you need to understand and configure.
    - train.py has some code lines commented by sth like 'you should not modify this line', if you see this, don't modify that line.

    The workflow consists of three phases:
    1. traverse {train_script_path} to choose the best matching train.sh and train.py based on medical task, and use copy_files to copy them to workspace.
    2. Understanding structure and reading files/code templates
    3. Initial code adjustment and refinement. Modify train.py and train.sh to make them ready. A Hint: You always have to import the dataset class from {work_path}/Datapath/dataloader.py
    4. Script execution (use run_script tool to execute train.sh) and debugging loop until successful training completion

    Phase 1 requires traversing train_script_path, choosing and copying the best train.sh and train.py to workspace.
    Phase 2 requires traversing the working directory and reading all crucial code to understand their connections. 
    Phase 3 involves careful review of train.py and train.sh, making necessary modifications to achieve an executable version. 
    Phase 4 involves executing train.sh and iteratively fixing errors based on error messages until successful execution.

    IMPORTANT: You must execute train.sh and make sure it's running normally before you exit

    Before each operation, you should consider its purpose and verify its appropriateness, especially when uncertain or experiencing potential hallucinations. Use traverse or read tools to check and understand corresponding parts. Always remember your ultimate goal is to successfully run the training script.
    """
    return system_prompt    

class SelectorState(TypedDict):
    messages: list

class ProcessorState(TypedDict):
    messages: list

class DataloaderState(TypedDict):
    messages: list

class TrainerState(TypedDict):
    messages: list  

def should_continue_selector(state: SelectorState) -> Literal["selector_tools", END]:
    messages = state['messages']
    last_message = messages[-1]

    if "<end>" in last_message.content:
        return END

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "selector_tools"
    
    return END

def should_continue_processor(state: ProcessorState) -> Literal["processor_tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    
    if "<end>" in last_message.content:
        return END
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "processor_tools"
    
    return END

def should_continue_dataloader(state: DataloaderState) -> Literal["dataloader_tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    
    if "<end>" in last_message.content:
        return END
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "dataloader_tools"
    
    return END

def should_continue_trainer(state: TrainerState) -> Literal["trainer_tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    
    if "<end>" in last_message.content:
        return END
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "trainer_tools"
    
    return END

selector_tools = [read_files]
processor_tools = [list_files_in_second_level, preview_file_content, write_file, edit_file, run_script, read_files]
dataloader_tools = [traverse_dirs, preview_file_content, read_file, write_file, edit_file, run_script]
trainer_tools = [traverse_dirs, read_files, write_file, edit_file, run_script, copy_files]

model = ChatAnthropic(model="claude-3-5-sonnet-20241022", max_tokens=4096) 

selector_tool_node = ToolNode(selector_tools)
processor_tool_node = ToolNode(processor_tools)
dataloader_tool_node = ToolNode(dataloader_tools)
trainer_tool_node = ToolNode(trainer_tools)

selector = create_react_agent(model, selector_tool_node)
processor = create_react_agent(model, processor_tool_node)
dataloader = create_react_agent(model, dataloader_tool_node)
trainer = create_react_agent(model, trainer_tool_node)

selector_workflow= StateGraph(SelectorState)
processor_workflow = StateGraph(ProcessorState)
data_loader_workflow = StateGraph(DataloaderState)
trainer_workflow = StateGraph(TrainerState)

# build the selector workflow
selector_workflow = StateGraph(SelectorState)

selector_workflow.add_node("selector", selector)
selector_workflow.add_node("selector_tools", selector_tool_node)

selector_workflow.add_edge(START, "selector")
selector_workflow.add_conditional_edges(
    "selector",
    should_continue_selector,
    {
        "selector_tools": "selector_tools",
        END: END
    }
)
selector_workflow.add_edge("selector_tools", "selector")
selector_graph = selector_workflow.compile()

# build the processor workflow
processor_workflow = StateGraph(ProcessorState)

processor_workflow.add_node("processor", processor)
processor_workflow.add_node("processor_tools", processor_tool_node)

processor_workflow.add_edge(START, "processor")
processor_workflow.add_conditional_edges(
    "processor",
    should_continue_processor,
    {
        "processor_tools": "processor_tools",
        END: END
    }
)
processor_workflow.add_edge("processor_tools", "processor")

processor_graph = processor_workflow.compile()

# build the dataloader workflow
data_loader_workflow = StateGraph(DataloaderState)

data_loader_workflow.add_node("dataloader", dataloader)
data_loader_workflow.add_node("dataloader_tools", dataloader_tool_node)

data_loader_workflow.add_edge(START, "dataloader")
data_loader_workflow.add_conditional_edges(
    "dataloader",
    should_continue_dataloader,
    {
        "dataloader_tools": "dataloader_tools",
        END: END
    }
)
data_loader_workflow.add_edge("dataloader_tools", "dataloader")

dataloader_graph = data_loader_workflow.compile()

# build the trainer workflow
trainer_workflow = StateGraph(TrainerState)

trainer_workflow.add_node("trainer", trainer)
trainer_workflow.add_node("trainer_tools", trainer_tool_node)

trainer_workflow.add_edge(START, "trainer")
trainer_workflow.add_conditional_edges(
    "trainer",
    should_continue_trainer,
    {
        "trainer_tools": "trainer_tools",
        END: END
    }
)
trainer_workflow.add_edge("trainer_tools", "trainer")

trainer_graph = trainer_workflow.compile()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_requirements", type=str)
    args = parser.parse_args()

    Human_Requirements = args.human_requirements
    print("\nHuman Requirements: ", Human_Requirements, "\n---------------------------------------------------------------------\n")

    description_path = "path/to/M3Builder_repo/ReferenceFiles/DataCard/descriptions.json"

    selector_prompt = create_selector_prompt(description_path)
    selector_result = selector_graph.invoke(
        {
            "messages": [
                SystemMessage(content=selector_prompt),
                HumanMessage(content=Human_Requirements)
            ]
        }
    )
    selector_content = selector_result['messages'][-1].content.replace("<end>", "")#
    print("Selector Content:", selector_content, '\n/////////////////////////////////////////////////////////////////////\n')


    save_path = "path/to/M3Builder_repo/TrainPipeline/Datapath"
    examples_path = "path/to/M3Builder_repo/ReferenceFiles/DataJsonExamples"

    processor_prompt = create_processor_prompt(selector_content, save_path, examples_path)
    processor_result = processor_graph.invoke(
        {
            "messages": [
                SystemMessage(content=processor_prompt),
                HumanMessage(content="Now, please generate the train, test, and label files.")
            ]
        },
        {"recursion_limit": 100}
    )

    processor_content = processor_result['messages'][-1].content.replace("<end>", "")
    print("Processor Content:", processor_content, "\n/////////////////////////////////////////////////////////////////////\n")


    dataindex_path = save_path
    template_path = "path/to/M3Builder_repo/ReferenceFiles/DataLoaderExamples"

    dataloader_prompt = create_dataloader_prompt(processor_content, dataindex_path, template_path, description=selector_content)

    dataloader_result = dataloader_graph.invoke(
        {
            "messages": [
                SystemMessage(content=dataloader_prompt),
                HumanMessage(content="I would like to start the dataloader creation process to create and validate the dataloader script."),
            ],
        },
        {"recursion_limit": 100}
    )

    dataloader_content = dataloader_result['messages'][-1].content.replace("<end>", "")
    print("DataLoader Content:", dataloader_content, "\n/////////////////////////////////////////////////////////////////////\n")


    work_path = "path/to/M3Builder_repo/TrainPipeline"
    train_script_path = "path/to/M3Builder_repo/ReferenceFiles/TrainingScripts"

    trainer_prompt = create_trainer_prompt(processor_content, dataloader_content, work_path, train_script_path)

    trainer_result = trainer_graph.invoke(
        {
            "messages": [
                SystemMessage(content=trainer_prompt),
                HumanMessage(content=f"Now I would like to start the training process to train a model."),
            ],
            "recursion_limit": 100
        },
        {"recursion_limit": 100}
    )

    print(trainer_result)

if __name__=="__main__":
    main()