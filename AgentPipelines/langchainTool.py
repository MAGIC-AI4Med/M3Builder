from langchain.tools import tool
import os
from langchain.tools.shell.tool import ShellTool
import shutil
import re
import csv
import json

def is_code_file(file_path):
    """判断文件是否为代码文件"""
    code_extensions = ['.py', '.java', '.cpp', '.c', '.js', '.html', '.css', '.php', '.go', '.rb', '.sh', '.json']
    return any(file_path.endswith(ext) for ext in code_extensions)

def should_skip_directory(dir_path, file_count_threshold=1000):
    """判断是否应该跳过该目录（基于文件数量）"""
    try:
        file_count = sum(1 for entry in os.scandir(dir_path) if entry.is_file())
        return file_count > file_count_threshold
    except (PermissionError, FileNotFoundError):
        # 如果无法访问目录，跳过
        return True

@tool
def traverse_dirs(directory: str) -> str:
    """
    Recursively get paths of all code files in a directory and return them as a single string.
    
    Args:
        directory: Directory path to scan

    Returns:
        str: A string containing paths of all code files, separated by newlines
    """
    print(f"Running traverse_dirs tool under {directory}...")
    code_files = []
    file_count_threshold = 10
    for root, dirs, files in os.walk(directory):
        if should_skip_directory(root, file_count_threshold):
            continue
        
        for file in sorted(files):
            file_path = os.path.join(root, file)
            if is_code_file(file_path):
                code_files.append(file_path)
        
        dirs.sort()
    
    # 将文件路径列表转换为以换行符分隔的字符串
    code_files_str = '\n'.join(code_files)
    # print(code_files_str)
    return code_files_str

@tool
def read_files(file_paths: list) -> dict:
    """
    Read file contents and return as dictionary
    
    Args:
        file_paths: List of file paths to read

    Returns:
        dict: Dictionary with {file_path: file_content} format
    """
    print(f"Running read_files tool to read {file_paths}...")
    file_contents = {}
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                file_contents[file_path] = content
        except (UnicodeDecodeError, PermissionError, FileNotFoundError) as e:
            print(f"Cannot read file {file_path}: {e}")
            file_contents[file_path] = None
    
    return file_contents

@tool
def copy_files(file_mapping: dict) -> str:
    """
    Copy multiple files from source paths to destination paths.

    Args:
        file_mapping (dict): A dictionary where keys are source file paths and values are destination file paths.
            Example: {
                "/path/to/source1.txt": "/path/to/destination1.txt",
                "/path/to/source2.txt": "/path/to/destination2.txt"
            }

    Returns:
        str: A message indicating the result of the operation.
    """
    print(f"Running copy_files tool to copy {file_mapping}...") 
    results = []
    for src, dest in file_mapping.items():
        try:
            # 检查源文件是否存在
            if not os.path.exists(src):
                results.append(f"源文件 {src} 不存在。")
                continue

            # 检查目标路径是否存在，如果不存在则创建
            dest_directory = os.path.dirname(dest)
            if not os.path.exists(dest_directory):
                os.makedirs(dest_directory)

            # 复制文件
            shutil.copy2(src, dest)
            results.append(f"文件 {src} 已成功复制到 {dest}")

        except Exception as e:
            results.append(f"复制文件 {src} 时出错: {e}")

    # 返回所有任务的结果
    return "\n".join(results)

@tool
def copy_directory(src_dir: str, dest_dir: str) -> str:
    """
    Copy source directory and its contents to destination directory, creating a new subdirectory
    with the same name as the source directory.
    
    Args:
        src_dir: Source directory path (e.g., '/path/to/source/folder_name')
        dest_dir: Parent destination directory path (e.g., '/path/to/dest')
                 A new subdirectory named 'folder_name' will be created here

    Returns:
        str: Operation result message

    Example:
        If src_dir is '/path/to/source/folder_name' and dest_dir is '/path/to/dest',
        the contents will be copied to '/path/to/dest/folder_name'
    """
    print(f"Running copy_directory tool to copy from {src_dir} to {dest_dir}...")
    try:
        # Check if source directory exists
        if not os.path.exists(src_dir):
            return f"Source directory {src_dir} does not exist"
            
        # Get the source directory name
        src_name = os.path.basename(src_dir.rstrip('/'))
        target_dir = os.path.join(dest_dir, src_name)
        
        # If destination directory exists, remove it first
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
            
        # Copy the entire directory tree
        shutil.copytree(src_dir, target_dir)
        return f"Directory {src_dir} has been successfully copied to {target_dir}"
        
    except Exception as e:
        return f"Error copying directory: {str(e)}"

@tool
def write_file(content: str, file_path: str) -> None:
    """
    Write a given string of code to a specified file.

    This function creates the necessary directories for the file (if they don't exist), 
    writes the content to the file, and handles any errors that may occur during the process.

    Args:
        content (str): The code or text you want to write into the file.
        file_path (str): The full path (including filename) where the content will be saved.

    Example:
        write_file('print("Hello World")', 'scripts/hello.py')
    """
    print(f"Running write_file tool to write {file_path}...")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        
        print(f"File successfully written to: {file_path}")
    except Exception as e:
        print(f"Error writing file: {e}")

@tool
def edit_file(new_content: str, file_path: str) -> None:
    """
    Completely overwrite a file with new content. The original file content will be replaced entirely.
    
    Args:
        new_content: Complete content to replace the existing file content. This should be the entire
                    desired content of the file after editing, not just the changes.
        file_path: Path of the file to edit

    Note:
        This function performs a complete overwrite operation. The original content will be lost.
        You must provide the complete desired final content, including both modified and unmodified parts.
    """
    print(f"Running edit_file tool to edit {file_path}...")
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(new_content)
        
        print(f"File {file_path} successfully edited.")
    except Exception as e:
        print(f"Error editing file: {e}")   

@tool
def run_script(command: str) -> str:
    """
    Execute shell command
    
    Args:
        command: Shell command to execute

    Returns:
        str: Command execution result
    """
    print("Running run_script tool...")
    shell_tool = ShellTool()
    result = shell_tool.run({
        "commands": [command]
    })
    return result


# @tool("list_files_in_second_level", 
#       description="Traverse the root directory and process each second-level entry. " 
#                   "For a file entry, record the file directly; for a directory entry, recursively "
#                   "collect all files (without depth limit), sort them naturally based on their relative paths, "
#                   "and return only the first max_files file paths along with the total file count. " 
#                   "Returns a dictionary with an 'entries' key containing the details.")

def natural_sort_key(s):
    """
    Generate a key for natural sorting.

    This function splits the string into numeric and non-numeric parts so that,
    for example, "file2" is sorted before "file10".

    Parameters:
      s (str): The string to generate a sorting key for.

    Returns:
      list: A list containing integers and lower-case string parts to be used as a sort key.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def get_second_level_entries(root_dir):
    """
    Retrieve all second-level entries (files and directories) under the specified root directory,
    and sort them so that directories come first, followed by files. Both are sorted by natural order.

    Parameters:
      root_dir (str): The root directory path.

    Returns:
      list: A list of os.DirEntry objects representing the second-level entries.
            Returns an empty list if an error occurs while scanning.
    """
    try:
        entries = list(os.scandir(root_dir))
    except Exception as e:
        print(f"Error scanning {root_dir}: {e}")
        return []
    
    # Sort entries: directories first, then files; both sorted naturally.
    entries.sort(key=lambda e: (not e.is_dir(), natural_sort_key(e.name)))
    return entries

def collect_all_files_from_directory(directory):
    """
    Recursively traverse the given directory and collect all file paths.

    The function sorts directories and files naturally at each level.
    It returns a list of tuples (relative_path, full_file_path) where:
      - relative_path is the file path relative to the input directory.
      - full_file_path is the absolute path to the file.

    Parameters:
      directory (str): The directory to traverse.

    Returns:
      list: A list of tuples for all files, sorted naturally by the relative path.
    """
    collected = []
    for root, dirs, files in os.walk(directory):
        dirs.sort(key=natural_sort_key)
        files.sort(key=natural_sort_key)
        for file in files:
            full_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_file_path, start=directory)
            collected.append((relative_path, full_file_path))
    collected.sort(key=lambda tup: natural_sort_key(tup[0]))
    return collected

@tool
def list_files_in_second_level(root_directory: str) -> dict:
    """
    Traverse all second-level entries under the given root directory and return a dictionary.
    For each entry, if it is a file, record it directly with a total file count of 1.
    If the entry is a directory, recursively gather all file paths and record the total file count,
    but only include the first 100 file paths.
    Parameters:
    root_directory: The path to the root directory.
    Returns:
    A dictionary with a key named "entries". The value is a list of dictionaries where each
    dictionary has:
        entry_name - the name of the entry,
        entry_path - the full path of the entry,
        total_files - the number of files under the entry (1 if it is a file),
        files - a list of file paths (up to 100 items for directories).
    """
    print(f"Running list_files_in_second_level tool under {root_directory}...")
    max_files = 10  # Limit to at most 100 file paths for directory entries

    results = []
    second_level_entries = get_second_level_entries(root_directory)
    
    for entry in second_level_entries:
        if entry.is_file():
            # Record the file entry directly.
            result_dict = {
                "entry_name": entry.name,
                "entry_path": entry.path,
                "total_files": 1,
                "files": [entry.path]
            }
            results.append(result_dict)
        elif entry.is_dir():
            # Recursively collect all files from the directory.
            collected_files = collect_all_files_from_directory(entry.path)
            total_file_count = len(collected_files)
            # Only include the first max_files file paths.
            top_files = [full_path for _, full_path in collected_files[:max_files]]
            result_dict = {
                "entry_name": entry.name,
                "entry_path": entry.path,
                "total_files": total_file_count,
                "files": top_files
            }
            results.append(result_dict)
    
    final_result = {"entries": results}
    # print(final_result)
    return final_result

@tool
def preview_file_content(file_path: str) -> str:
    """
    Reads a CSV, JSON, or TXT file and returns a preview string.
    
    CSV: returns the first 5 rows (comma-separated) and total rows.
    JSON: returns the first 5 key-value pairs (if dict) or 5 elements (if list) and total count.
    TXT: returns the first 10,000 words and total word count.
    """
    print(f"Running preview_file_content tool for {file_path}...")
    if file_path.lower().endswith('.csv'):
        rows = []
        total_rows = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    total_rows += 1
                    if total_rows <= 5:
                        rows.append(row)
        except Exception as e:
            return f"Error reading CSV file: {e}"
        
        preview_str = "CSV File Preview:\n"
        for row in rows:
            preview_str += ", ".join(row) + "\n"
        preview_str += f"Total rows: {total_rows}"
        return preview_str
    
    # JSON file processing
    elif file_path.lower().endswith('.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return f"Error reading JSON file: {e}"
        
        if isinstance(data, dict):
            items = list(data.items())
            total_items = len(items)
            preview_items = items[:5]
            preview_str = "JSON File Preview (first 5 key-value pairs):\n"
            for key, value in preview_items:
                preview_str += f"{key}: {value}\n"
            preview_str += f"Total key-value pairs: {total_items}"
        elif isinstance(data, list):
            total_items = len(data)
            preview_items = data[:5]
            preview_str = "JSON File Preview (first 5 elements):\n"
            for item in preview_items:
                preview_str += f"{item}\n"
            preview_str += f"Total elements: {total_items}"
        else:
            preview_str = f"Unsupported JSON type: {type(data)}"
        return preview_str

    # TXT file processing: Read the first 10,000 words
    elif file_path.lower().endswith('.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return f"Error reading TXT file: {e}"
        
        words = content.split()  # split text by any whitespace
        total_words = len(words)
        preview_words = words[:10000]
        preview_str = "TXT File Preview (first 10000 words):\n"
        preview_str += " ".join(preview_words)
        preview_str += f"\nTotal words: {total_words}"
        return "=== CSV Preview === \n" + preview_str

    else:
        return "Unsupported file type. Only CSV, JSON, and TXT files are supported."