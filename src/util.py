"""
Utility functions for the project
"""

import os
import sys
import socket
from datetime import datetime
from importlib.metadata import distributions
import shutil
import hashlib
import git

# pylint:disable=cell-var-from-loop


def calculate_input_file_info(file_list: list[str]) -> str:
    """
    Calculate a SHA256 checksum for each file in the file_list.
    Returns a descriptive string including file name,
    checksum, last modified date, and filesize.
    """
    file_info_strings = []

    for file_name in file_list:
        # Calculate individual file SHA256 checksum
        hash_sha256 = hashlib.sha256()
        with open(file_name, 'rb') as file:
            for byte_block in iter(
                lambda: file.read(4096), b""
            ):  # pylint:disable=cell-var-from-loop
                hash_sha256.update(byte_block)

        file_checksum = hash_sha256.hexdigest()
        # Get last modified time and size of the file
        file_stats = os.stat(file_name)
        last_modified_date = datetime.fromtimestamp(file_stats.st_mtime).strftime(
            '%Y-%m-%d %H:%M:%S'
        )
        file_size = float(file_stats.st_size)

        # Convert size to a human-friendly format
        suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        human_size = file_size
        i = 0
        while human_size >= 1024 and i < len(suffixes) - 1:
            human_size /= 1024.0
            i += 1
        file_size_human = f"{human_size:.2f} {suffixes[i]}"

        # Combine information into a string for each file
        file_info = (
            f"    File: {os.path.basename(file_name)}\n"
            f"    Checksum: {file_checksum}\n"
            f"    Last Modified: {last_modified_date}\n"
            f"    Size: {file_size_human}\n"
        )
        file_info_strings.append(file_info)

    return "\n".join(file_info_strings)


def store_info(
    path: str, input_data_paths: list[str] = [], seeds: list[int] = []
) -> str:
    """
    Store metadata enabling reproducibility of results
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata_content = f"Time: {timestamp}\n"

    # Check if the file already exists
    if os.path.isfile(path):
        timestamp_for_backup = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_folder = os.path.join(
            os.path.dirname(path), 'replaced_on_' + timestamp_for_backup
        )
        os.makedirs(backup_folder, exist_ok=True)
        shutil.move(path, backup_folder)
        metadata_content += f'Moved existing {path} in {backup_folder}\n'
        info_path = path + '.info'
        if os.path.isfile(info_path):
            shutil.move(info_path, backup_folder)
            metadata_content += f'Moved existing {info_path} in {backup_folder}\n'

    try:
        # Get the current repo SHA
        sha = git.Repo(os.getcwd()).head.commit.hexsha
        metadata_content += f'Repo SHA: {sha}\n'
    except git.exc.InvalidGitRepositoryError:
        metadata_content += 'Repo SHA: Git repo not found.\n'

    metadata_content += f'Hostname: {socket.gethostname()}\n'
    metadata_content += f'Python version: {sys.version}\n'

    # Get a list of installed packages and their versions using importlib.metadata
    installed_packages = [
        f"    {distribution.metadata['Name']}=={distribution.version}"
        for distribution in distributions()
    ]
    installed_packages_str = '\n'.join(installed_packages)
    metadata_content += f'Installed packages:\n{installed_packages_str}\n'

    command_line_args = ' '.join(sys.argv)
    metadata_content += f'Command line arguments: {command_line_args}\n'

    if input_data_paths:
        file_info = calculate_input_file_info(input_data_paths)
        metadata_content += 'Input file information:\n'
        metadata_content += file_info

    if seeds:
        metadata_content += f'Random Seeds: {seeds}\n'

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path + '.info', 'w', encoding='utf-8') as file:
        file.write(metadata_content)

    return path
