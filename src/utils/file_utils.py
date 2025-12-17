import os
import shutil
from pathlib import Path

def ensure_directory_exists(directory_path):
    """
    Ensure the directory exists; create it if it does not.
    
    Args:
        directory_path: Directory path
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Failed to create directory {directory_path}: {e}")
        return False

def clean_directory(directory_path):
    """
    Remove all contents of a directory.
    
    Args:
        directory_path: Directory path to clear
    """
    try:
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
        ensure_directory_exists(directory_path)
        return True
    except Exception as e:
        print(f"Failed to clear directory {directory_path}: {e}")
        return False

def get_file_size(file_path):
    """
    Get file size in bytes.
    
    Args:
        file_path: File path
    
    Returns:
        int: File size; returns 0 if the file does not exist
    """
    try:
        return os.path.getsize(file_path) if os.path.exists(file_path) else 0
    except Exception:
        return 0

def list_files_with_extension(directory, extension):
    """
    List all files with the specified extension in a directory.
    
    Args:
        directory: Directory path
        extension: File extension (e.g., '.xml', '.csv')
    
    Returns:
        list: List of matching file paths
    """
    try:
        files = []
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                if filename.lower().endswith(extension.lower()):
                    files.append(os.path.join(root, filename))
        return files
    except Exception as e:
        print(f"Failed to list files in {directory}: {e}")
        return []

def safe_remove_file(file_path):
    """
    Safely remove a file.
    
    Args:
        file_path: Path to the file to delete
    
    Returns:
        bool: True if deletion succeeded or file does not exist
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return True
    except Exception as e:
        print(f"Failed to delete file {file_path}: {e}")
        return False

def copy_file(src_path, dst_path):
    """
    Copy a file.
    
    Args:
        src_path: Source file path
        dst_path: Destination file path
    
    Returns:
        bool: True if copy succeeded
    """
    try:
        # Ensure destination directory exists
        dst_dir = os.path.dirname(dst_path)
        ensure_directory_exists(dst_dir)
        shutil.copy2(src_path, dst_path)
        return True
    except Exception as e:
        print(f"Failed to copy file {src_path} -> {dst_path}: {e}")
        return False

def get_directory_size(directory_path):
    """
    Get total size of a directory (bytes).
    
    Args:
        directory_path: Directory path
    
    Returns:
        int: Directory size
    """
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
    except Exception:
        pass
    return total_size

def format_file_size(size_bytes):
    """
    Format file size for display.
    
    Args:
        size_bytes: File size in bytes
    
    Returns:
        str: Human-readable size string
    """
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"