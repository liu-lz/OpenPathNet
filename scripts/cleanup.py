import os
import shutil

def cleanup_generated_files():
    """Cleans up generated scene files and datasets."""
    # Define directories to clean up
    scenes_dir = os.path.join("data", "scenes")
    datasets_dir = os.path.join("data", "datasets")
    logs_dir = "logs"

    # Remove scenes directory if it exists
    if os.path.exists(scenes_dir):
        shutil.rmtree(scenes_dir)
        print(f"Removed directory: {scenes_dir}")

    # Remove datasets directory if it exists
    if os.path.exists(datasets_dir):
        shutil.rmtree(datasets_dir)
        print(f"Removed directory: {datasets_dir}")

    # Remove logs directory if it exists
    if os.path.exists(logs_dir):
        shutil.rmtree(logs_dir)
        print(f"Removed directory: {logs_dir}")

    # Optionally, create the directories again for fresh use
    os.makedirs(scenes_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    print("Cleanup complete. Directories have been recreated.")

if __name__ == "__main__":
    cleanup_generated_files()