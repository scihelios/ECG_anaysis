import os
import shutil

def transfer_json_files(source_directory, destination_directory):
    """
    Transfers all JSON files from the source directory to the destination directory.

    :param source_directory: Path to the directory containing JSON files.
    :param destination_directory: Path to the directory where JSON files will be transferred.
    """
    # Create destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Counter for the number of files transferred
    file_count = 0
    print('on')
    # Loop through all files in the source directory
    for filename in os.listdir(source_directory):
        print(file_count)
        if filename.endswith(".json"):
            source_file = os.path.join(source_directory, filename)
            destination_file = os.path.join(destination_directory, filename)

            # Copy file to destination directory
            shutil.copy(source_file, destination_file)
            file_count += 1
            if file_count>20000:
                return
            # Optional: print progress
            print(f"Transferred {filename}")

    print(f"Total of {file_count} JSON files have been transferred.")

# Example usage
source_dir = 'C:/Users/ahmed mansour/Desktop/massive training'  # Replace with your source directory path
destination_dir = 'C:/Users/ahmed mansour/Desktop/smaller training'  # Replace with your destination directory path

transfer_json_files(source_dir, destination_dir)