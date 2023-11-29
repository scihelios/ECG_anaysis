import os
from shutil import copy2
from pathlib import Path

# Define the source directory where the 'Person_xx' folders are located.
src_directory = Path("C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/ecg-id-database-1.0.0")

# Define the target directory where all 'rec_xx.dat' files will be collected.
target_directory = Path("C:/collected data")
target_directory.mkdir(parents=True, exist_ok=True)

# Loop through each 'Person_xx' directory within the source directory.
for person_dir in src_directory.glob("Person_*"):
    if person_dir.is_dir():
        # Extract the person number from the directory name (e.g., 'Person_01' => '01').
        person_number = person_dir.name.split('_')[1]
        
        # In each 'Person_xx' directory, find all 'rec_xx.dat' files.
        for dat_file in person_dir.glob("rec_*.dat"):
            # Construct the new filename by adding the person number.
            new_filename = f"rec_{person_number}_{dat_file.name}"
            
            # Copy each 'rec_xx.dat' file to the 'collected data' directory with the new filename.
            copy2(dat_file, target_directory / new_filename)