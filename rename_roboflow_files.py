import os

def rename_roboflow_files(root_dir):
    """
    Renames files in a directory and its subdirectories using a general rule.

    The script identifies files containing ".rf." in their name. It then finds
    the last underscore ('_') that appears before ".rf." and renames the file
    using the portion of the name that came before that underscore, while
    preserving the original file extension.

    Example:
    - From: 'ANYTHING_GOES_HERE_extra.rf.somehash.jpg'
    - To:   'ANYTHING_GOES_HERE.jpg'

    Args:
        root_dir (str): The path to the root directory to start the search.
    """
    print(f"Scanning directory: {root_dir}...")
    renamed_count = 0
    skipped_count = 0

    for subdir, _, files in os.walk(root_dir):
        for filename in files:
            # Check if the file is a candidate for renaming
            if ".rf." in filename:
                try:
                    # Find the part of the name before ".rf."
                    # e.g., "20250509_093231_jpg" from "20250509_093231_jpg.rf.eda..."
                    prefix = filename.split('.rf.')[0]

                    # Find the position of the last underscore in that prefix
                    last_underscore_index = prefix.rfind('_')

                    # Proceed only if an underscore was actually found
                    if last_underscore_index != -1:
                        # The new base name is everything before that last underscore
                        base_name = prefix[:last_underscore_index]
                        
                        # Get the original file extension (e.g., '.jpg', '.txt')
                        original_extension = os.path.splitext(filename)[1]

                        # Construct the full new filename
                        new_filename = f"{base_name}{original_extension}"

                        # Prevent renaming if the name is already correct
                        if filename == new_filename:
                            continue

                        old_filepath = os.path.join(subdir, filename)
                        new_filepath = os.path.join(subdir, new_filename)

                        # Avoid overwriting a different, existing file
                        if os.path.exists(new_filepath):
                            print(f"Skipping rename: Target '{new_filepath}' already exists.")
                            skipped_count += 1
                            continue
                        
                        # Perform the rename
                        os.rename(old_filepath, new_filepath)
                        print(f"Renamed: '{filename}' to '{new_filename}'")
                        renamed_count += 1

                    else:
                        print(f"Skipping '{filename}': No underscore '_' found before '.rf.'")
                        skipped_count += 1

                except Exception as e:
                    print(f"Error processing '{filename}': {e}")
                    skipped_count += 1
    
    print("\nScript finished.")
    print(f"Total files renamed: {renamed_count}")
    print(f"Total files skipped: {skipped_count}")


if __name__ == "__main__":
    # ❗️ IMPORTANT: Replace this with the actual path to your folder
    target_folder = "/mnt/c/Work/Repos/MyProjects/DeepLearning/CoinsUltralytics/Data/CoinCount.v57i.yolov5pytorch"

    if not os.path.isdir(target_folder):
        print(f"Error: The directory '{target_folder}' does not exist.")
    else:
        rename_roboflow_files(target_folder)