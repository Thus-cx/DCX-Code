import json
from pathlib import Path
import json

AREAS = []
def process_json_files(directory_path):
    """
    Reads all JSON files in the specified directory.
    """
    path = Path(directory_path)

    # Make sure the directory actually exists
    if not path.is_dir():
        print(f"Error: Directory '{directory_path}' not found.")
        return

    # Iterate through all files ending in '.json'
    for filepath in path.glob('*.json'):
        try:
            # Open and read the file
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
                episode_id = data["episode_id"]
                furnitures_info = data["steps"][0]["furnitures"]
                for furniture_info in furnitures_info:
                    if furniture_info["name"].startswith("floor"):
                        area_name = furniture_info["name"].replace("floor_", "", 1).rsplit("_", 1)[0]
                        if area_name not in AREAS:
                            AREAS.append(area_name)


                # --- START DOING THINGS WITH YOUR DATA ---

                print(f"Successfully read: {filepath.name}")
                # Example: If your JSON has a key called 'id', you can access it like this:
                # print(data.get('id', 'No ID found'))

                # --- END DOING THINGS WITH YOUR DATA ---

        # Catch JSON formatting issues or other read errors
        except json.JSONDecodeError:
            print(f"Error: {filepath.name} is not a valid JSON file.")
        except Exception as e:
            print(f"An unexpected error occurred with {filepath.name}: {e}")


# Example usage:
directory_path = "test_dataset/metadata"  # <-- Replace with your folder path
process_json_files(directory_path)
print(AREAS)