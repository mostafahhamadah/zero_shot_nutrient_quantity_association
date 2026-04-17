import json
import csv
import os

# Configuration
INPUT_FILE = "GT.txt"
OUTPUT_DIR = "data/annotations"

def convert_annotations():
    # 1. Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # To group rows by image_id
    image_groups = {}

    # 2. Read the text file
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            # We use csv.DictReader to automatically map headers to keys
            reader = csv.DictReader(f)
            
            for row in reader:
                image_id = row['image_id']
                
                # Prepare the nutrient entry
                # We strip spaces and handle nulls for cleaner JSON
                entry = {
                    "nutrient": row['nutrient'].strip(),
                    "quantity": row['quantity'].strip(),
                    "unit": row['unit'].strip(),
                    "context": row['context'].strip(),
                    "nrv_percent": row['nrv_percent'].strip() if row['nrv_percent'] else None,
                    "serving_size": row['serving_size'].strip() if row['serving_size'] else None
                }
                
                if image_id not in image_groups:
                    image_groups[image_id] = []
                
                image_groups[image_id].append(entry)

        # 3. Write individual JSON files
        for image_id, nutrients in image_groups.items():
            # Remove the .jpeg or .png extension for the filename
            filename = os.path.splitext(image_id)[0] + ".json"
            file_path = os.path.join(OUTPUT_DIR, filename)
            
            # Construct the final JSON object
            json_output = {
                "image_id": image_id,
                "nutrients": nutrients
            }
            
            # ensure_ascii=False is critical to keep the 'µ' and 'ß' symbols intact
            with open(file_path, "w", encoding="utf-8") as json_file:
                json.dump(json_output, json_file, indent=4, ensure_ascii=False)
                
        print(f"Success! Converted {len(image_groups)} images into individual JSON files.")

    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please ensure the file exists.")
    except KeyError as e:
        print(f"Error: Missing expected column in text file: {e}")

if __name__ == "__main__":
    convert_annotations()