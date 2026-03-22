from pathlib import Path

def rename_images(raw_folder_path):
    raw_folder = Path(raw_folder_path)

    if not raw_folder.exists():
        print(f"Folder not found: {raw_folder}")
        return

    # Supported image extensions
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    # Get all image files and sort them for reproducibility
    image_files = sorted(
        [f for f in raw_folder.iterdir() if f.suffix.lower() in image_extensions]
    )

    print(f"Found {len(image_files)} images.")

    # Step 1: Temporarily rename to avoid name conflicts
    temp_files = []
    for idx, file in enumerate(image_files, start=1):
        temp_name = raw_folder / f"temp_{idx}{file.suffix.lower()}"
        file.rename(temp_name)
        temp_files.append(temp_name)

    # Step 2: Rename to final numeric names
    for idx, temp_file in enumerate(temp_files, start=1):
        final_name = raw_folder / f"{idx}{temp_file.suffix.lower()}"
        temp_file.rename(final_name)

    print("Renaming completed successfully.")


if __name__ == "__main__":
    # Adjust this path if needed
    rename_images("../../data/raw")