import os
from fruit_quality import analyze_fruit


DATASET_DIR = "dataset"
RESULTS_DIR = "results"


def main():

    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    )

    images = [
        file
        for file in os.listdir(DATASET_DIR)
        if file.lower().endswith(image_extensions)
    ]

    if not images:
        print("No images found in dataset.")
        return

    print("\n====================================")
    print("     FRUIT QUALITY DATASET TEST")
    print("====================================")
    print(f"Images found: {len(images)}")

    for index, filename in enumerate(images, 1):

        image_path = os.path.join(
            DATASET_DIR,
            filename
        )

        # Create separate folder for each image
        image_name = os.path.splitext(filename)[0]

        image_result_dir = os.path.join(
            RESULTS_DIR,
            image_name
        )

        os.makedirs(
            image_result_dir,
            exist_ok=True
        )

        print("\n------------------------------------")
        print(f"[{index}/{len(images)}] Processing: {filename}")
        print("------------------------------------")

        analyze_fruit(
            image_path,
            image_result_dir
        )

    print("\n====================================")
    print("       DATASET PROCESSING DONE")
    print("====================================")
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
