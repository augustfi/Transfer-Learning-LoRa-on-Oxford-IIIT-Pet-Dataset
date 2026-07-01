import os
import re
import pandas as pd
from pathlib import Path

IMAGE_DIR = Path(__file__).parent.parent / "data" / "images"


def load_dataset(image_dir: Path = IMAGE_DIR) -> pd.DataFrame:
    records = []
    for fname in os.listdir(image_dir):
        stem = os.path.splitext(fname)[0]
        match = re.match(r"^(.+?)_(\d+)$", stem) #this is the regex pattern of image filename


        breed_raw, number = match.group(1), int(match.group(2))
        is_cat     = int(stem[0].isupper())
        breed_name = breed_raw.replace("_", " ").strip()
        records.append({
            "filepath":  image_dir / fname,   # full path for loading images later
            "filename":  fname,
            "is_cat":    is_cat,
            "breed":     breed_name,
            "image_num": number,
        })

    df = pd.DataFrame(records)
    return df

def main():
    df = load_dataset()
    print(f"Total images: {len(df)}")
    print(f"Cats: {df['is_cat'].sum()}")
    print(f"Dogs: {len(df) - df['is_cat'].sum()}")
    print(f"Unique breeds:{df['breed'].nunique()}")
    print()
    print(df.groupby("breed").size().rename("count").to_string())

if __name__ == "__main__":
    main()
