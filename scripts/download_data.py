from argparse import ArgumentParser
from pathlib import Path

import httpx
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/images")

    args = parser.parse_args()

    # Create output directory if it not yet exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    for subdir in df["dataset_split"].unique():
        Path(args.output_dir).joinpath(subdir).mkdir(parents=True, exist_ok=True)

    failures: list[tuple[int, str]] = []

    for row in tqdm(df.itertuples(), total=len(df), desc="Downloading"):
        output_path = Path(args.output_dir) / row.dataset_split / f"{row.image_id}.jpg"
        if output_path.exists():
            continue
        try:
            response = httpx.get(row.image_url, follow_redirects=True, timeout=15.0)
            response.raise_for_status()
            output_path.write_bytes(response.content)
        except Exception as e:
            failures.append((row.image_id, str(e)))

    if failures:
        print(f"\n{len(failures)} failed:")
        for image_id, msg in failures:
            print(f"  {image_id}: {msg}")
    else:
        print("\nAll downloads successful.")
