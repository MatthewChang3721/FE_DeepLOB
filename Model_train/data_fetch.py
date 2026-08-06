import shutil
from pathlib import Path
from config import copy_data_path, fetch_file_path, raw_data_path, train_num_files, val_num_files
import pandas as pd

base_path = Path(copy_data_path)
target_dir = Path(raw_data_path)
primary_csv = Path(fetch_file_path)
required_num_files = train_num_files + val_num_files + 5 + 3 # 5 for window normalize, 3 for OOS.

def fetch_files(
    base_path: Path,
    target_dir: Path,
    primary_csv: Path,
) -> None:
    if target_dir.exists():
        for f in target_dir.iterdir():
            if f.is_file():
                f.unlink()
    target_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(primary_csv)[-required_num_files:]

    copied = 0
    missing = 0

    for _, row in df.iterrows():
        instrument_id = str(row["InstrumentID"]).strip()
        trading_day = str(row["TradingDay"]).strip().replace("-", "")

        source_file = base_path / trading_day / f"{instrument_id}_{trading_day}.csv"
        dest_file = target_dir / source_file.name

        if not source_file.is_file():
            print(f"Missing: {source_file}")
            missing += 1
            continue

        shutil.copy2(source_file, dest_file)
        copied += 1

    print(f"\nFinished: {copied} copied, {missing} missing.")


if __name__ == "__main__":
    fetch_files(base_path, target_dir, primary_csv)