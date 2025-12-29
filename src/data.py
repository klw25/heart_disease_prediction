from pathlib import Path
import pandas as pd

def load_data():
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parent.parent
    data_file_path = project_root / "data" / "heart.csv"
    df = pd.read_csv(data_file_path)

    return df