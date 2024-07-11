import pandas as pd
import seaborn as sns


def clean_data(input_path, output_path):
    """
    Clean raw data and save processed data to output_path
    Args:
        input_path: str, path to raw data
        output_path: str, path to save processed data
    Returns:
        None
    """
    # Load raw data
    data = pd.read_csv(input_path)
    # change the column names to lowercase and replace spaces with underscores
    data.columns = [col.strip().replace(" ", "_").lower() for col in data.columns]

    # Save processed data
    data.to_csv(output_path, index=False)

    # Print path to processed data
    print("Data cleaning completed and saved to", output_path)


if __name__ == "__main__":
    clean_data(
        "data/raw/Weather.csv",
        "data/processed/Weather_processed.csv",
    )
