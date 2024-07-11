import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def create_visualizations(input_path):
    """Create visualizations from the input data and save them to results/figures/.

    Args:
        input_path: str, path to the input data
    Returns:
        None
    """
    # Load data
    data = pd.read_csv(input_path)

    # Check if results folder exists
    if not os.path.exists("results/figures"):
        os.makedirs("results/figures")

    # Pairplot of all numerical variables
    sns.pairplot(data, hue="weather_type")
    plt.savefig("results/figures/pairplot.png")

    print("Visualizations created and saved to results/figures/")


if __name__ == "__main__":
    create_visualizations("data/processed/Weather_processed.csv")
