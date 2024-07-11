import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def perform_eda(input_path):
    """Perform Exploratory Data Analysis on the input data and save the results as figures.

    Args:
        input_path: str, path to the input data
    Returns:
        None
    """

    # Load data
    data = pd.read_csv(input_path)

    # check if results folder exists
    if not os.path.exists("results/figures"):
        os.makedirs("results/figures")

    # Basic statistics
    print("Basic Statistics:")
    print(data.describe())

    # Select numeric columns for correlation matrix
    numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns

    # Correlation Matrix
    print("Generating Correlation Matrix...")
    corr_matrix = data[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.savefig("results/figures/correlation_matrix.png")
    plt.close()

    # Distribution Plots for Numerical Variables
    print("Generating Distribution Plots...")
    plt.figure(figsize=(12, 10))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(3, 3, i)
        sns.histplot(data[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("results/figures/distribution_plots.png")
    plt.close()

    # Categorical Plots
    print("Generating Categorical Plots...")
    categorical_cols = ["season", "location", "weather_type"]
    plt.figure(figsize=(14, 12))
    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(3, 1, i)
        sns.countplot(data=data, x=col, hue="weather_type", palette="Set2")
        plt.title(f"Countplot of {col} by weather_type")
        plt.xlabel(col)
        plt.ylabel("Count")

        # Get current Axes object and legend information
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()

        # Only add legend if labels are found
        if labels:
            ax.legend(
                handles=handles, labels=labels, title="Weather Type", loc="upper right"
            )
        else:
            pass
    plt.tight_layout()
    plt.savefig("results/figures/categorical_plots.png")
    plt.close()

    # Box Plots
    print("Generating Box Plots...")
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=data, x="season", y="temperature", hue="weather_type", palette="Set3"
    )
    plt.title("Boxplot of Temperature across Seasons by weather_type")
    plt.xlabel("Season")
    plt.ylabel("Temperature")
    plt.legend(loc="upper right")
    plt.savefig("results/figures/boxplot_season_temperature.png")
    plt.close()

    print("EDA completed.")


if __name__ == "__main__":
    perform_eda("data/processed/Weather_processed.csv")
