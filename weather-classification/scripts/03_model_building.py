import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


def build_model(input_path):
    """Build a machine learning model to predict the weather type based on the input data.

    Args:
        input_path: str, path to the input data
    Returns:
        None
    """
    # Load the dataset
    data = pd.read_csv(input_path)

    # Separate features and target variable
    X = data.drop("weather_type", axis=1)
    y = data["weather_type"]

    # One-hot encode the feature columns only
    X = pd.get_dummies(X)

    # Encode the target variable if it's categorical
    label_encoder_y = LabelEncoder()
    y = label_encoder_y.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder_y.classes_,
        yticklabels=label_encoder_y.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    build_model("data/processed/Weather_processed.csv")
