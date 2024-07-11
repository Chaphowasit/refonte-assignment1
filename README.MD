# Weather Classification Project

This project demonstrates a complete data science workflow using the Weather dataset. The workflow includes data cleaning, exploratory data analysis (EDA), model building, and visualization.

## Project Structure

```
refonte-assignment1/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── weather-classification/
    ├── data/
    │   ├── processed/
    │   │   └── Weather_processed.csv
    │   └── raw/
    │       └── Weather.csv
    ├── notebooks/
    │   ├── 01_data_cleaning.ipynb
    │   ├── 02_EDA.ipynb
    │   ├── 03_model_building.ipynb
    │   └── 04_visualization.ipynb
    ├── results/
    │   ├── figures/
    │   └── results_summary.md
    └── scripts/
        ├── 01_data_cleaning.py
        ├── 02_EDA.py
        ├── 03_model_building.py
        └── 04_visualization.py
```

## How to Activate Virtual Environments

1. **Create Virtual Environment**: (I used Python 3.12.4)

   - If you haven't already created a virtual environment, run the following command in your terminal or command prompt:

     ```bash
     python -m venv .venv
     ```

2. **Activate Virtual Environment**:

   - To activate the virtual environment, run the appropriate command based on your operating system:

     - **Windows**:

       ```bash
       .venv\Scripts\activate
       ```

     - **macOS and Linux**:

       ```bash
       source .venv/bin/activate
       ```

     After activation, you should see `(.venv)` prefixed to your terminal prompt, indicating that the virtual environment is active.

3. **Install Dependencies**:

   - Once the virtual environment is active, install the required dependencies using pip. Assuming you have a `requirements.txt` file listing your dependencies:

     ```bash
     pip install -r requirements.txt
     ```

## Running Scripts and Notebooks

1. **Navigate to Project Directory**:

   - Open your terminal or command prompt and change directory (`cd`) to your project directory.

     ```bash
     cd path/to/weather-classification/
     ```

2. **Example Commands**:

   - Use the following examples to run specific scripts or notebooks within your project:

     - **Running Data Cleaning Script**:

       ```bash
       python scripts/01_data_cleaning.py
       ```

     - **Running EDA Script**:

       ```bash
       python scripts/02_EDA.py
       ```

     - **Running Model Building Script**:

       ```bash
       python scripts/03_model_building.py
       ```

     - **Running Visualization Script**:

       ```bash
       python scripts/04_visualization.py
       ```

     - **Running Jupyter Notebooks**:
       If using Jupyter notebooks (`*.ipynb`), start the Jupyter server:
       ```bash
       jupyter notebook
       ```
       Then, navigate to `notebooks/` and open the desired notebook (e.g., `01_data_cleaning.ipynb`, `02_EDA.ipynb`, etc.).

3. **Execution Order**:
   - Ensure you run scripts or notebooks in the order necessary for your workflow (e.g., data cleaning before EDA and model building).

## Requirements

- Python 3.6+
- pandas
- seaborn
- matplotlib
- scikit-learn
