# CensusIncomeClassifier

CensusIncomeClassifier predicts income levels using census data through preprocessing, feature encoding, and a Random Forest classification model. It includes data analysis, model training, and evaluation with accuracy and classification metrics.

## How to Use

1. Clone the repository.
2. Place the `census_income.csv` dataset inside the `data/` folder.
3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the classifier:

    ```bash
    python src/classifier.py
    ```

## Dataset

The dataset file `census_income.csv` should be placed inside the `data` folder. You can download it from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/census+income).

## Dependencies

- pandas
- matplotlib
- seaborn
- scikit-learn
