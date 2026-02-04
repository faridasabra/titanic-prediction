# Titanic Survival Prediction

This project predicts which passengers survived the disaster based on their characteristics.

## Project Description

Using passenger data from the Titanic, this project builds a machine learning model to predict survival outcomes. The model learns patterns from known survival data and applies them to make predictions on new passengers. It's a binary classification problem: survived or didn't survive.

## Datasets

The Kaggle competition provides three datasets:

* **Training set**: Contains passenger data with known survival outcomes for model development
* **Test set**: Passenger data without survival labels for making final predictions
* **Sample submission**: Template showing the required submission format

## Data Features

Each dataset includes these passenger attributes:

* `PassengerId` (typically used as an index)
* `Survived` (target variable)
* `Pclass` (ticket class: 1st, 2nd, or 3rd)
* `Name`
* `Sex`
* `Age`
* `SibSp` (siblings/spouses aboard)
* `Parch` (parents/children aboard)
* `Ticket`
* `Fare`
* `Cabin`
* `Embarked` (port where passenger boarded)

## Data Preprocessing and Feature Engineering

The data preparation pipeline includes:

* **Combining datasets**: Merging training and test sets ensures consistent preprocessing
* **Handling missing values**:
   * `Age`: Filled using mean values, sometimes stratified by other features like class or title
   * `Embarked`: Filled with the most common port
   * `Cabin`: Usually dropped due to excessive missing data
* **Creating new features**:
   * Extract `Title` from passenger names (Mr., Mrs., Miss., etc.)
   * Calculate `FamilySize` by combining `SibSp` and `Parch`
   * Convert categorical variables (`Pclass`, `Sex`, `Embarked`) into dummy variables
   * Remove non-predictive columns like `PassengerId`, `Name`, and `Ticket` after extracting useful information

## Machine Learning Models

While the specific models used aren't detailed in the notebook, typical algorithms for this problem include:

* Logistic Regression
* Support Vector Machines (SVM)
* Decision Trees
* Random Forest
* K-Nearest Neighbors (KNN)
* Naive Bayes

## Getting Started

To run this project:

1. Clone the repository
2. Install required Python libraries:
   * NumPy
   * Pandas
   * Matplotlib
   * Seaborn
   * Scikit-Learn
   * SciPy
   * IPython
   * StatsModels
   * Patsy
3. Download `train.csv` and `test.csv` from the [Kaggle Titanic competition page](https://www.kaggle.com/c/titanic)
4. Run the Jupyter Notebook (`titanic.ipynb`) to explore the data, build models, and generate predictions
