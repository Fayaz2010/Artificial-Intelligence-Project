# Student Depression Classification and Clustering

This CSE422 Artificial Intelligence project uses Python to classify depression severity labels in university student survey data and explore patterns with K-means clustering. The notebook was developed in Google Colab.

[Open the notebook in Google Colab](https://colab.research.google.com/github/Fayaz2010/Artificial-Intelligence-Project/blob/main/CSE422_Student_Depression_Project.ipynb)

## Run the notebook

1. Download [`13.csv`](13.csv) and keep its original filename.
2. Open the notebook in Google Colab and select **Runtime > Run all**.
3. If Colab prompts for a file, upload `13.csv`. The remaining cells load the CSV, train the models, and display the results.

### Run locally in VS Code or Jupyter

1. Keep `13.csv` in the project folder and use that folder as the notebook's working directory.
2. Install the dependencies in the Python environment used by your notebook kernel.

   ```sh
   python -m pip install ipykernel pandas numpy matplotlib seaborn scikit-learn tensorflow
   ```

3. Select that Python environment as the notebook kernel and run all cells.

## Data and workflow

[`13.csv`](13.csv) contains 1,977 survey records and 37 columns before cleaning. It includes demographic and academic information, questionnaire responses, and scores and labels for anxiety, stress, and depression. The classification target is `Depression Label`.

1. Remove duplicate rows and rows with missing values, encode categorical columns, and standardize the feature values.
2. Plot the label distribution and a feature correlation heatmap.
3. Compare nine classifiers using a stratified 80/20 train/test split: decision tree, 5-nearest neighbors, neural network, logistic regression, Gaussian discriminant analysis (linear discriminant analysis with shared covariance), Gaussian naive Bayes, random forest, gradient boosting, and support vector classifier.
4. Fit K-means with three clusters, visualize the clusters in two dimensions with PCA, and calculate the silhouette score.

## Results

Run the notebook to see the current test accuracy, weighted precision, weighted recall, weighted F1 score, and confusion matrix for each classifier.

The eight scikit-learn classifiers were verified locally on the same 394 test records. The neural network scores come from the original Colab run.

| Classifier | Accuracy | Weighted precision | Weighted recall | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| Decision tree | 100.0% | 1.000 | 1.000 | 1.000 |
| K-nearest neighbors | 69.8% | 0.716 | 0.698 | 0.689 |
| Neural network | 89.8% | 0.899 | 0.898 | 0.891 |
| Logistic regression | 96.7% | 0.968 | 0.967 | 0.966 |
| Gaussian discriminant analysis | 91.1% | 0.916 | 0.911 | 0.911 |
| Gaussian naive Bayes | 80.2% | 0.802 | 0.802 | 0.801 |
| Random forest | 99.7% | 0.997 | 0.997 | 0.997 |
| Gradient boosting | 100.0% | 1.000 | 1.000 | 1.000 |
| Support vector classifier | 90.4% | 0.906 | 0.904 | 0.901 |

K-means with three clusters produced a silhouette score of 0.139.

Interpret the classifier scores in light of potential data leakage. The feature set includes `Depression Value`, which can directly reveal the target label, and the feature scaler is fitted on the full dataset before the train/test split.
