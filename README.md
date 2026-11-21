# Student Depression Classification and Clustering

This CSE422 Artificial Intelligence project uses Python to classify depression severity labels in university student survey data and explore patterns with K-means clustering. The notebook was developed in Google Colab.

[Open the notebook in Google Colab](https://colab.research.google.com/github/Fayaz2010/Artificial-Intelligence-Project/blob/main/CSE422_Student_Depression_Project.ipynb)

## Run the notebook

1. Download [`13.csv`](13.csv) and keep its original filename.
2. Open the notebook in Google Colab and select **Runtime > Run all**.
3. When the first cell prompts you to upload a file, select `13.csv`. The remaining cells load the CSV, train the models, and display the results.

The notebook uses pandas, NumPy, Matplotlib, seaborn, scikit-learn, and TensorFlow/Keras.

## Data and workflow

[`13.csv`](13.csv) contains 1,977 survey records and 37 columns before cleaning. It includes demographic and academic information, questionnaire responses, and scores and labels for anxiety, stress, and depression. The classification target is `Depression Label`.

1. Remove duplicate rows and rows with missing values, encode categorical columns, and standardize the feature values.
2. Plot the label distribution and a feature correlation heatmap.
3. Compare a decision tree, a 5-nearest-neighbors classifier, and a feedforward neural network using a stratified 80/20 train/test split.
4. Fit K-means with three clusters, visualize the clusters in two dimensions with PCA, and calculate the silhouette score.

## Results

These test metrics come from the saved outputs in [`CSE422_Student_Depression_Project.ipynb`](CSE422_Student_Depression_Project.ipynb).

| Classifier | Accuracy | Weighted precision | Weighted recall | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| Decision tree | 100.0% | 1.000 | 1.000 | 1.000 |
| K-nearest neighbors | 69.8% | 0.716 | 0.698 | 0.689 |
| Neural network | 89.8% | 0.899 | 0.898 | 0.891 |

The notebook also displays a confusion matrix for each classifier. The three-cluster K-means run produced a silhouette score of 0.139.

Interpret the classifier scores in light of potential data leakage. The feature set includes `Depression Value`, which can directly reveal the target label, and the feature scaler is fitted on the full dataset before the train/test split.
