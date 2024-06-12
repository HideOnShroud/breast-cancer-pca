# PCA Breast Cancer Analysis Exercise

## Project Overview

This project performs a Principal Component Analysis (PCA) on the breast cancer dataset from the `sklearn.datasets` library. The analysis aims to reduce the dataset's dimensionality while retaining as much variance as possible. The project also visualizes the data in both 2D and 3D to better understand the distribution and clustering of the data based on the principal components.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Data Description](#data-description)
5. [Code Explanation](#code-explanation)
6. [Results](#results)
7. [Usage](#usage)
8. [Contributing](#contributing)
9. [License](#license)
10. [Acknowledgements](#acknowledgements)

## Getting Started

These instructions will help you set up the project and run the analysis on your local machine.

### Prerequisites

- Python 3.6+
- Jupyter Notebook or Jupyter Lab
- Basic knowledge of Python and data analysis concepts

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/pca-breast-cancer-analysis.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd pca-breast-cancer-analysis
   ```
3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file should include the following packages:
   ```text
   pandas
   plotly
   matplotlib
   seaborn
   scikit-learn
   ```

4. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

5. **Open the `pca_breast_cancer_analysis.ipynb` notebook** and run the cells sequentially.

## Data Description

The breast cancer dataset used in this analysis is part of the `sklearn.datasets` module. It consists of 569 instances of cancer data with 30 feature columns. The dataset contains two classes: 'malignant' and 'benign'. The features are numerical values representing various attributes of the cell nuclei present in the digitized image of a breast mass.

## Code Explanation

### Importing Libraries

We start by importing the necessary libraries for data handling, visualization, and PCA.

```python
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```

### Loading and Preparing the Data

1. **Load the dataset**:
   ```python
   data = load_breast_cancer()
   df = pd.DataFrame(data.data, columns=data.feature_names)
   ```

2. **Visualize correlations**:
   We create a correlation matrix to understand the relationship between features.
   ```python
   plt.figure(figsize=(24, 24))
   sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
   plt.title('Correlation Matrix')
   plt.show()
   ```

3. **Add target variable**:
   ```python
   df['type'] = data.target
   df["type"] = df["type"].map({0: data.target_names[0], 1: data.target_names[1]})
   ```

4. **Standardize the data**:
   Standardization is essential for PCA as it ensures that each feature contributes equally to the analysis.
   ```python
   scaled_data = StandardScaler().fit_transform(data.data)
   df_scaled = pd.DataFrame(scaled_data, columns=data.feature_names)
   df_scaled['type'] = data.target
   df_scaled["type"] = df_scaled["type"].map({0: data.target_names[0], 1: data.target_names[1]})
   ```

### Performing PCA

1. **PCA with 2 components**:
   ```python
   pca = PCA(n_components=2)
   pca_breast_cancer = pca.fit_transform(scaled_data)
   ```

2. **Explained variance**:
   ```python
   explained_variance = pca.explained_variance_ratio_
   print(f'Explained variance by component 1: {explained_variance[0]*100:.2f}%')
   print(f'Explained variance by component 2: {explained_variance[1]*100:.2f}%')
   ```

3. **Create PCA DataFrame**:
   ```python
   df_pca = pd.DataFrame(
       pca_breast_cancer, columns=["Principal Component 1", "Principal Component 2"]
   )
   df_pca["type"] = data.target
   df_pca["type"] = df_pca["type"].map({0: data.target_names[0], 1: data.target_names[1]})
   ```

### Visualizations

1. **2D Scatter Plot**:
   ```python
   fig = px.scatter(
       df_pca,
       x="Principal Component 1",
       y="Principal Component 2",
       color="type",
       title='2D PCA of Breast Cancer Dataset'
   )
   fig.show()
   ```

2. **3D Scatter Plot**:
   ```python
   pca = PCA(n_components=3)
   pca_breast_cancer_3d = pca.fit_transform(scaled_data)
   df_pca_3d = pd.DataFrame(
       pca_breast_cancer_3d, 
       columns=["Principal Component 1", "Principal Component 2", "Principal Component 3"]
   )
   df_pca_3d["type"] = data.target
   df_pca_3d["type"] = df_pca_3d["type"].map(
       {0: data.target_names[0], 1: data.target_names[1]}
   )
   fig_3d = px.scatter_3d(
       df_pca_3d,
       x="Principal Component 1",
       y="Principal Component 2",
       z="Principal Component 3",
       color="type",
       title='3D PCA of Breast Cancer Dataset'
   )
   fig_3d.show()
   ```

## Results

- **Explained Variance**: The two principal components account for a significant amount of the variance in the dataset, making them useful for visualization and analysis.
- **2D Visualization**: The 2D scatter plot shows clear clustering of the two cancer types, indicating that PCA is effective in differentiating between them.
- **3D Visualization**: The 3D scatter plot provides an additional dimension of information, offering a more comprehensive view of the data distribution.

## Usage

To run the analysis, follow these steps:

1. Clone the repository and navigate to the project directory.
2. Install the required packages using `pip install -r requirements.txt`.
3. Open `pca_breast_cancer_analysis.ipynb` in Jupyter Notebook or Jupyter Lab.
4. Execute the cells sequentially to perform the analysis and generate the visualizations.

## Contributing

Contributions are welcome! Please open an issue to discuss any changes or enhancements. You can also submit a pull request with your proposed modifications.

## Acknowledgements

- The breast cancer dataset is provided by the `sklearn.datasets` module.
- The `pandas`, `plotly`, `matplotlib`, `seaborn`, and `scikit-learn` libraries are instrumental in performing the data analysis and visualization.
