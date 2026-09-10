# HCAI Project 2 

## Overview

This project focuses on **Explainable Artificial Intelligence (XAI)** and investigates how machine learning models can provide interpretable predictions. The project uses the **Palmer Penguins** dataset and compares different classification approaches while considering both predictive performance and model complexity.

The project consists of five main tasks:

1. Decision Tree and model visualization
2. Regularization and model complexity
3. Logistic Regression
4. Counterfactual explanations
5. Feature Effect Plots using PDP and ALE

The application provides an interactive Streamlit dashboard for exploring the trained models and their explanations.

---

## Dataset

The project uses the **Palmer Penguins** dataset. The target variable is the penguin species, with three possible classes:

* Adelie
* Gentoo
* Chinstrap

The dataset contains numerical, categorical, and binary features, including:

* Bill length
* Bill depth
* Flipper length
* Body mass
* Species
* Sex
* Island
* Year

The dataset is used to investigate both model performance and different forms of model explanation.

---

## Task 1 – Decision Tree

A **Decision Tree classifier** is trained on the Palmer Penguins dataset.

The resulting decision tree is visualized in the application, allowing the decision-making structure of the model to be inspected directly. The number of leaves is also displayed as a measure of model complexity.

The selected model achieved:

* **Test Accuracy:** 94.0%
* **Number of Leaves:** 3
* **λ:** 0.01

The small number of leaves makes the resulting tree relatively easy to interpret while still achieving high classification accuracy.

---

## Task 2 – Regularization

The second task investigates the relationship between **model complexity and predictive performance**.

For the Decision Tree, the number of leaves is used as the complexity measure:

> Ω(f) = number of leaves

Different levels of regularization are considered, and the model is selected based on a trade-off between test accuracy and model complexity.

The selected model uses **λ = 0.01** and contains **3 leaves**, achieving a test accuracy of **94.0%**.

This demonstrates how regularization can be used to avoid unnecessarily complex models while maintaining good predictive performance.

---

## Task 3 – Logistic Regression

The regularization analysis is repeated using **Logistic Regression**.

For Logistic Regression, the number of non-zero weights is used as the complexity measure. This provides an indication of how many model parameters contribute to the final prediction.

The selected model achieved:

* **Test Accuracy:** 62.7%
* **Non-zero weights:** 3
* **λ:** 0.01

Compared with the Decision Tree, Logistic Regression achieved substantially lower accuracy on the test set.

---

## Task 4 – Counterfactual Explanations

The fourth task focuses on **counterfactual explanations**.

For a selected input example, the user can specify a desired target class. The system then searches for nearby samples that belong to the desired class.

The candidate points are ranked according to a **MAD-weighted L1 distance**, where the Median Absolute Deviation (MAD) is used to account for differences in feature scales.

The interface displays the best counterfactual examples and allows the user to investigate how a prediction could change when the input features are modified.

The implementation also considers different feature types, including numerical, binary, and categorical features.

---

## Task 5 – Feature Effect Plots

The final task investigates how individual numerical features affect model predictions.

Two model-agnostic explanation techniques are implemented:

* **Partial Dependence Plots (PDP)**
* **Accumulated Local Effects (ALE)**

The plots show the effect of a selected numerical feature on the predicted probability of each of the three penguin species.

Both methods are implemented manually rather than relying on a pre-built library implementation. This provides a better understanding of how the explanation methods work internally.

---

## Model Selection and Comparison

The two main classification approaches were compared using their predictive performance and respective complexity measures.

| Model               |    λ | Test Accuracy |         Complexity |
| ------------------- | ---: | ------------: | -----------------: |
| Decision Tree       | 0.01 |     **94.0%** |           3 leaves |
| Logistic Regression | 0.01 |         62.7% | 3 non-zero weights |

The Decision Tree performed considerably better than Logistic Regression on the test set. This suggests that the relationships between the input features and penguin species are better captured by the tree-based model in this experiment.

Although both selected models have a complexity value of 3, the measures are defined differently. For the Decision Tree, complexity is measured by the number of leaves, while for Logistic Regression it is measured by the number of non-zero weights. Therefore, these values should not be interpreted as directly equivalent.

Overall, the comparison highlights the importance of considering **both predictive performance and model complexity** when developing explainable machine learning systems.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib
* Palmer Penguins dataset

---

## How to Run

Install the required dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python3 -m streamlit run project2/app.py
```

The application can then be accessed through the local Streamlit URL shown in the terminal.

---

## Conclusion

This project demonstrates several approaches to making machine learning models more interpretable. Decision Tree visualization provides a direct representation of the model's decision process, while counterfactual explanations and feature effect plots provide additional ways of understanding model predictions.

The comparison between the Decision Tree and Logistic Regression also shows that model interpretability needs to be considered together with predictive performance and model complexity.
