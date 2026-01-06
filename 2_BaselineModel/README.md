# Baseline Model

**[Notebook](baseline_model.ipynb)**

## Baseline Model Results

### Model Selection
- **Baseline Model Type:** Linear Regression
- **Rationale:** The linear regression model was used as the baseline model which our neural network will be meassured on. It is easily Interpretable and helps us to find out if our data is clean and makes sense. 
It is very easy to set up and does not overcomplicate the idea.

### Model Performance
- **Evaluation Metric:**  R²
- **Performance Score:**  R² of 0,7224 
- **Cross-Validation Score:** 0.6987 (+/- 0.1169)

### Evaluation Methodology
- **Data Split:** [Train/Validation/Test split ratios: 67/16,5/16,6]
- **Evaluation Metrics:** 
MSE (Mean squared Error) - for identifiying outliers, RMSE (Root Mean Squared Error) - easier interpretable, MAE (Mean Absolut Error) - less resposnisve to outliers, 
R2 - Score - to meassure the overall fit

### Metric Practical Relevance
MSE: Can help to find out, if values are meassurement errors and therefore influence the model in a wronge way. Without that wrong values could remain undetected and lead to overestimations.
RMSE: As this value gives actual numbers, it can help to make decisions based on monetary values. 
R2: Helps to estimate, how reliable the model actually is.

[Explain the practical relevance and business impact of each chosen evaluation metric. How do these metrics translate to real-world performance and decision-making? What do the metric values mean in the context of your specific problem domain?]

## Next Steps
This baseline model serves as a reference point for evaluating more sophisticated models in the [Model Definition and Evaluation](../3_Model/README.md) phase.
