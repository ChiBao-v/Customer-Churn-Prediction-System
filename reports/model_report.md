# Model Training Report

## Model Comparison

| model               |   accuracy |   precision |   recall |   f1_score |   roc_auc |
|:--------------------|-----------:|------------:|---------:|-----------:|----------:|
| Logistic Regression |   0.738112 |    0.504303 | 0.783422 |   0.613613 |  0.841298 |
| Random Forest       |   0.782115 |    0.617544 | 0.470588 |   0.534143 |  0.821664 |
| XGBoost             |   0.804116 |    0.66443  | 0.529412 |   0.589286 |  0.842185 |

## Best Model

The best model selected based on ROC-AUC is **XGBoost**.

## Notes

For customer churn prediction, recall and ROC-AUC are important metrics. Recall helps identify more customers who are likely to churn, while ROC-AUC measures the model's ability to distinguish between churn and non-churn customers.