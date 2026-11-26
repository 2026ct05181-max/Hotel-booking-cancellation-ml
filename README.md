# Hotel Booking Cancellation Prediction

## 1. Problem Statement
Predict whether a hotel booking will be cancelled (`is_canceled`) from booking details, and compare five classification algorithms (Logistic Regression, Decision Tree, kNN, Gaussian Naive Bayes, Random Forest) on the same dataset using Accuracy, AUC, Precision, Recall, F1 and MCC.

## 2. Dataset Description
- **Source:** Kaggle, Hotel Booking Demand (`hotel_bookings.csv`), https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
- **Why chosen:** realistic business problem (overbooking and revenue management); more than 12 features, more than 1,000 rows, real missing values, mixed feature types and an imbalanced binary target.
- **Instances:** 119,390 raw, 86,944 after cleaning | **Features:** 31 raw
- **Target variable:** `is_canceled` (0 = not cancelled, 1 = cancelled)
- **Feature types (raw):** 19 numeric, 12 categorical
- **Class distribution:** raw 62.96 % not cancelled / 37.04 % cancelled; after cleaning 72.69 % / 27.31 %
- **Missing values:** company (94.31 %), agent (13.69 %), country (0.41 %), children (4 rows)
- **Duplicates:** 31,994 duplicate rows in the raw data

### Preprocessing
- Dropped leakage columns `reservation_status` and `reservation_status_date` (they reveal the outcome)
- Dropped `company` (about 94 % missing) and `agent` (high-cardinality ID)
- Removed duplicate rows and zero-guest bookings
- Added `total_nights` and `total_guests`
- Median imputation + StandardScaler for numeric features; most-frequent imputation + One-Hot Encoding (rare categories grouped) for categorical features
- Stratified 80:20 train/test split, `random_state=42`
- All preprocessing is inside a scikit-learn Pipeline, so it is fitted on the training data only

## 3. GitHub Repository Link
<PASTE YOUR REPO URL HERE>

## 4. Models Used
### Comparison table (test set)
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7921 | 0.8412 | 0.6721 | 0.4658 | 0.5502 | 0.4327 |
| Decision Tree | 0.8178 | 0.8688 | 0.6876 | 0.6098 | 0.6464 | 0.5259 |
| kNN | 0.7962 | 0.8250 | 0.6418 | 0.5742 | 0.6061 | 0.4706 |
| Naive Bayes | 0.5184 | 0.7490 | 0.3520 | 0.9076 | 0.5072 | 0.2731 |
| Random Forest (Ensemble) | 0.8440 | 0.9017 | 0.7718 | 0.6090 | 0.6808 | 0.5866 |

### Graphs
![Grouped bar chart](images/metrics_grouped_bar.png)
![Heatmap](images/metrics_heatmap.png)
![Individual metrics](images/metrics_individual.png)
![ROC curves](images/roc_curves.png)
![Confusion matrices](images/confusion_matrices.png)

## 5. Observations
| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Accuracy 0.7921, AUC 0.8412, MCC 0.4327. Fast with almost no overfitting, but the lowest recall (0.4658): it misses more than half of the real cancellations because a linear boundary cannot capture the non-linear patterns. |
| **Decision Tree** | Accuracy 0.8178, AUC 0.8688, MCC 0.5259. Second best overall. Depth limit kept overfitting small (gap 0.0157) and recall (0.6098) is much better than logistic regression. Easy to interpret. |
| **kNN** | Accuracy 0.7961, AUC 0.8250, MCC 0.4704. Close to logistic regression but weaker than the tree models. Moderate overfitting (gap 0.0512); distances in 85 dimensions are less meaningful and prediction is slow. |
| **Naive Bayes** | Accuracy 0.5184, AUC 0.7490, MCC 0.2731. Worst model: highest recall (0.9076) but very low precision (0.3520), so it flags most bookings as cancellations. The Gaussian and independence assumptions fit the encoded features poorly. Fastest model. |
| **Random Forest (Ensemble)** | Accuracy 0.8440, AUC 0.9017, Precision 0.7718, MCC 0.5866. Best on 5 of 6 metrics. Averaging 200 trees reduces variance. Overfits the most (gap 0.104) and trains slowest, but generalises best on the test set. |
| **Overall Winner for your dataset?** | **Random Forest**: best average rank (1.33) across the six metrics and the highest accuracy, AUC, precision, F1 and MCC. Decision Tree is a good simpler runner-up. |


## 6. How to Run
```
pip install -r requirements.txt
```
1. Download `hotel_bookings.csv` from Kaggle and place it in the same folder (or update the path in the notebook).
2. Open `model/ML_Assignment_2_Cognizant.ipynb` in Jupyter or Google Colab and run all cells.
3. Test data used in the experiments is in `test_data.csv`.
