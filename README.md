# Obesity Level Classification

## a. Problem statement
Predict a person's obesity level (7 classes, multi-class classification) from eating habits and physical condition attributes. Five classification models (Logistic Regression, Decision Tree, kNN, Gaussian Naive Bayes, SVM) are trained on the same dataset, compared using six evaluation metrics, and demonstrated in an interactive Streamlit app.

## b. Dataset description
- **Source:**  *Estimation of obesity levels based on eating habits and physical condition* (on Kaggle).
- **Instances:** 2,111
- **Features used:** 14 (16 original features minus `Height` and `Weight`)
- **Target:** `NObeyesdad` - 7 classes: Insufficient_Weight, Normal_Weight, Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III
- **Note on data:** about 77% of the records are synthetic (generated with SMOTE) and 23% were collected from users in Mexico, Peru and Colombia.
- **Why Height and Weight were dropped:** the class labels are defined from BMI, which is computed from these two columns, so keeping them leaks the answer and makes the problem trivial.

| Feature | Meaning |
|---|---|
| Gender, Age | Demographics |
| family_history_with_overweight | Family history of overweight |
| FAVC | Frequent consumption of high-calorie food |
| FCVC | Frequency of vegetable consumption |
| NCP | Number of main meals per day |
| CAEC | Eating between meals |
| SMOKE | Smoker or not |
| CH2O | Daily water consumption |
| SCC | Monitors calorie intake |
| FAF | Physical activity frequency |
| TUE | Time using technology devices |
| CALC | Alcohol consumption |
| MTRANS | Transportation used |

**Preprocessing:** numeric features standardised (StandardScaler), categorical features one-hot encoded; 80/20 stratified train-test split (random_state=42). Preprocessing is bundled with each saved model as a scikit-learn Pipeline.

## c. GitHub repository link
[(https://github.com/KhushiGangopadhyay/ML_Assignment_BITS_PILANI/tree/main)]

Live Streamlit app: [(https://mlassignmentbitspilani-n5n7dqsxpw8kes29wnzeus.streamlit.app/)]

Repo structure:
```
app.py               # Streamlit app
requirements.txt
README.md
test_data.csv        # held-out test split used in the app
model/               # saved models (.joblib), metrics.csv, training notebook
```

## d. Models used

### Comparison table
Multi-class metrics use weighted averaging; AUC is one-vs-rest (weighted).

ML Model Name,Accuracy,AUC,Precision,Recall,F1,MCC
Logistic Regression,0.6241,0.886,0.6087,0.6241,0.5925,0.5676
Decision Tree,0.721,0.8558,0.7207,0.721,0.7197,0.6745
kNN,0.7139,0.9359,0.7204,0.7139,0.6932,0.6717
Naive Bayes,0.4988,0.8166,0.536,0.4988,0.4479,0.4387
Support Vector Machine,0.7589,0.9507,0.7607,0.7589,0.7561,0.7192


### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy is 0.624, F1 is 0.593 and MCC is 0.568, so it is the second-weakest model. A linear boundary struggles with the middle classes. Overweight_Level_II has a recall of only 0.121, and 24 of its 58 samples were predicted as Obesity_Type_I. Normal_Weight recall is also low (0.397). It does well on the extreme classes, with Obesity_Type_III at F1 0.941. |
| Decision Tree | Accuracy is 0.721, F1 is 0.720 and MCC is 0.674, which is solid and its precision and recall are balanced. It handles Overweight_Level_II much better than logistic regression (F1 0.63), but its AUC (0.856) is lower than the kNN’s and SVM’s, because tree probabilities are coarse. Normal_Weight is still the weak spot (F1 0.50), confused with Obesity_Type_I and Overweight_Level_I.|
| kNN | Accuracy is 0.714 and MCC is 0.672, with a high AUC of 0.936, so it ranks the classes well even when its hard predictions are off. Its main failure is Normal_Weight, with a recall of only 0.259. Only 15 of 58 were found, and the rest were spread across neighbouring classes, mostly Insufficient_Weight (13) and Obesity_Type_I (12). It depends on feature scaling, which the pipeline provides. |
| Naive Bayes | This is the weakest model (Accuracy 0.499, F1 0.448, MCC 0.439). It assumes features are independent and Gaussian, which doesn’t hold for one-hot encoded and ordinal lifestyle features. It over-predicts Obesity_Type_II (recall 0.917 but precision only 0.357) and almost never predicts Overweight_Level_II (recall 0.069).|
| Support Vector Machine | This is the best model on every metric (Accuracy 0.759, AUC 0.951, Precision 0.761, Recall 0.759, F1 0.756, MCC 0.719). The RBF kernel captures non-linear boundaries between classes, and it has the best Normal_Weight (F1 0.615) and Insufficient_Weight (F1 0.865) results. It still confuses Overweight_Level_II with Obesity_Type_I |
| **Overall winner for this dataset** | SVM, with the highest scores on all six metrics. Decision Tree and kNN are close behind, and Naive Bayes is clearly the worst.|

## Running locally
```
pip install -r requirements.txt
streamlit run app.py
```
Upload `test_data.csv` in the sidebar, choose a model, and view metrics, the confusion matrix and the classification report.
