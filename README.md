🩺 Diabetes Prediction
This project builds a machine learning model to predict whether a person has diabetes based on diagnostic health measurements. It uses a public dataset from Kaggle and implements data exploration, cleaning, model training, evaluation, and prediction.

📁 Dataset
Source: Kaggle - Diabetes Dataset

Attributes:

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction (renamed to DPF)

Age

Outcome (Target variable: 0 = No Diabetes, 1 = Has Diabetes)

🧰 Libraries Used
Pandas, NumPy – Data manipulation

Matplotlib, Seaborn – Data visualization

Scikit-learn – Machine learning (training, evaluation, prediction)

📊 Workflow
1. Data Exploration
View dataset shape, datatypes, and basic statistics

Visualize class distribution using bar plots

2. Data Cleaning
Replace invalid zero values in certain columns with NaN

Fill missing values using mean or median based on the feature distribution

Standardize features using StandardScaler

3. Model Selection
Evaluate multiple models using GridSearchCV and ShuffleSplit cross-validation:

Logistic Regression

Decision Tree

Random Forest (best performing)

Support Vector Machine (SVM)

4. Model Evaluation
Evaluate model performance using:

Accuracy score

Confusion matrix (for both test and training sets)

Classification report (Precision, Recall, F1-score)

5. Prediction Function
A custom function predict_diabetes() allows predictions based on new input data (e.g., from a user form or API).

✅ Example Predictions
python
Copy
Edit
# Example usage
predict_diabetes(2, 81, 72, 15, 76, 30.1, 0.547, 25)
# Output: "Oops! You have diabetes." or "Great! You don't have diabetes."
🚀 How to Run
Clone the repository or copy the code into a Jupyter notebook.

Install required libraries:

nginx
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn
Ensure diabetes.csv is downloaded and update the path in the script.

Run all cells to build and evaluate the model.

📌 Notes
Random Forest showed the best accuracy and generalization performance.

The model was fine-tuned using cross-validation and hyperparameter tuning.

Feature scaling is crucial for optimal performance of algorithms like SVM and Logistic Regression.

📈 Future Improvements
Deploy the model using Flask or Streamlit for real-time predictions.

Perform feature engineering or try advanced ensemble methods like XGBoost or LightGBM.

Integrate SHAP or LIME for explainability.
