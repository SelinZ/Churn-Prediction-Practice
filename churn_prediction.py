import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from google.cloud import storage

df = pd.read_csv("/Users/selinzobu/Desktop/ChurnPred/churn_data.csv", header = 0)
#print(df.head())

'''Using BigQuery
from google.cloud import bigquery
client = bigquery.Client()
query = "SELECT * FROM `your_project.dataset.churn_table`"
df = client.query(query).to_dataframe()
'''

df.fillna(0, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Optionally, you can fill NaN values with the median or mean of the column
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Identify categorical columns
categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                       'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                       'StreamingMovies', 'Contract', 'PaperlessBilling', 
                       'PaymentMethod', 'Churn']  # You can add or remove columns here

# Create dummy variables
df_dummies = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

#Split data into training and testing sets

X = df_dummies.drop(columns=['Churn_Yes', 'customerID'])
y = df_dummies['Churn_Yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model (e.g., RandomForest or XGBoost)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Evaluate the model

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Deploy the Model to GCP: Save the trained model
joblib.dump(model, "churn_model.pkl")

#Upload it to Google Cloud Storage (GCS)

client = storage.Client()
bucket = client.bucket("your-bucket-name")
blob = bucket.blob("models/churn_model.pkl")
blob.upload_from_filename("churn_model.pkl")

print("Model uploaded to GCS!")

'''
# Save the model locally instead of uploading to Google Cloud
import joblib

# Assuming 'model' is your trained model
joblib.dump(model, 'churn_model.pkl')
'''