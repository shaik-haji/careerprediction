import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Load the modified dataset
df = pd.read_csv(r"D:\Career\Careerprediction\Dataset\Career_Prediction_Modified.csv")

# Encode categorical features
label_encoders = {}
categorical_columns = ["Coding_Skills", "Communication_Skills", "Leadership_Experience", 
                        "Preferred_Work_Environment", "Career"]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"Encoding for {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Split features and target
X = df.drop(columns=["Career"])
y = df["Career"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")

# Save the model and label encoders
pickle.dump(rf_model, open(r'D:\Career\Careerprediction\Model\model.pkl', 'wb'))
pickle.dump(label_encoders, open(r'D:\Career\Careerprediction\Model\label_encoders.pkl', 'wb'))
