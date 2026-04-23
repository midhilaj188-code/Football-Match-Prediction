import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("football_dataset.csv")

df = df.select_dtypes(include=['number']).dropna()

# Features
X = df[['halftime_home', 'halftime_away', 'total_goals', 'goal_difference']]

# Target (win / lose)
y = (df['home_points'] > df['away_points']).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)

print("Sample predictions:", model.predict(X_test[:10]))

joblib.dump(model, "model.pkl")

print("Model saved!")