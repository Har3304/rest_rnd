import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('menu_performance.csv')

print(df.head())
print(df.columns)
obj_col = df.select_dtypes(include=['object']).columns.tolist()

print(np.unique(df['menu_item']))

df_dummies = pd.get_dummies(df, columns=obj_col, dtype=int)
print(df_dummies)

encoder = LabelEncoder()
df['menu_item'] = encoder.fit_transform(df['menu_item'])

df = pd.get_dummies(df, columns=['category'], dtype=int)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

X=df.drop('menu_item', axis=1)
y=df['menu_item']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
