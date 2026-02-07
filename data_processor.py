import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error


def load_and_preprocess(filepath):
    df = pd.read_csv(filepath, sep=';')

    # Label Encoding for categorical strings
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop(['G3'], axis=1)
    y = df['G3']

    # SVM and Linear Regression perform better with scaled features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


def train_all_models(X_train, y_train, X_test, y_test):
    models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "SVM (SVR)": SVR(kernel='rbf'),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "LightGBM": lgb.LGBMRegressor(random_state=42)
    }

    results = {}
    trained_objects = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)

        results[name] = {"R2": r2, "MSE": mse}
        trained_objects[name] = model
        print(f"{name} -> R2: {r2:.4f}")

    return trained_objects, results