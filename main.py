import joblib
import pandas as pd
from data_processor import load_and_preprocess, train_all_models
from explainability import generate_shap_plots, generate_lime_explanation, visualize_performance


def run_project():
    # 1. Prep Data (Scales data for SVM/Linear Regression baseline)
    print("--- Loading and Preprocessing Data ---")
    X_train, X_test, y_train, y_test = load_and_preprocess('student-mat.csv')

    # 2. Train All Models (Baseline + Advanced Ensembles)
    print("\n--- Training Models ---")
    trained_models, results = train_all_models(X_train, y_train, X_test, y_test)

    # 3. Visualize Performance Comparison
    # This creates the bar chart comparing R2 scores for your report
    print("\n--- Generating Performance Visualization ---")
    visualize_performance(results)

    # 4. Save the Best Model for the Streamlit UI
    # We choose XGBoost as the 'best_model' based on your abstract's focus
    best_model = trained_models["XGBoost"]
    joblib.dump(best_model, 'best_model.pkl')
    print("\n✅ Best model (XGBoost) saved as 'best_model.pkl' for Streamlit.")

    # 5. Explainability (Global and Local)
    print("\n--- Generating Interpretability Reports ---")
    generate_shap_plots(best_model, X_test)

    # We pass .values here to avoid the KeyError we saw earlier
    generate_lime_explanation(best_model, X_train, X_test, instance_index=5)


if __name__ == "__main__":
    run_project()