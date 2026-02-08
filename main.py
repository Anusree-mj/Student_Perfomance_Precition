import joblib
import pandas as pd
from data_processor import load_and_preprocess, train_all_models
from explainability import generate_shap_plots, generate_lime_explanation, visualize_performance


def run_project():
    # 1. preparing data
    print("loading and preprocessing data")
    X_train, X_test, y_train, y_test = load_and_preprocess('student-mat.csv')

    # 2. training all models
    print("\n training models")
    trained_models, results = train_all_models(X_train, y_train, X_test, y_test)

    # 3. visualizing perfomance comparison, creating the bar chart comparing r2 scores of the report
    print("\n generating performance visualization")
    visualize_performance(results)

    # 4. finding and saving the best model
    best_model_name = max(results, key=lambda k: results[k]["R2"])
    best_model = trained_models[best_model_name]

    print(f"best model selected dynamically: {best_model_name}")
    joblib.dump(best_model, 'best_model.pkl')

    # 5. explainability
    print("\ngenerating explainaibiltiy plot")
    print("\n shap::::")
    generate_shap_plots(best_model, X_test)

    print("\n lime::::")
    generate_lime_explanation(best_model, X_train, X_test, instance_index=5)


if __name__ == "__main__":
    run_project()