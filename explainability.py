import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def generate_shap_plots(model, X_test):
    print("Generating SHAP Explanations...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    # Global Feature Importance
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig('shap_summary.png')
    print("SHAP Summary plot saved as 'shap_summary.png'")


def generate_lime_explanation(model, X_train, X_test, instance_index=0):
    print(f"Generating LIME for student index {instance_index}...")
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns.tolist(),  # Convert to list for compatibility
        class_names=['G3'],
        mode='regression'
    )

    # FIX: Use .values to pass a raw NumPy array to LIME
    data_row = X_test.iloc[instance_index].values

    exp = explainer_lime.explain_instance(
        data_row,
        model.predict,
        num_features=10
    )
    exp.save_to_file('lime_report.html')
    print("LIME report saved as 'lime_report.html'")

    import seaborn as sns
    import matplotlib.pyplot as plt

def visualize_performance(results):
    # Convert results dictionary to a DataFrame for Seaborn
    names = list(results.keys())
    r2_values = [results[name]['R2'] for name in names]

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Create the bar plot
    ax = sns.barplot(x=r2_values, y=names, palette="viridis")

    # Add labels and title
    plt.title('Comparison of Model Accuracy (R2 Score)', fontsize=15)
    plt.xlabel('R2 Score (Higher is Better)', fontsize=12)
    plt.xlim(0, 1.0)  # Accuracy scale

    # Add value labels on the bars
    ax.bar_label(ax.containers[0], fmt='%.3f', padding=3)

    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.show()

# To use it in your main.py:
# trained_models, results = train_all_models(X_train, y_train, X_test, y_test)
# visualize_performance(results)