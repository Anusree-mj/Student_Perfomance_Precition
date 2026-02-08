import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def generate_shap_plots(model, X_test):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig('shap_summary.png')
    print("shap summary saved as 'shap_summary.png'")


def generate_lime_explanation(model, X_train, X_test, instance_index=0):
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns.tolist(),
        class_names=['G3'],
        mode='regression'
    )

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
    names = list(results.keys())
    r2_values = [results[name]['R2'] for name in names]

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    ax = sns.barplot(x=r2_values, y=names, palette="viridis")

    plt.title('Comparison of Model Accuracy (R2 Score)', fontsize=15)
    plt.xlabel('R2 Score (Higher is Better)', fontsize=12)
    plt.xlim(0, 1.0)

    ax.bar_label(ax.containers[0], fmt='%.3f', padding=3)

    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.show()
