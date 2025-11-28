import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os

top_n_features = 30


def shap_analysis_binary_aa(X, y, genotype, genotype_real_name, feature_names, max_features,
                            model_type='Random Forest'):
    y_binary = (y == genotype).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.4, random_state=42, stratify=y_binary
    )

    if model_type == 'LightGBM':
        model = lgb.LGBMClassifier(random_state=42, learning_rate=0.1, max_depth=10, n_estimators=100,
                                   num_leaves=31, verbose=-1)
    elif model_type == 'XGBoost':
        model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss',
                                  learning_rate=0.1, max_depth=10, n_estimators=300, subsample=1.0)
    elif model_type == 'Random Forest':
        model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None, min_samples_leaf=1,
                                       min_samples_split=2)
    else:
        raise ValueError("not support")

    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    print('********************** shap_values size:', len(shap_values))

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    if model_type == 'Random Forest':
        shap_vals = shap_vals[:, :, 1]

    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

    sorted_indices = np.argsort(mean_abs_shap)[::-1]
    top_indices = sorted_indices[:max_features]

    top_feature_names = [feature_names[i] for i in top_indices]

    aa_mapping = {
        0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K', 9: 'L',
        10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'V', 18: 'W', 19: 'Y',
        20: '*', 21: '-'
    }

    plt.figure(figsize=(12, 10))

    X_train_top = X_train[:, top_indices]
    shap_vals_top = shap_vals[:, top_indices]

    shap.summary_plot(shap_vals_top, X_train_top,
                      feature_names=top_feature_names,
                      show=False,
                      max_display=max_features)

    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xlabel(ax.get_xlabel(), fontsize=20)
    ax.set_ylabel(ax.get_ylabel(), fontsize=20)

    plt.title(f'SHAP summary plot - {genotype_real_name}', fontsize=22)
    plt.tight_layout()
    plt.savefig(f'shap_summary_{model_type.lower()}_aa_genotype_{genotype_real_name}_revision.png', dpi=600,
                bbox_inches='tight')
    plt.close()

    for i, feat_idx in enumerate(top_indices[:top_n_features]):
        try:
            plt.figure(figsize=(8, 10))

            feature_values = X_train[:, feat_idx]
            shap_val = shap_vals[:, feat_idx]

            valid_mask = (feature_values >= 0) & (feature_values <= 20)
            feature_values_valid = feature_values[valid_mask]
            shap_val_valid = shap_val[valid_mask]

            unique_aa, counts = np.unique(feature_values_valid, return_counts=True)
            aa_counts = dict(zip(unique_aa, counts))

            total_valid = len(feature_values_valid)
            aa_percentages = {aa: (count / total_valid * 100) for aa, count in aa_counts.items()}

            filtered_aa = [aa for aa in unique_aa if aa_counts[aa] >= 10]
            filtered_aa = sorted(filtered_aa, key=lambda x: aa_counts[x], reverse=True)[:3]

            if not filtered_aa:
                print(f" {feature_names[feat_idx]} failed")
                plt.close()
                continue

            x_positions = np.arange(len(filtered_aa))

            aa_to_position = {aa: pos for pos, aa in enumerate(filtered_aa)}

            feature_positions = np.array([aa_to_position[aa] for aa in feature_values_valid
                                          if aa in filtered_aa])
            shap_val_filtered = np.array([shap_val_valid[i] for i, aa in enumerate(feature_values_valid)
                                          if aa in filtered_aa])

            positive_mask = shap_val_filtered > 0
            negative_mask = shap_val_filtered <= 0

            grouped_data = []
            for pos in x_positions:
                mask = (feature_positions == pos)
                grouped_data.append(shap_val_filtered[mask])

            violin_color = '#8FBC8F'
            box_face_color = '#D2B48C'
            box_edge_color = '#8B4513'
            median_color = '#000000'

            violin_parts = plt.violinplot(grouped_data, positions=x_positions,
                                          widths=0.4, showmeans=False, showmedians=False, showextrema=False)

            for pc in violin_parts['bodies']:
                pc.set_facecolor(violin_color)
                pc.set_edgecolor('darkgreen')
                pc.set_linewidth(1)
                pc.set_alpha(0.3)

            box_plot = plt.boxplot(grouped_data, positions=x_positions, widths=0.2,
                                   patch_artist=True, showfliers=False)

            for box in box_plot['boxes']:
                box.set_facecolor(box_face_color)
                box.set_edgecolor(box_edge_color)
                box.set_linewidth(2)
                box.set_alpha(0.6)

            for whisker in box_plot['whiskers']:
                whisker.set_color(box_edge_color)
                whisker.set_linewidth(2)
                whisker.set_alpha(0.8)

            for cap in box_plot['caps']:
                cap.set_color(box_edge_color)
                cap.set_linewidth(2)
                cap.set_alpha(0.8)

            for median in box_plot['medians']:
                median.set_color(median_color)
                median.set_linewidth(2)
                median.set_alpha(0.8)

            jitter_strength = 0.05
            plt.scatter(feature_positions[positive_mask] + np.random.uniform(-jitter_strength, jitter_strength,
                                                                             size=np.sum(positive_mask)),
                        shap_val_filtered[positive_mask],
                        alpha=0.7, color='red', label='Positive SHAP', s=35, edgecolors='darkred', linewidth=0.5)
            plt.scatter(feature_positions[negative_mask] + np.random.uniform(-jitter_strength, jitter_strength,
                                                                             size=np.sum(negative_mask)),
                        shap_val_filtered[negative_mask],
                        alpha=0.7, color='blue', label='Negative SHAP', s=35, edgecolors='darkblue', linewidth=0.5)

            plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5)

            aa_labels = [aa_mapping.get(int(aa), '?') for aa in filtered_aa]
            plt.xticks(x_positions, aa_labels)

            plt.xticks(fontsize=36)
            plt.yticks(fontsize=34)

            y_min, y_max = plt.ylim()
            text_y_position = y_min - 0.045 * (y_max - y_min)

            for pos, aa_val in enumerate(filtered_aa):
                percentage = aa_percentages[aa_val]
                plt.text(pos, text_y_position, f'{percentage:.1f}%',
                         ha='center', va='top', fontsize=32,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            plt.ylim(text_y_position - 0.09 * (y_max - y_min), y_max)

            plt.xlim(-0.5, len(filtered_aa) - 0.5)

            feature_name = feature_names[feat_idx]
            plt.xlabel(f'{feature_name}', fontsize=38)
            plt.ylabel('SHAP value', fontsize=38)
            plt.grid(True, alpha=0.3)

            plt.tight_layout()

            plt.savefig(
                f'shap_dependence_{model_type.lower()}_aa_genotype_{genotype_real_name}_feature_{i}_{feature_name}.png',
                dpi=600, bbox_inches='tight')
            plt.close()


        except Exception as e:
            print(f"failed {i}: {e}")
            import traceback
            traceback.print_exc()

    return model, explainer, shap_values
