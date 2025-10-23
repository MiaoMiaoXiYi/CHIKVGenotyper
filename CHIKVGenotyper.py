import sys
import pandas as pd
import numpy as np
from Bio import SeqIO
import joblib
import os
from collections import defaultdict

# Parameters
SITE_START = 1
SITE_END = 11811
COVERAGE_THRESHOLD = 0.8
FEATURE_INDICES_FILE = './features_nn_1197.txt'
INPUT_FASTA_FILE = './input_data.fasta'

# Model files (adjust paths as needed)
MODEL_FILES = {
    'Decision Tree': 'decision_tree_model.pkl',
    'Random Forest': 'random_forest_model.pkl',
    'XGBoost': 'xgboost_model.pkl',
    'LightGBM': 'lightgbm_model.pkl'
}

fasta_inputs_ref = SeqIO.parse('./reference.fasta', "fasta")
ref_seq = []
for seq in fasta_inputs_ref:
    ref_seq = str(seq.seq)

ref_len = len(ref_seq)


def calculate_coverage(seq):
    """Calculate sequence coverage (proportion of valid nucleotides)"""
    valid_nucleotides = 0
    total_positions = len(seq)

    for nt in seq:
        nt_upper = nt.upper()
        if nt_upper in ['A', 'C', 'G', 'T']:
            valid_nucleotides += 1

    return valid_nucleotides / total_positions if total_positions > 0 else 0


def extract_sequence_region(seq, start, end):
    """Extract specified region from sequence"""
    if start is None:
        start = 1
    if end is None:
        end = len(seq)

    start_idx = start - 1
    end_idx = end

    return seq[start_idx:end_idx]


def load_feature_indices(file_path):
    """Load feature indices"""
    with open(file_path, 'r') as f:
        indices = [int(line.strip()) for line in f.readlines()]
    return indices


def encode_sequence(seq, feature_indices):
    """Encode sequence by extracting only feature positions"""
    encoded = []
    for idx in feature_indices:
        pos = idx - 1
        nt = seq[pos].upper()
        nt_ref = ref_seq[pos].upper()
        if nt not in ['A', 'C', 'G', 'T']:
            nt = nt_ref
        if nt == 'A':
            encoded.append(0)
        elif nt == 'C':
            encoded.append(1)
        elif nt == 'G':
            encoded.append(2)
        elif nt == 'T':
            encoded.append(3)
        else:
            print('error1', nt, nt_ref)
            sys.exit()

    return encoded


def load_models_and_mappings():
    """Load pre-trained models and their genotype mappings"""
    models = {}
    genotype_mappings = {}

    for model_name, model_file in MODEL_FILES.items():
        if os.path.exists(model_file):
            try:
                # Load model (assuming it contains both model and mapping)
                model_data = joblib.load(model_file)

                # Handle different model storage formats
                if isinstance(model_data, dict) and 'model' in model_data:
                    # Model stored as dictionary with model and mapping
                    models[model_name] = model_data['model']
                    if 'index_to_genotype' in model_data:
                        genotype_mappings[model_name] = model_data['index_to_genotype']
                    elif 'genotype_mapping' in model_data:
                        genotype_mappings[model_name] = model_data['genotype_mapping']
                else:
                    # Model stored directly
                    models[model_name] = model_data
                    print(f"Warning: No genotype mapping found for {model_name}")

                print(f"Loaded {model_name} from {model_file}")

            except Exception as e:
                print(f"Error loading {model_name}: {e}")
        else:
            print(f"Warning: Model file {model_file} not found")

    return models, genotype_mappings


def prepare_prediction_data(fasta_file, feature_indices):
    """Prepare prediction data from FASTA file"""
    prediction_data = []
    valid_sequences = 0
    low_coverage_count = 0

    fasta_inputs = SeqIO.parse(fasta_file, "fasta")
    for seq in fasta_inputs:
        seq_id = seq.id
        seq_str = str(seq.seq)

        # Extract specified region
        if SITE_START is not None or SITE_END is not None:
            seq_str = extract_sequence_region(seq_str, SITE_START, SITE_END)

        # Calculate coverage
        coverage = calculate_coverage(seq_str)

        if coverage >= COVERAGE_THRESHOLD:
            # Encode sequence
            encoded_seq = encode_sequence(seq_str, feature_indices)
            prediction_data.append({
                'id': seq_id,
                'sequence': encoded_seq,
                'coverage': coverage
            })
            valid_sequences += 1
        else:
            low_coverage_count += 1

    print(f"Total sequences processed: {len(list(SeqIO.parse(fasta_file, 'fasta')))}")
    print(f"Valid sequences (coverage >= {COVERAGE_THRESHOLD}): {valid_sequences}")
    print(f"Low coverage sequences filtered: {low_coverage_count}")

    return prediction_data


def predict_sequences(models, genotype_mappings, prediction_data):
    """Predict genotypes for sequences using all loaded models"""
    results = defaultdict(list)

    for model_name, model in models.items():
        print(f"Making predictions with {model_name}...")

        # Prepare data for prediction
        X_pred = np.array([item['sequence'] for item in prediction_data])

        # Make predictions
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_pred)
            predictions = model.predict(X_pred)
        else:
            probabilities = None
            predictions = model.predict(X_pred)

        # Get genotype mapping for this model
        if model_name in genotype_mappings:
            index_to_genotype = genotype_mappings[model_name]
        else:
            # If no mapping available, use generic labels
            n_classes = probabilities.shape[1] if probabilities is not None else len(np.unique(predictions))
            index_to_genotype = {i: f"Class_{i}" for i in range(n_classes)}

        # Store results
        for i, item in enumerate(prediction_data):
            pred_class = predictions[i]
            genotype = index_to_genotype.get(pred_class, f"Unknown_{pred_class}")

            result = {
                'sequence_id': item['id'],
                'coverage': item['coverage'],
                'predicted_genotype': genotype
            }

            # Add probabilities if available
            if probabilities is not None:
                for class_idx, class_name in index_to_genotype.items():
                    if class_idx < probabilities.shape[1]:
                        result[f'prob_{class_name}'] = probabilities[i, class_idx]

            results[model_name].append(result)

    return results


def save_prediction_results(results, output_dir="prediction_results"):
    """Save prediction results to CSV files"""
    os.makedirs(output_dir, exist_ok=True)

    # Save individual model results
    for model_name, predictions in results.items():
        df = pd.DataFrame(predictions)
        output_file = os.path.join(output_dir, f"{model_name.replace(' ', '_').lower()}_predictions.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved {model_name} predictions to {output_file}")

    # Create combined results file
    combined_results = []
    for model_name, predictions in results.items():
        for pred in predictions:
            combined_result = {
                'sequence_id': pred['sequence_id'],
                'coverage': pred['coverage'],
                'model': model_name,
                'predicted_genotype': pred['predicted_genotype']
            }

            # Add probabilities
            prob_cols = {k: v for k, v in pred.items() if k.startswith('prob_')}
            combined_result.update(prob_cols)

            combined_results.append(combined_result)

    if combined_results:
        combined_df = pd.DataFrame(combined_results)
        combined_file = os.path.join(output_dir, "all_models_predictions_combined.csv")
        combined_df.to_csv(combined_file, index=False)
        print(f"Saved combined predictions to {combined_file}")

    # Create summary file (one row per sequence with predictions from all models)
    summary_data = []
    sequence_ids = list(set([pred['sequence_id'] for predictions in results.values() for pred in predictions]))

    for seq_id in sequence_ids:
        summary_row = {'sequence_id': seq_id}

        # Find coverage (should be same across models)
        coverage = None
        for predictions in results.values():
            for pred in predictions:
                if pred['sequence_id'] == seq_id:
                    coverage = pred['coverage']
                    break
            if coverage is not None:
                break
        summary_row['coverage'] = coverage

        # Add predictions from each model
        for model_name in results.keys():
            model_pred = next((pred for pred in results[model_name] if pred['sequence_id'] == seq_id), None)
            if model_pred:
                summary_row[f'{model_name}_prediction'] = model_pred['predicted_genotype']
                # Add highest probability
                prob_cols = {k: v for k, v in model_pred.items() if k.startswith('prob_')}
                if prob_cols:
                    max_prob_genotype = max(prob_cols, key=prob_cols.get).replace('prob_', '')
                    summary_row[f'{model_name}_max_prob'] = max_prob_genotype
                    summary_row[f'{model_name}_confidence'] = max(prob_cols.values())

        summary_data.append(summary_row)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, "prediction_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved prediction summary to {summary_file}")


def main():
    """Main function for prediction only"""
    # Load feature indices
    print("Loading feature indices...")
    feature_indices = load_feature_indices(FEATURE_INDICES_FILE)
    print(f"Loaded {len(feature_indices)} feature indices")

    # Load pre-trained models
    print("Loading pre-trained models...")
    models, genotype_mappings = load_models_and_mappings()

    if not models:
        print("Error: No models loaded. Please check model file paths.")
        return

    print(f"Successfully loaded {len(models)} models")

    # Prepare prediction data
    print("Preparing prediction data...")
    prediction_data = prepare_prediction_data(INPUT_FASTA_FILE, feature_indices)

    if not prediction_data:
        print("Error: No valid sequences found for prediction.")
        return

    # Make predictions
    print("Making predictions...")
    results = predict_sequences(models, genotype_mappings, prediction_data)

    # Save results
    print("Saving prediction results...")
    save_prediction_results(results)

    print("Prediction completed successfully!")


if __name__ == "__main__":
    main()