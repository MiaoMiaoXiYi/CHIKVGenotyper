import pandas as pd
import os
from Bio import SeqIO
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import argparse


def load_feature_indices(file_path):
    """Load feature indices (1-based) and convert to 0-based"""
    with open(file_path, 'r') as f:
        indices = [int(line.strip()) - 1 for line in f.readlines()]
    return indices


def normalize_sequence(seq):
    """Convert 'N' and 'n' to '-' in sequence"""
    return seq.upper().replace('N', '-')


def extract_features(sequence, feature_indices):
    """Extract sequence features from specified indices"""
    return ''.join([sequence[i] for i in feature_indices if i < len(sequence)])


def calculate_coverage(seq):
    """Calculate sequence coverage (proportion of valid nucleotides)"""
    valid_nucleotides = 0
    total_positions = len(seq)

    for nt in seq:
        nt_upper = nt.upper()
        if nt_upper in ['A', 'C', 'G', 'T']:
            valid_nucleotides += 1

    return valid_nucleotides / total_positions if total_positions > 0 else 0


def build_pwm_matrix(genotype_seqs, pseudocount=1, include_gaps=True):
    """Build Position Weight Matrix (PWM) for each genotype"""
    # Calculate background frequency
    all_nucleotides = []
    for seq_list in genotype_seqs.values():
        for seq in seq_list:
            all_nucleotides.extend([nt.upper() for nt in seq if nt.upper() in ['A', 'C', 'G', 'T']])

    background_counter = Counter(all_nucleotides)
    total_nucleotides = sum(background_counter.values())
    background_freq = {nt: count / total_nucleotides for nt, count in background_counter.items()}

    # Build PWM for each genotype
    pwm_matrices = {}

    if include_gaps:
        nucleotide_types = ['A', 'C', 'G', 'T', '-']
    else:
        nucleotide_types = ['A', 'C', 'G', 'T']

    for genotype, seq_list in genotype_seqs.items():
        n_seqs = len(seq_list)
        seq_length = len(seq_list[0])

        # Initialize count matrix
        count_matrix = np.zeros((seq_length, len(nucleotide_types)))

        # Verify sequence length consistency
        for s in seq_list:
            if len(s) != seq_length:
                raise ValueError(f"Sequence length mismatch for genotype {genotype}")

        # Count nucleotide occurrences
        for pos in range(seq_length):
            for seq in seq_list:
                nt = seq[pos].upper()

                if nt == 'N':
                    nt = '-'

                if not include_gaps and nt == '-':
                    continue

                if nt in nucleotide_types:
                    nt_idx = nucleotide_types.index(nt)
                else:
                    if include_gaps:
                        nt_idx = nucleotide_types.index('-')
                    else:
                        continue

                count_matrix[pos, nt_idx] += 1

        # Apply pseudocount and convert to probabilities
        prob_matrix = (count_matrix + pseudocount) / (n_seqs + pseudocount * len(nucleotide_types))

        # Convert to log-odds
        log_odds_matrix = np.zeros((seq_length, len(nucleotide_types)))
        for i, nt in enumerate(nucleotide_types):
            if nt in background_freq:
                log_odds_matrix[:, i] = np.log2(prob_matrix[:, i] / background_freq[nt])
            elif nt == '-':
                log_odds_matrix[:, i] = np.log2(prob_matrix[:, i] / 0.01)
            else:
                avg_freq = np.mean(list(background_freq.values())) if background_freq else 0.25
                log_odds_matrix[:, i] = np.log2(prob_matrix[:, i] / avg_freq)

        pwm_matrices[genotype] = {
            'log_odds': log_odds_matrix,
            'probabilities': prob_matrix,
            'nucleotide_types': nucleotide_types
        }

    return pwm_matrices, background_freq


def classify_sequence_pwm(sequence, pwm_matrices):
    """Classify single sequence using PWM matrices"""
    all_scores = {}

    for genotype, pwm_data in pwm_matrices.items():
        log_odds_matrix = pwm_data['log_odds']
        nucleotide_types = pwm_data['nucleotide_types']

        score = 0
        for pos, nt in enumerate(sequence):
            if pos >= len(log_odds_matrix):
                break

            if nt in nucleotide_types:
                nt_idx = nucleotide_types.index(nt)
            else:
                if '-' in nucleotide_types:
                    nt_idx = nucleotide_types.index('-')
                else:
                    continue

            score += log_odds_matrix[pos, nt_idx]

        all_scores[genotype] = score

    # Find best matching genotype
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    best_genotype = sorted_scores[0][0]

    # Calculate confidence (difference between top two scores)
    if len(sorted_scores) > 1:
        confidence = sorted_scores[0][1] - sorted_scores[1][1]
    else:
        confidence = float('inf')

    return best_genotype, all_scores, confidence


def evaluate_pwm_classification(test_seqs, test_labels, pwm_matrices, confidence_threshold=None):
    """Evaluate PWM classifier performance"""
    correct = 0
    total = len(test_seqs)
    predictions = []
    confidences = []
    uncertain_count = 0

    for i, (seq, true_label) in enumerate(zip(test_seqs, test_labels)):
        pred_label, _, confidence = classify_sequence_pwm(seq, pwm_matrices)

        if confidence_threshold is not None and confidence < confidence_threshold:
            pred_label = "Uncertain"
            uncertain_count += 1

        predictions.append(pred_label)
        confidences.append(confidence)

        if pred_label == true_label:
            correct += 1

    accuracy = correct / (total - uncertain_count) if (total - uncertain_count) > 0 else 0
    return accuracy, predictions, confidences, uncertain_count


def plot_confusion_matrix_with_labels(y_true, y_pred, genotypes, title="Confusion Matrix", normalize=False):
    """Plot confusion matrix with genotype labels"""
    # Filter out uncertain predictions
    filtered_true = []
    filtered_pred = []
    for true, pred in zip(y_true, y_pred):
        if pred != "Uncertain":
            filtered_true.append(true)
            filtered_pred.append(pred)

    # Calculate confusion matrix
    cm = confusion_matrix(filtered_true, filtered_pred, labels=genotypes)

    # Normalize if requested
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        display_matrix = cm_normalized
        fmt = '.2f'
    else:
        display_matrix = cm
        fmt = 'd'

    # Plot confusion matrix
    plt.figure(figsize=(7, 7))
    plt.imshow(display_matrix, interpolation='nearest', cmap="YlGnBu")
    plt.colorbar(shrink=0.7)

    # Add labels
    tick_marks = np.arange(len(genotypes))
    plt.xticks(tick_marks, genotypes, rotation=45, fontsize=10)
    plt.yticks(tick_marks, genotypes, fontsize=10)

    # Add value labels
    thresh = display_matrix.max() / 2.
    for i in range(len(genotypes)):
        for j in range(len(genotypes)):
            if normalize:
                text_value = format(display_matrix[i, j] * 100, '.1f') + '%'
            else:
                text_value = format(display_matrix[i, j], 'd')

            plt.text(j, i, text_value,
                     horizontalalignment="center",
                     color="white" if display_matrix[i, j] > thresh else "black",
                     fontsize=10)

    plt.tight_layout()
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # Save plot
    if normalize:
        filename = 'confusion_matrix_normalized.png'
    else:
        filename = 'confusion_matrix_genotypes.png'

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    return display_matrix


# Main program
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CHIKV Genotype Identification')
    parser.add_argument('--include_gaps', action='store_true',
                        help='Include gap characters in PWM')
    parser.add_argument('--confidence_threshold', type=float, default=5.0,
                        help='Confidence threshold for uncertain predictions')
    args = parser.parse_args()

    args.include_gaps = False

    # Load feature indices
    FEATURE_INDICES_FILE = 'nn_indices.txt'
    print("Loading feature indices...")
    feature_indices = load_feature_indices(FEATURE_INDICES_FILE)
    print(f"Loaded {len(feature_indices)} feature indices")

    # Process NCBI data
    target_seq = './input.fasta'
    ids = []
    sequences = []
    genotypes = []

    fasta_inputs = SeqIO.parse(target_seq, "fasta")
    low_cov_num = 0

    for seq in fasta_inputs:
        normalized_seq = normalize_sequence(str(seq.seq))
        coverage = calculate_coverage(normalized_seq)

        if coverage < 0.90:
            print('low coverage seq: ', seq.id, coverage)
            low_cov_num += 1
            continue

        parts = seq.id.split('_')
        if len(parts) < 2:
            print(f"Warning: Skipping ID '{seq.id}' - invalid format")
            continue

        genotype = '_'.join(parts[1:])

        feature_seq = extract_features(normalized_seq, feature_indices)

        ids.append(seq.id)
        sequences.append(feature_seq)
        genotypes.append(genotype)

    print(f'Valid sequences: {len(sequences)}')
    print(f'Low coverage sequences: {low_cov_num}')
    print(f'Feature sequence length: {len(sequences[0]) if sequences else 0}')

    # Group sequences by genotype
    genotype_seqs = defaultdict(list)
    for seq, genotype in zip(sequences, genotypes):
        genotype_seqs[genotype].append(seq)

    # Build PWM matrices
    print("Building PWM matrices...")
    pwm_matrices, background_freq = build_pwm_matrix(
        genotype_seqs,
        pseudocount=1,
        include_gaps=args.include_gaps
    )

    # Save PWM matrices
    gap_status = "with_gaps" if args.include_gaps else "no_gaps"
    pwm_output_file = f'./pwm_matrices_{len(feature_indices)}feat_{gap_status}.pkl'
    with open(pwm_output_file, 'wb') as f:
        pickle.dump({
            'pwm_matrices': pwm_matrices,
            'background_freq': background_freq,
            'genotypes': list(genotype_seqs.keys()),
            'feature_indices': feature_indices,
            'include_gaps': args.include_gaps
        }, f)

    print(f"PWM matrices saved to {pwm_output_file}")

    # Self-validation on NCBI data
    print("Performing self-validation...")
    accuracy, predictions, confidences, uncertain_count = evaluate_pwm_classification(
        sequences, genotypes, pwm_matrices, confidence_threshold=args.confidence_threshold
    )
    print(f"Self-validation accuracy: {accuracy:.4f}")
    print(f"Uncertain predictions: {uncertain_count}")

    # Analyze confusion matrix
    genotype_labels = list(genotype_seqs.keys())
    cm = plot_confusion_matrix_with_labels(genotypes, predictions, genotype_labels,
                                           title=f"Confusion Matrix (Accuracy: {accuracy:.4f})")

    # Calculate per-genotype accuracy
    genotype_accuracies = {}
    for genotype in genotype_labels:
        correct = 0
        total = 0
        for true, pred in zip(genotypes, predictions):
            if true == genotype and pred != "Uncertain":
                total += 1
                if pred == genotype:
                    correct += 1
        if total > 0:
            genotype_accuracies[genotype] = correct / total
        else:
            genotype_accuracies[genotype] = 0

    print("\nPer-genotype accuracy:")
    for genotype, acc in genotype_accuracies.items():
        print(f"{genotype}: {acc:.4f}")

    def test_pwm_on_fasta(test_fasta_path, pwm_matrices, feature_indices, confidence_threshold=None):
        """Classify sequences in FASTA file"""
        test_ids = []
        test_sequences = []

        for seq_record in SeqIO.parse(test_fasta_path, "fasta"):
            test_ids.append(seq_record.id)
            normalized_seq = normalize_sequence(str(seq_record.seq))
            feature_seq = extract_features(normalized_seq, feature_indices)
            test_sequences.append(feature_seq)

        results = []
        uncertain_count = 0

        for seq_id, seq in zip(test_ids, test_sequences):
            pred_genotype, all_scores, confidence = classify_sequence_pwm(seq, pwm_matrices)

            if confidence_threshold is not None and confidence < confidence_threshold:
                pred_genotype = "Uncertain"
                uncertain_count += 1

            results.append({
                'id': seq_id,
                'predicted_genotype': pred_genotype,
                'confidence': confidence,
                'scores': all_scores
            })

        print(f"Uncertain predictions: {uncertain_count}")
        return results

    def load_pwm_and_classify(pwm_file_path, test_fasta_path, confidence_threshold=None):
        """Load PWM matrices and classify sequences in FASTA file"""
        with open(pwm_file_path, 'rb') as f:
            pwm_data = pickle.load(f)

        pwm_matrices = pwm_data['pwm_matrices']
        feature_indices = pwm_data['feature_indices']

        return test_pwm_on_fasta(test_fasta_path, pwm_matrices, feature_indices, confidence_threshold)

    # Example: Classify test FASTA file
    test_fasta_path = "./test_sequences.fasta"
    if os.path.exists(test_fasta_path):
        print(f"Classifying {test_fasta_path}...")
        classification_results = test_pwm_on_fasta(
            test_fasta_path, pwm_matrices, feature_indices,
            confidence_threshold=args.confidence_threshold
        )

        results_df = pd.DataFrame(classification_results)
        results_output_path = f"./pwm_classification_results_{len(feature_indices)}feat_{gap_status}.csv"
        results_df.to_csv(results_output_path, index=False)
        print(f"Classification results saved to {results_output_path}")