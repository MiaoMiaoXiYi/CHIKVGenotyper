import pandas as pd
from Bio import SeqIO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import random

# Parameters
SITE_START = 77  # Start position (1-based)
SITE_END = 11313  # End position (1-based), None for full sequence
COVERAGE_THRESHOLD = 0.9  # Sequence coverage threshold (0-1)
MAX_SAMPLES_PER_GENOTYPE = 200  # Maximum samples per genotype (downsampling threshold)


def calculate_similarity(seq1, seq2):
    """Calculate similarity between two sequences (considering comparable sites)"""
    matches = 0
    total_comparable = 0

    for n1, n2 in zip(seq1, seq2):
        n1_upper = n1.upper()
        n2_upper = n2.upper()

        # Count as comparable if both are not gaps
        if n1_upper != '-' and n2_upper != '-':
            total_comparable += 1
            if n1_upper == n2_upper:
                matches += 1

    return matches / total_comparable if total_comparable > 0 else 0


def calculate_coverage(seq):
    """Calculate sequence coverage (proportion of valid nucleotides)"""
    valid_nucleotides = 0
    total_positions = len(seq)

    for nt in seq:
        nt_upper = nt.upper()
        # Valid nucleotides: A, C, G, T (excluding gaps '-' and unknown 'N')
        if nt_upper in ['A', 'C', 'G', 'T']:
            valid_nucleotides += 1

    return valid_nucleotides / total_positions if total_positions > 0 else 0


def extract_sequence_region(seq, start, end):
    """Extract specified region from sequence"""
    if start is None:
        start = 1  # 1-based indexing
    if end is None:
        end = len(seq)

    # Convert to 0-based indexing
    start_idx = start - 1
    end_idx = end

    return seq[start_idx:end_idx]


def main():
    # File paths
    fasta_file = './data/representative_samples.fasta'
    output_matrix_file = './data/genotype_difference_matrix_revised.txt'
    output_heatmap_file = './data/genotype_difference_heatmap_revised.png'

    # Read and process FASTA file
    print("Reading FASTA file and processing sequences...")
    id_to_seq = {}
    id_to_genotype = {}
    filtered_count = 0

    fasta_inputs = SeqIO.parse(fasta_file, "fasta")
    for seq in fasta_inputs:
        seq_id = seq.id
        seq_str = str(seq.seq)

        # Extract genotype from sequence ID
        if '_' in seq_id:
            parts = seq_id.split('_')
            genotype = parts[-1]  # Use last part as genotype
        else:
            print(f"Warning: Sequence ID {seq_id} has unexpected format, skipping")
            continue

        # Extract specified region
        if SITE_START is not None or SITE_END is not None:
            seq_str = extract_sequence_region(seq_str, SITE_START, SITE_END)

        # Calculate coverage
        coverage = calculate_coverage(seq_str)

        # Filter low coverage sequences
        if coverage < COVERAGE_THRESHOLD:
            filtered_count += 1
            continue

        id_to_seq[seq_id] = seq_str
        id_to_genotype[seq_id] = genotype

    print(f"Original samples: {len(list(SeqIO.parse(fasta_file, 'fasta')))}")
    print(f"Filtered {filtered_count} low coverage samples")
    print(f"Remaining samples: {len(id_to_seq)}")

    # Group samples by genotype
    print("Grouping samples by genotype...")
    genotype_groups = {}
    for seq_id, genotype in id_to_genotype.items():
        if genotype not in genotype_groups:
            genotype_groups[genotype] = []

        if seq_id in id_to_seq:
            genotype_groups[genotype].append(id_to_seq[seq_id])
        else:
            print(f"Warning: Sequence not found for sample {seq_id}")

    # Downsample genotypes with too many samples
    print("Downsampling genotypes with excessive samples...")
    downsampled_counts = {}
    for genotype, sequences in genotype_groups.items():
        if len(sequences) > MAX_SAMPLES_PER_GENOTYPE:
            downsampled_counts[genotype] = MAX_SAMPLES_PER_GENOTYPE
            genotype_groups[genotype] = random.sample(sequences, MAX_SAMPLES_PER_GENOTYPE)
        else:
            downsampled_counts[genotype] = len(sequences)

    # Get all genotypes
    genotypes = sorted(genotype_groups.keys())
    n_genotypes = len(genotypes)

    # Print sample counts per genotype
    print("\nSample counts per genotype (after downsampling):")
    for genotype in genotypes:
        print(f"{genotype}: {downsampled_counts[genotype]} samples")

    # Initialize difference matrix
    difference_matrix = np.zeros((n_genotypes, n_genotypes))

    # Calculate inter-genotype differences
    print("Calculating inter-genotype differences...")
    for i, genotype1 in enumerate(genotypes):
        seqs1 = genotype_groups[genotype1]
        n1 = len(seqs1)

        for j, genotype2 in enumerate(genotypes):
            if i == j:
                difference_matrix[i, j] = 0  # Same genotype difference is 0
                continue

            seqs2 = genotype_groups[genotype2]
            n2 = len(seqs2)

            print('diff calculating: ', genotype1, genotype2)
            # Calculate average difference between all sample pairs
            total_difference = 0
            total_pairs = 0

            for seq1 in seqs1:
                for seq2 in seqs2:
                    similarity = calculate_similarity(seq1, seq2)
                    difference = 1 - similarity
                    total_difference += difference
                    total_pairs += 1

            if total_pairs > 0:
                avg_difference = total_difference / total_pairs
                difference_matrix[i, j] = avg_difference
            else:
                difference_matrix[i, j] = 0

    # Save difference matrix to text file
    print("Saving difference matrix to text file...")
    with open(output_matrix_file, 'w') as f:
        # Write parameter information
        f.write(f"# Parameters:\n")
        f.write(f"# Site range: {SITE_START}-{SITE_END}\n")
        f.write(f"# Coverage threshold: {COVERAGE_THRESHOLD}\n")

        # Write header
        f.write("Genotype\t" + "\t".join(genotypes) + "\n")

        # Write matrix data (5 decimal places)
        for i, genotype in enumerate(genotypes):
            row_data = [f"{difference_matrix[i, j]:.5f}" for j in range(n_genotypes)]
            f.write(f"{genotype}\t" + "\t".join(row_data) + "\n")

    # Plot heatmap
    print("Plotting heatmap...")
    plot_heatmap(difference_matrix, genotypes, output_heatmap_file,
                 SITE_START, SITE_END, COVERAGE_THRESHOLD)

    print(f"Difference matrix saved to {output_matrix_file}")
    print(f"Heatmap saved to {output_heatmap_file}")


def plot_heatmap(matrix, genotypes, output_file, site_start=None, site_end=None, coverage_threshold=None):
    """Plot difference matrix heatmap using fixed genotype order"""
    # Ensure matrix is symmetric
    symmetric_matrix = (matrix + matrix.T) / 2

    # Set diagonal to 0 (same genotype difference is 0)
    np.fill_diagonal(symmetric_matrix, 0)

    # Define fixed genotype order
    fixed_order = ['AAL', 'MAL', 'EAL', 'IOL', 'SAL', 'AUL', 'AUL-Am', 'WA']

    # Get genotypes from current matrix
    current_genotypes = list(genotypes)

    # Create mapping from genotype to index in fixed order
    genotype_to_index = {genotype: i for i, genotype in enumerate(fixed_order)}

    # Reorder matrix according to fixed order
    n_fixed = len(fixed_order)
    reordered_matrix = np.zeros((n_fixed, n_fixed))

    # Fill reordered matrix
    for i, genotype_i in enumerate(fixed_order):
        if genotype_i not in current_genotypes:
            continue
        orig_i = current_genotypes.index(genotype_i)
        for j, genotype_j in enumerate(fixed_order):
            if genotype_j not in current_genotypes:
                continue
            orig_j = current_genotypes.index(genotype_j)
            reordered_matrix[i, j] = symmetric_matrix[orig_i, orig_j]

    plt.figure(figsize=(6, 6))

    # Create mask to hide upper triangle (including diagonal)
    mask = np.triu(np.ones_like(reordered_matrix, dtype=bool))

    # Determine color map range
    vmin = 0
    vmax = np.max(reordered_matrix) * 1.05  # Add 5% margin

    # Create heatmap
    ax = sns.heatmap(
        reordered_matrix,
        mask=mask,  # Hide upper triangle
        xticklabels=fixed_order,
        yticklabels=fixed_order,
        annot=True,
        fmt=".3f",  # 3 decimal places
        cmap="YlGnBu",
        square=True,
        cbar_kws={'shrink': 0.7},
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor='lightgray',
        annot_kws={"size": 9}
    )

    # Adjust label font size
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

    return fixed_order, reordered_matrix


def plot_heatmap_from_file(matrix_file, output_heatmap_file):
    """Read difference matrix from text file and plot heatmap"""
    print(f"Reading difference matrix from {matrix_file}...")

    # Read difference matrix
    with open(matrix_file, 'r') as f:
        lines = f.readlines()

    # Extract parameter information
    params = {}
    genotype_line = 0
    for i, line in enumerate(lines):
        if line.startswith('#'):
            if ':' in line:
                key, value = line[2:].strip().split(':', 1)
                params[key.strip()] = value.strip()
        else:
            genotype_line = i
            break

    # Read genotype names and matrix data
    genotypes = lines[genotype_line].strip().split('\t')[1:]
    n_genotypes = len(genotypes)

    difference_matrix = np.zeros((n_genotypes, n_genotypes))

    for i, line in enumerate(lines[genotype_line + 1:genotype_line + 1 + n_genotypes]):
        parts = line.strip().split('\t')
        for j in range(1, n_genotypes + 1):
            difference_matrix[i, j - 1] = float(parts[j])

    # Extract parameters
    site_range = params.get('Site range', '')
    coverage_threshold = params.get('Coverage threshold', '')

    # Parse site_range
    site_start, site_end = None, None
    if site_range and '-' in site_range:
        try:
            site_start, site_end = map(int, site_range.split('-'))
        except ValueError:
            pass

    # Parse coverage_threshold
    try:
        coverage_threshold_val = float(coverage_threshold)
    except (ValueError, TypeError):
        coverage_threshold_val = None

    # Plot heatmap
    ordered_genotypes, ordered_matrix = plot_heatmap(
        difference_matrix, genotypes, output_heatmap_file,
        site_start, site_end, coverage_threshold_val
    )

    # Save ordered matrix (optional)
    ordered_matrix_file = './data/ordered_genotype_difference_matrix.txt'
    with open(ordered_matrix_file, 'w') as f:
        # Write parameter information
        f.write(f"# Parameters:\n")
        f.write(f"# Site range: {site_start}-{site_end}\n")
        f.write(f"# Coverage threshold: {coverage_threshold}\n")
        f.write(f"# Matrix ordered by hierarchical clustering\n")

        # Write header
        f.write("Genotype\t" + "\t".join(ordered_genotypes) + "\n")

        # Write matrix data (5 decimal places)
        for i, genotype in enumerate(ordered_genotypes):
            row_data = [f"{ordered_matrix[i, j]:.5f}" for j in range(len(ordered_genotypes))]
            f.write(f"{genotype}\t" + "\t".join(row_data) + "\n")

    print(f"Ordered difference matrix saved to {ordered_matrix_file}")


if __name__ == "__main__":
    # If difference matrix file exists, plot heatmap directly
    matrix_file = './data/genotype_difference_matrix.txt'
    output_heatmap_file = './data/genotype_difference_heatmap.png'

    if os.path.exists(matrix_file):
        print("Existing difference matrix file detected, plotting heatmap...")
        plot_heatmap_from_file(matrix_file, output_heatmap_file)
    else:
        print("Difference matrix file not found, starting calculation...")
        main()