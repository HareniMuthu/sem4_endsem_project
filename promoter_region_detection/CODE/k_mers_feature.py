import pandas as pd
from collections import defaultdict

# Function to calculate k-mer frequencies
def calculate_kmer_frequencies(sequence, k=3):
    kmer_counts = defaultdict(int)
    total_kmers = len(sequence) - k + 1
    for i in range(total_kmers):
        kmer = sequence[i:i+k]
        kmer_counts[kmer] += 1
    kmer_freq = {k: v / total_kmers for k, v in kmer_counts.items()}
    return kmer_freq

# Load the promoter dataset
promoters_df = pd.read_csv('/Users/muthusupriya/Documents/Hareni/sem4/BIO/DATASET/promoters (1).csv')

# Define the value of k
k = 3

# Calculate k-mer frequencies for each sequence
kmer_freq_list = []
for sequence in promoters_df['Sequence']:
    kmer_freq_list.append(calculate_kmer_frequencies(sequence, k))

# Convert list of k-mer frequencies to DataFrame
kmer_freq_df = pd.DataFrame(kmer_freq_list).fillna(0)

# Combine k-mer frequencies with the original promoter dataset
promoters_with_kmer_df = pd.concat([promoters_df, kmer_freq_df], axis=1)

# Save the combined dataset
promoters_with_kmer_df.to_csv('/Users/muthusupriya/Documents/Hareni/sem4/BIO/DATASET/promoters_with_kmer_frequencies.csv', index=False)

import ace_tools as tools; tools.display_dataframe_to_user(name="Promoters with K-mer Frequencies", dataframe=promoters_with_kmer_df)
