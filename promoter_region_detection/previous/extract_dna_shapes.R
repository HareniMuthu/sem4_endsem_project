# Load required libraries
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("DNAshapeR")

library(DNAshapeR)

# Load the DNA sequences from CSV
sequences <- read.csv("/Users/muthusupriya/Documents/Hareni/sem4/BIO/promoters (1).csv", stringsAsFactors = FALSE)
sequences <- sequences$Sequence  # Assuming sequences are in a column named 'Sequence'

# Write sequences to a FASTA file
writeXStringSet(DNAStringSet(sequences), "sequences.fasta")

# Predict DNA shapes
shape_predictions <- getShape("sequences.fasta")

# Extract shape features into a data frame
shape_features <- data.frame(
  MGW = unlist(shape_predictions$MGW),
  HelT = unlist(shape_predictions$HelT),
  ProT = unlist(shape_predictions$ProT),
  Roll = unlist(shape_predictions$Roll)
)

# Save the shape features to a CSV file
write.csv(shape_features, "dna_shape_features.csv", row.names = FALSE)
