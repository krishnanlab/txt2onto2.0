# This demo shows how to predict labels using provided model.
# Here we predict labels using metadata from clinicaltrials as input, and predict studies related to atrial fibrillation (MONDO:0004981)

# preprocess given text data from clinicaltrials
python preprocess.py \
-input ../demo/clinicaltrials_desc.tsv \
-out ../results/clinicaltrials_desc_processed.tsv

# generate embedding table for given text data from clinicaltrials (required GPU for fast embedding generation)
python embedding_lookup_table.py \
-input ../results/clinicaltrials_desc_processed.tsv \
-out ../results/clinicaltrials_desc_embedding.npz

# predict disease labels for studies from clinicaltrials
python predict.py \
-input ../results/clinicaltrials_desc_processed.tsv \
-id ../demo/clinicaltrials_ID.tsv \
-input_embed ../results/clinicaltrials_desc_embedding.npz \
-train_embed ../data/disease_desc_embedding.npz \
-model ../bins/MONDO_0004981__model.pkl \
-out ../results
