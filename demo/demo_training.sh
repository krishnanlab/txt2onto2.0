# Here, we replicate the process of disease model training in the paper to demonstrate process of training a new model, using atrial fibrillation (MONDO:0004981) as example

# preprocess input text
python preprocess.py \
-input ../demo/disease_desc.tsv \
-out ../results/disease_desc_processed.tsv

# generate embedding table for training data (required GPU for fast embedding generation)
python embedding_lookup_table.py \
-input ../results/disease_desc_processed.tsv \
-out ../results/disease_demo_desc_embedding.npz \
-batch_size 2000

# prepare train inputs
python input.py \
-onto MONDO:0004981 \
-gs ../demo/disease_labels.csv.gz \
-text ../results/disease_desc_processed.tsv \
-id ../demo/disease_ID.tsv \
-out ../results

# train LR + word model
python train.py \
-input ../results/MONDO_0004981__train_input.tsv \
-out ../results

# We also replicate the process of tissue model training in the paper to demonstrate process of training a new model, using bone marrow cell (CL:0002092) as example

# preprocess input text
python preprocess.py \
-input ../demo/tissue_desc.tsv \
-out ../results/tissue_desc_processed.tsv

# generate embedding table for training data (required GPU for fast embedding generation)
python embedding_lookup_table.py \
-input ../results/tissue_desc_processed.tsv \
-out ../results/tissue_demo_desc_embedding.npz \
-batch_size 2000

# prepare train inputs
python input.py \
-onto CL:0000082 \
-gs ../demo/tissue_labels.csv.gz \
-text ../results/tissue_desc_processed.tsv \
-id ../demo/tissue_ID.tsv \
-out ../results

# train LR + word model
python train.py \
-input ../results/CL_0000082__train_input.tsv \
-out ../results