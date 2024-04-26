####################################################
# install env
####################################################
conda create -n txt2onto2 python==3.9.16
conda activate txt2onto2
pip install -r requirements.txt

####################################################
# disease model training
####################################################
# preprocess input text
python ../src/preprocess.py \
-input ../data/disease_desc.tsv \
-out ../data/disease_desc_processed.tsv

# generate embedding table for training data (required GPU for fast embedding generation)
python ../src/embedding_lookup_table.py \
-corpus ../data/disease_desc_processed.tsv \
-out ../data/disease_desc_embedding.npz

# prepare train inputs
python ../src/input.py \
-onto MONDO:0004981 \
-gs ../data/disease_labels.csv.gz \
-text ../data/disease_desc_processed.tsv \
-id ../data/disease_ID.tsv \
-out ../data

# train LR + word model
python ../src/train.py \
-input ../data/MONDO_0004981__train_input.tsv \
-out ../results

####################################################
# disease annotation
####################################################
# preprocess given text data from clinicaltrials
python ../src/preprocess.py \
-input ../data/clinicaltrials_desc.tsv \
-out ../data/clinicaltrials_desc_processed.tsv

# generate embedding table for given text data from clinicaltrials (required GPU for fast embedding generation)
python ../src/embedding_lookup_table.py \
-corpus ../data/clinicaltrials_desc_processed.tsv \
-out ../data/clinicaltrials_desc_embedding.npz

# predict disease labels for studies from clinicaltrials
python ../src/predict.py \
-input ../data/clinicaltrials_desc_processed.tsv \
-id ../data/clinicaltrials_ID.tsv \
-input_embed ../data/clinicaltrials_desc_embedding.npz \
-train_embed ../data/disease_desc_embedding.npz \
-model ../results/MONDO_0004981__model.pkl \
-out ../results


####################################################
# tissue model training
####################################################
# preprocess input text
python ../src/preprocess.py \
-input ../data/tissue_desc.tsv \
-out ../data/tissue_desc_processed.tsv

# generate embedding table for training data (required GPU for fast embedding generation)
python ../src/embedding_lookup_table.py \
-corpus ../data/tissue_desc_processed.tsv \
-out ../data/tissue_desc_embedding.npz

# prepare train inputs
python ../src/input.py \
-onto UBERON:0019319 \
-gs ../data/tissue_labels.csv.gz \
-text ../data/tissue_desc_processed.tsv \
-id ../data/tissue_ID.tsv \
-out ../data

# train LR + word model
python ../src/train.py \
-input ../data/UBERON_0019319__train_input.tsv \
-out ../results