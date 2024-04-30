# install env
conda create -n txt2onto2 python==3.9.16
conda activate txt2onto2
pip install -r requirements.txt

# prepare files
# get labels
cp /mnt/ufs18/rs-032/FishEvoDevoGeno/Hao/Cross_platform_text_based_classification/data/processed/non_redundant_true_labels/true_label__inst_type=sample__task=tissue__jac_thresh=0.9.csv.gz ../data/tissue_labels.csv.gz
cp /mnt/ufs18/rs-032/FishEvoDevoGeno/Hao/Cross_platform_text_based_classification/data/processed/non_redundant_true_labels/true_label__inst_type=study__task=disease__jac_thresh=0.5.csv.gz ../data/disease_labels.csv.gz

# get raw desc
# tissue
python ../src/generate_desc.py \
-anno /mnt/ufs18/rs-032/FishEvoDevoGeno/Hao/Cross_platform_text_based_classification/data/processed/disease_tissue_annotations/2023-09-04__tissue_sample_annotations.tsv \
-meta /mnt/ufs18/rs-032/FishEvoDevoGeno/Hao/Cross_platform_text_based_classification/data/metadata/2023-09-02__metadata.pkl.gz \
-outdir ../data/ \
-inst_type sample \
-level l2 \
-fields /mnt/ufs18/rs-032/FishEvoDevoGeno/Hao/Cross_platform_text_based_classification/data/metadata/l2_fields.js
mv ../data/desc.tsv ../data/tissue_desc.tsv
mv ../data/ID.tsv ../data/tissue_ID.tsv

# disease
python ../src/generate_desc.py \
-anno /mnt/ufs18/rs-032/FishEvoDevoGeno/Hao/Cross_platform_text_based_classification/data/processed/disease_tissue_annotations/2023-09-04__disease_study_annotations.tsv \
-meta /mnt/ufs18/rs-032/FishEvoDevoGeno/Hao/Cross_platform_text_based_classification/data/metadata/2023-09-02__metadata.pkl.gz \
-outdir ../data/ \
-inst_type study \
-level l2 \
-fields /mnt/ufs18/rs-032/FishEvoDevoGeno/Hao/Cross_platform_text_based_classification/data/metadata/l2_fields.js
mv ../data/desc.tsv ../data/disease_desc.tsv
mv ../data/ID.tsv ../data/disease_ID.tsv

# get text from clinicaltrials for predictions
python ../src/preprocess_clinicaltrials_metadata.py \
-infile /mnt/ufs18/rs-032/FishEvoDevoGeno/Hao/Cross_platform_text_based_classification/data/testing_data/clinicaltrials/2023-12-06_ctg-studies.csv \
-outdir ../data/ \
-select_n 5000 \
-ent_type word
mv ../data/desc.tsv ../data/clinicaltrials_desc.tsv
mv ../data/ID.tsv ../data/clinicaltrials_ID.tsv

# preprocess input text
python ../src/preprocess.py \
-input ../data/disease_desc.tsv \
-out ../data/disease_desc_processed.tsv

python ../src/preprocess.py \
-input ../data/tissue_desc.tsv \
-out ../data/tissue_desc_processed.tsv

python ../src/preprocess.py \
-input ../data/clinicaltrials_desc.tsv \
-out ../data/clinicaltrials_desc_processed.tsv

# generate embedding table
python ../src/embedding_lookup_table.py \
-corpus ../data/disease_desc_processed.tsv \
-out ../data/disease_desc_embedding.npz

python ../src/embedding_lookup_table.py \
-corpus ../data/tissue_desc_processed.tsv \
-out ../data/tissue_desc_embedding.npz

python ../src/embedding_lookup_table.py \
-corpus ../data/clinicaltrials_desc_processed.tsv \
-out ../data/clinicaltrials_desc_embedding.npz

# prepare train inputs
python ../src/input.py \
-onto MONDO:0004981 \
-gs ../data/disease_labels.csv.gz \
-text ../data/disease_desc_processed.tsv \
-id ../data/disease_ID.tsv \
-out ../data

python ../src/input.py \
-onto UBERON:0019319 \
-gs ../data/tissue_labels.csv.gz \
-text ../data/tissue_desc_processed.tsv \
-id ../data/tissue_ID.tsv \
-out ../data

# train model
python ../src/train.py \
-input ../data/MONDO_0004981__train_input.tsv \
-out ../results

python ../src/train.py \
-input ../data/UBERON_0019319__train_input.tsv \
-out ../results

# predict
python ../src/predict.py \
-input ../data/clinicaltrials_desc_processed.tsv \
-id ../data/clinicaltrials_ID.tsv \
-input_embed ../data/clinicaltrials_desc_embedding.npz \
-train_embed ../data/disease_desc_embedding.npz \
-model ../results/MONDO_0004981__model.pkl \
-out ../results

# reformat models
mkdir ../bins
python ../src/reformat_model.py \
/mnt/gs21/scratch/groups/compbio/2024-03-29_nbc/models/mode-full_test__task-disease__instance-study__preprocess-l2__lemmatize-False/method-no_cluster__ent_type-word__model-LR__c-1.0__l1r-0.4 \
../bins/

python ../src/reformat_model.py \
/mnt/gs21/scratch/groups/compbio/2024-03-29_nbc/models/mode-full_test__task-tissue__instance-sample__preprocess-l2__lemmatize-False/method-no_cluster__ent_type-word__model-LR__c-1.0__l1r-0.4 \
../bins/

# check into prediction see why models cannot distinguish among some similar terms
# frontal lobe
python ../src/predict.py \
-input ../data/tissue_desc_processed.tsv \
-id ../data/tissue_ID.tsv \
-input_embed ../data/tissue_desc_embedding.npz \
-train_embed ../data/tissue_desc_embedding.npz \
-model ../bins/UBERON_0016525__model.pkl \
-out ../results

# frontal cortex
~/anaconda3/envs/txt2onto2/bin/python ../src/predict.py \
-input ../data/tissue_desc_processed.tsv \
-id ../data/tissue_ID.tsv \
-input_embed ../data/tissue_desc_embedding.npz \
-train_embed ../data/tissue_desc_embedding.npz \
-model ../bins/UBERON_0001870__model.pkl \
-out ../results

# bronchial epithelial cell
~/anaconda3/envs/txt2onto2/bin/python ../src/predict.py \
-input ../data/tissue_desc_processed.tsv \
-id ../data/tissue_ID.tsv \
-input_embed ../data/tissue_desc_embedding.npz \
-train_embed ../data/tissue_desc_embedding.npz \
-model ../bins/CL_0002328__model.pkl \
-out ../results

python ../src/predict.py \
-input ../data/disease_desc_processed.tsv \
-id ../data/disease_ID.tsv \
-input_embed ../data/disease_desc_embedding.npz \
-train_embed ../data/disease_desc_embedding.npz \
-model ../bins/MONDO_0005034__model.pkl \
-out ../results

python ../src/predict.py \
-input ../data/disease_desc_processed.tsv \
-id ../data/disease_ID.tsv \
-input_embed ../data/disease_desc_embedding.npz \
-train_embed ../data/disease_desc_embedding.npz \
-model ../bins/MONDO_0005148__model.pkl \
-out ../results

python ../src/predict.py \
-input ../data/disease_desc_processed.tsv \
-id ../data/disease_ID.tsv \
-input_embed ../data/disease_desc_embedding.npz \
-train_embed ../data/disease_desc_embedding.npz \
-model ../bins/MONDO_0005097__model.pkl \
-out ../results

python ../src/predict.py \
-input ../data/disease_desc_processed.tsv \
-id ../data/disease_ID.tsv \
-input_embed ../data/disease_desc_embedding.npz \
-train_embed ../data/disease_desc_embedding.npz \
-model ../bins/MONDO_0018906__model.pkl \
-out ../results
