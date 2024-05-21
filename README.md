# Annotating public omics samples and studies using interpretable modeling of unstructured metadata

*txt2onto 2.0* is a Python utility for classifying unstructured descriptions of samples and studies to controlled vocabularies in tissue and disease ontologies using a combination of natural language processing (NLP) and machine learning (ML). In this repo, we provide trained models that can be directly used for tissue and disease annotation. We also provide a demo script to train models from user-provided metadata.

The *txt2onto 2.0* method is described in this preprint bioRxiv DOI: xxx.

## More info
Today, there are millions of publicly available omics samples accessible via resources like the Gene Expression Omnibus (GEO), the Sequence Read Archive (SRA), PRIDE, and MetaboLights. In GEO alone, there are currently >7.1 million samples belonging to >224 thousand studies that each contribute to a vast collection of genomics samples from a variety of biological contexts. This massive data collection can be incredibly valuable in elucidating undiscovered characteristics of the molecular basis of tissues, phenotypes, and environments. However, although these data are available, finding datasets and samples relevant to a biological context of interest is still difficult because these data are described using unstructured, unstandardized, free text, called "metadata", which is not easily machine-readable. Previously, Hawkin et al. developed *txt2onto 1.0* to automatically annotate tissue labels for samples from GEO using sample-level descriptions from metadata. *txt2onto 1.0* represented sample descriptions as the average embeddings of words in metadata, then trained ML models to annotate tissue and cell types using these embeddings. One issue with the txt2onto approach is that averaging the embeddings of all words in a description can dampen the signal from informative biomedical terms. Furthermore, the trained model coefficients of the embedding features do not provide insight into which specific words in the sample descriptions contributed to the model's predictions, limiting interpretability.

Building upon our previous work, [*txt2onto 1.0*](https://github.com/krishnanlab/txt2onto), we present *txt2onto 2.0*, a novel and lightweight approach for metadata annotation. Compared to the previous version, *txt2onto 2.0* introduces improvements in both interpretability and performance. Instead of using average word embeddings as features, *txt2onto 2.0* converts metadata to a TF-IDF matrix, which serves as input to the ML classifier. During the prediction phase, our model accepts a TF-IDF matrix from new metadata as input and leverages word embeddings from a large language model (LLM) to match unseen words in the new metadata. By utilizing text as features, our model can track the key biomedical words that strongly influence model predictions through model coefficients. Since *txt2onto 2.0* uses a TF-IDF matrix to represent text, which directly uses words or phrases as features, it naturally avoids mixing signals from predictive words with the rest of the text. *txt2onto 2.0* is more sensitive to signals from predictive words, resulting in better performance than *txt2onto 1.0*, especially when there are a limited number of positive instances available for training. We have also demonstrated that *txt2onto 2.0* is a versatile annotation tool, which can be used to annotate metadata from any source. 

In the 2.0 implementation, besides annotating tissue and cell type for samples, we also trained models to annotate disease labels for studies using study-level metadata. All samples and studies can be annotated to standardized vocabularies in ontologies. Studies are annotated to the MONDO ontology. Samples are annotated to cell type or tissue from the CL or UBERON ontology. Notably, we did not train a disease classification model for sample-level metadata, since in GEO, disease information is often only available in study-level metadata. Study-level metadata typically do not include explicit information for tissues; thus, we also did not train a tissue classification model for study-level metadata. Despite only using a non-redundant set of disease and tissue terms to assess performance in the article, we provide a full list of models under `bins`. Performance of models are available at `data/*_model_stats.csv`

# Installation

Install enviroments
```
conda create -n txt2onto2 python==3.9.16
conda activate txt2onto2
pip install -r requirements.txt
```

# Usage
## Making predictions using provided models
Assume we are going to run code in the src folder. We will use the following code to showcase the implementation of *txt2onto 2.0* using studies from ClinicalTrials as input and see which metadata are related to atrial fibrillation (MONDO:0004981).

Input for making predictions is the concatenated text from metadata, with one description per line. The metadata is sourced from ClinicalTrials. [Here](https://github.com/krishnanlab/txt2onto2.0/blob/main/data/clinicaltrials_desc.tsv) are the first few lines of an example using concatenated metadata from ClinicalTrials:

```
Effect of Roflumilast on Lung Function in Chronic Obstructive Pulmonary Disease (COPD) Patients Treated With Salmeterol: The EOS Study (BY217/M2-127) The aim of the study is to compare the efficacy of roflumilast on pulmonary function and symptomatic parameters in patients with chronic obstructive pulmonary disease (COPD) during concomitant administration of salmeterol. The study duration will last up to 28 weeks. The study will provide further data on safety and tolerability of roflumilast.
Pulmonary Vein Isolation With Versus Without Continued Antiarrhythmic Drugs in Persistent Atrial Fibrillation In the POWDER 1 study, paroxysmal atrial fibrillation (AF) patients undergoing conventional contact force (CF)-guided PVI were investigated. Patients were randomized between continuing previously ineffective antiarrhythmic drug therapy (ADT) or stopping ADT at the end of the blanking period. This trial, showed an added value of ADT after ablation (in support of 'hybrid rhythm control' as an alternative treatment strategy for AF in some patients).  In the POWDER 2 trial, an analogue study in persistent AF patients will be performed. All patients will undergo ablation index (AI)- and IL distance (ILD)-guided PVI (just like in VISTAX trial) and continue previously ineffective ADT during the blanking period. 'PVI only' was chosen as the ablation strategy according to the STAR AF trial findings.
Chloroquine as an Anti-Autophagy Drug in Stage IV Small Cell Lung Cancer (SCLC) Patients Chloroquine might very well be able to increase overall survival in small cell lung cancer by sensitizing cells resistant to chemotherapy and radiotherapy.
```

We need the following three steps to get predictions.

Step 1: Preprocess input text, remove potential uninformative characters, including stop words, punctuation, and URLs, and finally convert words to lowercase.
```
python ../src/preprocess.py \
-input ../data/clinicaltrials_desc.tsv \
-out ../data/clinicaltrials_desc_processed.tsv
```
We read the concatenated metadata from `../data/clinicaltrials_desc.tsv `, then get processed text `../data/clinicaltrials_desc_processed.tsv`, in which uninformative elements have been cleaned.

Step 2: Generate an embedding table for given text data from ClinicalTrials
```
python ../src/embedding_lookup_table.py \
-corpus ../data/clinicaltrials_desc_processed.tsv \
-out ../data/clinicaltrials_desc_embedding.npz
```
We read `../data/clinicaltrials_desc_processed.tsv` as input, get all words in the text, then calculate text embedding of each word. Although this script supports using CPUs to generate embeddings, it will take hours for thousands of words. We recommend using GPU for fast embedding generation. The default model to generate embedding is [BiomedBERT](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract) pretrained on abstracts from PubMed.

Step 3: Predict labels
```
python ../src/predict.py \
-input ../data/clinicaltrials_desc_processed.tsv \
-id ../data/clinicaltrials_ID.tsv \
-input_embed ../data/clinicaltrials_desc_embedding.npz \
-train_embed ../data/disease_desc_embedding.npz \
-model ../results/MONDO_0004981__model.pkl \
-out ../results
```
We read the following files as input:

- `-input` specifies the processed text file, which is `../data/clinicaltrials_desc_processed.tsv`
- `-id` specifies the ID of each input text. The order should be the same as the file for processed text. Check the example file `../data/clinicaltrials_ID.tsv` for details. Users can include multiple columns to provide additional information about each instance, but only the first column will be used in the analysis. The first column must contain unique IDs for each instance.
- `-input_embed` specifies the word embedding matrix for all words in the processed text, which was generated in the last step.
- We also need the embedding matrix from the training data as input. Since we are trying to predict disease labels, we use `../data/disease_desc_embedding.npz` as input to `-train_embed`, which is the embedding matrix of words for disease model features.
- `-model` is used to input the model file. In this example, the model file `../results/MONDO_0004981__model.pkl` is for atrial fibrillation (MONDO:0004981) prediction.

This script produces the following output:
- `-out` specifies the output directory. The output from this code will be `../results/MONDO_0004981__preds.csv`, which includes the probability of prediction for every input metadata and task-related words for each prediction. Here is the first few lines of the output.

| ID           | prob                | log2(prob/prior)    | related_words             |
|--------------|--------------------:|--------------------:|---------------------------|
| NCT04496336  | 0.7842907444469499  | 8.686026609365644   | atrial                    |
| NCT01970969  | 0.743665375713356   | 8.609291637672486   | atrial                    |
| NCT01796080  | 0.5746653363470876  | 8.237360063179139   | atrial, supraventricular  |
| NCT00848445  | 0.4652301906664786  | 7.932582756502403   | af, atrial                |

`ID` is the ID of every instance, in this case, is the study ID from ClinicalTrials. `prob` is the prediction probability. `log2(prob/prior)` is the probability normalized by `prior`, where `prior` represents the expected random prediction for this model. Thus, `log2(prob/prior)` indicates how much this prediction is better than random. The final column `related_words` represents words in the given text that are related to predictions. `prob` might be underestimated when very few positive instances are included during training. `log2(prob/prior)` is a good indicator of good performance. Generally `log2(prob/prior) >= 2` means trustable positive predictions.

Besides the example we showed here, we also provide a collection of models for tissue and disease classification. All available models are under `../bin`. The files `../data/*_model_stats.csv` specify the performance of all models. Generally, if `log2(auprc/prior) >= 2`, it indicates a good model.

## Training new models
Besides making predictions using the provided models, users can also train their own models. Again, assume we are running code in the `src` folder. We will demonstrate this by training a model for atrial fibrillation (MONDO:0004981) predictions.

Step 1: preprocess input text for training data
```
python ../src/preprocess.py \
-input ../data/disease_desc.tsv \
-out ../data/disease_desc_processed.tsv
```
First, we preprocess the training text, which is also concatenated text from metadata. Here, we take `../data/disease_desc.tsv` as input, which includes all the positive and negative instances of atrial fibrillation (MONDO:0004981). The preprocessed text will be outputted to `../data/disease_desc_processed.tsv`, where uninformative elements were cleaned. Notably, it is common to train multiple models simultaneously, and the training data for those models might overlap. To streamline the process, users can include all the metadata required for training all the models in a single file.

Step 2: generate an embedding table for training data
```
python ../src/embedding_lookup_table.py \
-corpus ../data/disease_desc_processed.tsv \
-out ../data/disease_desc_embedding.npz
```
Now, we generate embedding for preprocessed text, similar to step 2 from [Making prediction using provided models](#making-prediction-using-provided-models). We don't need it for training, but we do need it during prediction.

Step 3: prepare train inputs
```
python ../src/input.py \
-gs ../data/disease_labels.csv.gz \
-onto MONDO:0004981 \
-text ../data/disease_desc_processed.tsv \
-id ../data/disease_ID.tsv \
-out ../data
```
Before training the model, we need to prepare an input, which requires the following files as input:
- `-gs`  is a CSV file for the gold standard, which specifies the ground truth label for the correspondence between an instance and a term. In this file, each column is an ontology, and each row is an instance. Each cell corresponds to the category of the instance, either positive (1), negative (-1), or unknown (0). Please check `../data/disease_labels.csv.gz` for details. Here is a snapshot of the gold standard file:

|      | MONDO:0000001 | MONDO:0000004 | MONDO:0000005 |
|-----------|--------------:|--------------:|--------------:|
| GSE85493  | 1             | -1            | -1            |
| GSE74016  | 1             | -1            | -1            |
| GSE135461 | 1             | 0             | -1            |
| GSE93624  | 1             | -1            | -1            |
| GSE127948 | 1             | -1            | -1            |

- `-onto` is the ontology ID for which you want to generate input. In this example, it is MONDO:0004981.
- `-text` is processed training text from step 1
- `-id` is the ID of the training instances. The row number of the ID file should correspond to the description from the input text. Users can include multiple columns to provide additional information about each instance, but only the first column will be used in the analysis. The first column must contain unique IDs for each instance.

This script produces the following output:
- `-out` specifies the output directory, in this example the output will be `../data/MONDO_0004981__train_input.tsv`. The output file contains the ID, label, and text required for training a model. The prepared input file is formatted in this structure:
```
ID	label	text
ID1  1  TEXT
ID2 -1  TEXT
ID3 -1  TEXT
ID4  1  TEXT  
```
where the first column is ID, the second column is label and the last column is the processed text. 

Step 4: train LR + word model
```
python ../src/train.py \
-input ../data/MONDO_0004981__train_input.tsv \
-out ../results
```
Finally, we train the model for atrial fibrillation (MONDO:0004981) prediction, which takes prepared text `../data/MONDO_0004981__train_input.tsv` as input, then output trained model to output directory `../results`, the output model will be `../results/MONDO_0004981__model.pkl`.

# Overview of the repository
Here, we list the files we included in the repository.
- `bin` contains all provided models stored as pickle files.
- `data` folder contains files other than models required for prediction and also files generated for the demo.
  - Input files required for disease and tissue prediction:
    - `disease_desc_embedding.npz`: word embedding matrix for features in the provided disease models
    - `tissue_desc_embedding.npz`: word embedding matrix for features in the provided tissue models
  - Full list of models provided and model performance
    - `disease_model_stats.csv`: full list of provided disease models
    - `tissue_model_stats.csv`: full list of provided tissue models
  - Demo files included in section [Making prediction using provided models](#making-prediction-using-provided-models)
    - `clinicaltrials_ID.tsv`: Study ID from ClinicalTrials
    - `clinicaltrials_desc.tsv`: Corresponding unprocessed descriptions
    - `clinicaltrials_desc_processed.tsv`: Processed text
    - `clinicaltrials_desc_embedding.npz`: Word embeddings for every word in the processed text
  - Data from manuscript. Some of them are also included in the demo outlined above
    - Disease classification
      - `disease_ID.tsv`: Study ID from GEO
      - `disease_desc.tsv`: Corresponding unprocessed descriptions for studies
      - `disease_desc_processed.tsv`: Processed text for studies
      - `disease_desc_embedding.npz`: Word embeddings for every word in processed text
      - `disease_labels.csv.gz`: Curated gold standard matrix for diseases, labels have been propagated to general terms
      - `MONDO_0004981__train_input.tsv`: Prepared input file to train a model for atrial fibrillation (MONDO:0004981) prediction
    - Tissue classification
      - `tissue_ID.tsv`: Sample ID (the first column) and study ID (the second column) from GEO
      - `tissue_desc.tsv`: Corresponding unprocessed descriptions for samples
      - `tissue_desc_processed.tsv`: Processed text for samples
      - `tissue_desc_embedding.npz`: Word embeddings for every word in the processed text
      - `tissue_labels.csv.gz`: Curated gold standard matrix for tissues, labels have been propagated to general terms
      - `UBERON_0019319__train_input.tsv`: Prepared input file to train a model for exocrine gland of integumental system (UBERON:0019319). We did not showcase the example in the demo above, but we included it in `src/demo.sh` so users can check the commands shown there to train a tissue classification model.
    - `results`: Files generated from the demo
      - `MONDO_0004981__model.pkl`: Trained model for atrial fibrillation (MONDO:0004981) prediction
      - `MONDO_0004981__preds.csv`: Atrial fibrillation (MONDO:0004981) prediction on ClinicalTrials studies
      - `UBERON_0019319__model.pkl`: Trained model for exocrine gland of integumental system (UBERON:0019319)
    - `src`: Scripts for *txt2onto 2.0*
      - `demo.sh`: Demo script for using the provided model for predictions and training a new model
      - `embedding_lookup_table.py`: Script to generate word embedding matrix
      - `preprocess.py`: Script to preprocess text
      - `input.py`: Script to prepare input for training
      - `train.py`: Script to train models using prepared input
      - `predict.py`: Script to predict labels
      - `tfidf_calculator.py`: Modules used to calculate TF-IDF
      - `model_builder.py`: Modules used to build classification models

# Additional Information

### Support
For support, please contact Hao Yuan at yuanhao5@msu.edu.

### Inquiry
All general inquiries should be directed to [Arjun Krishnan](www.thekrishnanlab.org) at arjun.krishnan@cuanschutz.edu

### License
This repository and all its contents are released under the [Creative Commons License: Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode); See [LICENSE.md](https://github.com/krishnanlab/txt2onto/blob/main/LICENSE).

<!-- ### Citation
If you use this work, please cite:  -->

<!-- ### Funding
This work was primarily supported by US National Institutes of Health (NIH) grants R35 GM128765 to AK and in part by MSU start-up funds to AK and MSU Rasmussen Doctoral Recruitment Award and Engineering Distinguished Fellowship to NTH. -->

### Acknowledgements
The authors would like to thank Keenan Manpearl for testing the code, and all members of the [Krishnan Lab](www.thekrishnanlab.org) for valuable discussions and feedback on the project.
