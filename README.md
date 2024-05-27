# Annotating public omics samples and studies using interpretable modeling of unstructured metadata

Today, millions of publicly available omics samples are accessible via resources such as GEO, SRA, and PRIDE. Although these data are available, finding datasets and samples relevant to a specific biological context remains difficult because they are described using unstructured free text, called "metadata", which is not easily machine-readable. 

Here, we present *txt2onto 2.0* a Python utility for classifying metadata of samples and studies to controlled vocabularies in tissue and disease ontologies using a combination of natural language processing (NLP) and machine learning (ML). Comparing to previous version [*txt2onto 1.0*](https://github.com/krishnanlab/txt2onto), which represents metadata using the average embedding of words, *txt2onto 2.0* employs a TF-IDF matrix for metadata representation. This modification brings better classification performance, particularly when positive training instances are limited. By utilizing the TF-IDF matrix, users can easily identify the key biomedical words that strongly influence the model's predictions through model coefficients, bringing better transparency to the model than *txt2onto 1.0*. Detailed implementation and benchmark are described in this preprint bioRxiv DOI: xxx.

Here is a summary of comparison between *txt2onto 1.0* and *2.0*

| Feature | *txt2onto 1.0* | *txt2onto 2.0* |
|----------|----------|----------|
| Tissue classification (UBERON ontology) | yes | yes |
| Number of tissue classification models | 296 | 402 |
| Cell type classification (CL ontology) | yes | yes |
| Number of cell type classification models | 50 | 188 |
| Disease classification (MONDO ontology) | no | yes |
| Number of Disease classification models | 0 | 1,166 |
| Interpretable model features | no | yes |
| Extracting predictive words from input metadata | no | yes |

# Installation

Install environments
```
git clone https://github.com/krishnanlab/txt2onto2.0.git
cd txt2onto2.0
conda create -n txt2onto2 python==3.11.3
conda activate txt2onto2
pip install -r requirements.txt
```

# Usage
## Making predictions using provided models
Assume we are going to run code in the `src` folder. We will use the following tutorials to showcase how to use provided models for predictions. 

We also provide `../demo/demo_prediction.sh` as example to demonstrate using *txt2onto 2.0* for prediction.

### Models
To figure out what models are available in *txt2onto 2.0*. Users can check `../data/disease_model_stats.csv` for disease models and `../data/tissue_model_stats.csv` for tissue and cell type models. Here we show few beginning lines of the table:

| ID           | name                                          | log2(auprc/prior) | num_of_pos | pred_words | coef |
|--------------|-----------------------------------------------|-------------------|------------| -----------| ------------|
| MONDO:0000167 | Huntington disease and related disorders      | 5.841352896201352 | 35         | hd,huntingtons,htt,disease | 6.583,5.902,2.170,2.033 |
| MONDO:0000248 | dengue shock syndrome                         | 9.517669388133813 | 3          | dengue,dss | 2.198,1.272 |
| MONDO:0000270 | lower respiratory tract disorder              | 3.9844081310560657| 170        | lung,asthma,nsclc,airway | 12.366,4.467,3.501,3.130 |
| MONDO:0000314 | primary bacterial infectious disease          | 6.607595291876909 | 26         | tb,tuberculosis,ltbi,infection | 4.979,3.298,1.459,1.304 |
| MONDO:0000315 | commensal bacterial infectious disease        | 8.519636252843213 | 4          | trachoma,conjunctival,trachomatis | 1.498,1.091,0.098 |

Each column are:
- `ID`: The first column provides the ID for the tissue or disease. Tissue IDs come from the CL and UBERON ontologies, while disease IDs come from the MONDO ontology.
- `name`: The second column contains the name of the tissue or disease term.
- `log2(auprc/prior)`: The third column indicates the performance of the model, which is evaluated based on metadata from GEO. Generally, `log2(auprc/prior) >= 2` indicates a good model.
- `num_of_pos`: The fourth column shows the number of positive instances used during the training of the model.
- `pred_words`: The fifth column lists the predictive word features of the model, ordered in descending order of their coefficients.
- `coef`: The final column provides the coefficient value for each predictive word feature, higher means more important to the predictions.

We used study-level metadata from GEO to train disease classification models, and used sample-level metadata from GEO to train tissue classification models. Notably, we did not train a disease classification model for sample-level metadata, since disease information is often only available in study-level metadata in GEO. Study-level metadata typically do not include explicit information for tissues, thus, we also did not train a tissue classification model for study-level metadata. As far as we know, this rule generally applies to metadata from any databases, so we recommend use study-level metadata to predict diseases, and sample-level metadata to predict tissues.

### Step 1: Prepare user input
Input for making predictions is the concatenated text from metadata, with one description per line. Here we use an example study [NCT00313209](https://clinicaltrials.gov/study/NCT00313209) from ClinicalTrial to show how to generate description.

Here we use `title` and `brief summary` of the study as input.

The title of this study is:
```
Effect of Roflumilast on Lung Function in Chronic Obstructive Pulmonary Disease (COPD) Patients Treated With Salmeterol: The EOS Study (BY217/​M2-127) (EOS)
```

Brief summary is:
```
The aim of the study is to compare the efficacy of roflumilast on pulmonary function and symptomatic parameters in patients with chronic obstructive pulmonary disease (COPD) during concomitant administration of salmeterol. The study duration will last up to 28 weeks. The study will provide further data on safety and tolerability of roflumilast.
```

We simply concatenate the content of `title` and `brief summary` together without the field names. The concatenated description looks like:
```
Effect of Roflumilast on Lung Function in Chronic Obstructive Pulmonary Disease (COPD) Patients Treated With Salmeterol: The EOS Study (BY217/M2-127) (EOS). The aim of the study is to compare the efficacy of roflumilast on pulmonary function and symptomatic parameters in patients with chronic obstructive pulmonary disease (COPD) during concomitant administration of salmeterol. The study duration will last up to 28 weeks. The study will provide further data on safety and tolerability of roflumilast.
```

Additionally, we need to prepare an ID file, where one ID per line. The order of the ID should follow the order of metadata in the descriptions file.

`../demo/clinicaltrials_desc.tsv` is an example of the description file, which shows 5,000 prepared descriptions collected from ClinicalTrials. `../demo/clinicaltrials_ID.tsv` is the corresponding ID file. User can refer to these files for detail. We do not provide any scripts to prepare metadata, since metadata from various database comes in different format.

### Step 2: Preprocess input text
The second step is to preprocess the descriptions by removing potential uninformative characters, which might affect predicting process. This step involves removing stop words, punctuation, and URLs, and finally convert words to lowercase.
```
python preprocess.py \
-input <path_to_description_file.tsv> \
-out <path_to_processed_description_file.tsv>
```
We read the concatenated metadata from `path_to_description_file.tsv`, then output processed text `path_to_processed_description_file.tsv`, in which uninformative elements have been cleaned.

### Step 3: Generate an embedding table for unique words in metadata
We need to generate word embedding for all unique words in the preprocessed descriptions. We need this embedding to map the unseen words in the descriptions to training features in the provided models.
```
python embedding_lookup_table.py \
-input <path_to_processed_description_file.tsv> \
-out <path_to_embedding_table.npz> \
-batch_size 5000
```
We read `path_to_processed_description_file.tsv` as input, get all unique words in the text, then calculate text embedding of each word, and output to `path_to_embedding_table.npz`. Although this script supports using CPUs to generate embeddings, we recommend using GPU for faster embedding generation. The default model to generate embedding is [BiomedBERT](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract) pretrained on abstracts from PubMed. By default, when using a GPU, the batch size for calculating embeddings is set to 5,000. However, if this exceeds the available VRAM of your computational resource, you can reduce the `-batch_size` parameter to accommodate your hardware constraints.

### Step 4: Predict labels
Now, we can use previously generated files to predict disease or tissue labels
```
python ../src/predict.py \
-input <path_to_processed_description_file.tsv> \
-id <path_to_ID_file.tsv> \
-input_embed <path_to_embedding_table.npz> \
-train_embed ../data/*_desc_embedding.npz \
-model ../bins/<name_of_task>__model.pkl \
-out <path_to_output_directory>
```

This script takes following files that generated from previous steps:
- Processed description; `path_to_processed_description_file.tsv` from step 2.
- Corresponding ID file: `path_to_ID_file.tsv` from user input.
- Embedding of the descriptions: `path_to_embedding_table.npz` from step 3.

We also need embeddings for disease or tissue model features `../data/*_desc_embedding.npz`. We need to input disease model features embedding  `../data/disease_desc_embedding.npz` for disease classifications, or tissue model features embedding `../data/tissue_desc_embedding.npz` for tissue classifications.

We need model files `../bins/<name_of_task>__model.pkl` for disease or tissue of interest. For example, if you want to predict studies related to atrial fibrillation (MONDO:0004981), the path to model file is `../bins/MONDO_0004981__model.pkl`. **NOTE: when input model names, we need to substitute semicolon ":" as underscore "_"**

This script produces the output to  `path_to_output_directory`, which is named as `path_to_output_directory/<name_of_task>__preds.csv`. The result file looks like this:

| ID           | prob                | log2(prob/prior)    | related_words             |
|--------------|--------------------:|--------------------:|---------------------------|
| NCT04496336  | 0.7842907444469499  | 8.686026609365644   | atrial                    |
| NCT01970969  | 0.743665375713356   | 8.609291637672486   | atrial                    |
| NCT01796080  | 0.5746653363470876  | 8.237360063179139   | atrial, supraventricular  |
| NCT00848445  | 0.4652301906664786  | 7.932582756502403   | af, atrial                |

`ID` is the ID of every instance. `prob` is the prediction probability. `log2(prob/prior)` is the probability normalized by `prior`, where `prior` represents the expected random prediction for this model. Thus, `log2(prob/prior)` indicates how much this prediction is better than random. The final column `related_words` represents words in the given text that are related to predictions. `prob` might be underestimated when very few positive instances are included during training. `log2(prob/prior)` is a good indicator of good performance. Generally `log2(prob/prior) >= 1` means trustable positive predictions.

## Training new models
Besides making predictions using the provided models, users can also train their own models for diseases or tissues outside of our provided list. Users can also use our framework to predict any kind of labels using metadata, such as sex, sequencing platform or drug.

We also provide `../demo/demo_training.sh` as example to demonstrate using *txt2onto 2.0* to train new model from scratch.

Again, assume we are running code under the `src` folder. We will demonstrate how to training a model using metadata from scratch.

### Step 1: Prepare user input
To train a new model, we first need metadata for training. The preparation process is exactly the same as in **Step 1: Prepare user input** from the **Making predictions using provided models** section. Users can also refer to `../demo/disease_desc.tsv` for detail, which is the prepared metadata used for training disease models from the paper.

Similarly, we need corresponding ID files for training data, one ID per line. Please refer to `../demo/disease_ID.tsv` for example.

Additionally, we need to prepare ground truth labels, which specifies the ground truth label for the correspondence between an instance and a term. It should be in CSV format. To save space, users can also compress the table by `gzip`. Thus, input file should either be `*.csv` or `*.csv.gz`.

In this file, each column is an ontology, and each row is an instance. Each cell corresponds to the category of the instance, either positive (1), negative (-1), or unknown (0). Please check `../demo/disease_labels.csv.gz` for details, which is the gold standard used in the paper. Here is a snapshot of the gold standard file:

|      | MONDO:0000001 | MONDO:0000004 | MONDO:0000005 |
|-----------|--------------:|--------------:|--------------:|
| GSE85493  | 1             | -1            | -1            |
| GSE74016  | 1             | -1            | -1            |
| GSE135461 | 1             | 0             | -1            |
| GSE93624  | 1             | -1            | -1            |
| GSE127948 | 1             | -1            | -1            |

If users want to predict labels other than tissue or disease. The column does not have to be ontology. For example, the gold standard for sex prediction could be:
|      | sex |
|-----------|--------------:|
| GSE85493  | 1             |
| GSE74016  | -1            |
| GSE135461 | 0             |
| GSE93624  | -1            |
| GSE127948 | 1             |

### Step 2: Preprocess input text for training data
This step is to preprocess the metadata for training. The command is exactly the same as **Step 2: Preprocess input text** from the **Making predictions using provided models** section.
```
python preprocess.py \
-input <path_to_description_file.tsv> \
-out <path_to_processed_description_file.tsv>
```
Now, we input metadata of training description `path_to_description_file.tsv`, then get processed output `path_to_processed_description_file.tsv`

### Step 3: Generate an embedding table for unique words in training metadata
This step is to generate embedding for unique words in training metadata. The command is exactly the same as **Step 3: Generate an embedding table for unique words in metadata** from the **Making predictions using provided models** section.
```
python embedding_lookup_table.py \
-input <path_to_processed_description_file.tsv> \
-out <path_to_embedding_table.npz> \
-batch_size 5000
```
We generate embedding for preprocessed text `path_to_processed_description_file`, then get embedding file `path_to_embedding_table.npz`. We don't need this file for training, but we do need it during prediction.

### Step 4: Prepare train inputs
In this step, we need to prepare training input for a classification task. Users can loop this process to prepare input for every task.
```
python input.py \
-gs <path_to_gold_standard.csv.gz> \
-onto <name_of_task> \
-text <path_to_processed_description_file.tsv> \
-id <path_to_ID_file.tsv> \
-out <path_to_output_directory>
```
We need input from users:
- `-gs`  the gold standard files prepared in **Step 1: Prepare user input** in this section.
- `-onto` is the name of the task for which you want to generate input. It has to be one of the column name in `path_to_gold_standard.csv.gz`.
- `-text` is processed training text prepared in **Step 2: Preprocess input text for training data** in this section.
- `-id` is the ID of the training instances that need to be provided by users in **Step 1: Prepare input** in this section.

This script produces the output to `path_to_output_directory`. The output file is `path_to_output_directory/<name_of_task>__train_input.tsv`. The output file contains the ID, label, and text required for training a model. The prepared input file is formatted in the following structure:
```
ID	label	text
ID1  1  TEXT
ID2 -1  TEXT
ID3 -1  TEXT
ID4  1  TEXT  
```
where the first column is ID, the second column is label and the last column is the processed text. 

### Step 5: Train model
Finally, we train the model using prepared input. We train it using logistic regression with word as features. 
```
python train.py \
-input <path_to_prepared_train_input.tsv> \
-out <path_to_output_directory>
```
It takes prepared text `path_to_prepared_train_input.tsv` as input, then output trained model to output directory `path_to_output_directory`, the output model will be `<path_to_output_directory>/<name_of_task>__model.pkl`.

# Overview of the repository
Here, we list the files we included in the repository.
- `bin` contains all provided models stored as pickle files.
- `data` folder contains files other than models required for prediction
  - `disease_desc_embedding.npz`: word embedding matrix for features in the provided disease models
  - `tissue_desc_embedding.npz`: word embedding matrix for features in the provided tissue models
  - `disease_model_stats.csv`: full list of provided disease models and their performance statistics
  - `tissue_model_stats.csv`: full list of provided tissue models and their performance statistics
- `demo` user input files in demo
  - input files for label prediction
    - `demo_prediction.sh`: demo for predicting disease and tissues using existing data
    - `clinicaltrials_ID.tsv`: Study ID from ClinicalTrials
    - `clinicaltrials_desc.tsv`: Corresponding unprocessed study descriptions from ClinicalTrials
    - `geo_sample_desc_ID.tsv`: Sample ID from GEO
    - `geo_sample_desc.tsv`: Corresponding unprocessed sample descriptions from GEO
  - input files for building classification models
  - `demo_training.sh`: demo for training model from scratch
    - files for building disease classification models
      - `disease_ID.tsv`: Study ID from GEO
      - `disease_desc.tsv`: Corresponding unprocessed descriptions for studies
      - `disease_labels.csv.gz`: Curated gold standard matrix for diseases, labels have been propagated to general terms
    - files for building disease classification models
      - `tissue_ID.tsv`: Sample ID (the first column) and study ID (the second column) from GEO
      - `tissue_desc.tsv`: Corresponding unprocessed descriptions for samples
      - `tissue_labels.csv.gz`: Curated gold standard matrix for tissues, labels have been propagated to general terms
- `results`: Dir to put intermediate and final output from demo
- scripts required for training and prediction
  - `preprocess.py`: Script to preprocess text
  - `embedding_lookup_table.py`: Script to generate word embedding matrix
  - `predict.py`: Script to predict labels
  - `input.py`: Script to prepare input for training
  - `train.py`: Script to train models using prepared input
- other utility scripts:
  - `tfidf_calculator.py`: Modules used to calculate TF-IDF
  - `model_builder.py`: Modules used to build classification models

# Additional Information

### Support
For support, please contact Hao Yuan at yuanhao5@msu.edu.

### Inquiry
All general inquiries should be directed to [Arjun Krishnan](www.thekrishnanlab.org) at arjun.krishnan@cuanschutz.edu

### License
This repository and all its contents are released under the [Creative Commons License: Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode); See [LICENSE](https://github.com/krishnanlab/txt2onto2/blob/main/LICENSE).

### Citation
If you use this work, please cite: 

### Funding
This work was primarily supported by US National Institutes of Health (NIH) grants R35 GM128765 to AK and in part by MSU start-up funds to AK and MSU Rasmussen Doctoral Recruitment Award and Engineering Distinguished Fellowship to NTH.

### Acknowledgements
The authors would like to thank Keenan Manpearl for valuable suggestions on the repo documents and testing the code. We also thank all members of the [Krishnan Lab](www.thekrishnanlab.org) for valuable discussions and feedback on the project.
