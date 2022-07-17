# Fake News Detection on COVID-19

## 
![d2k](https://media.licdn.com/dms/image/C4E0BAQHJaTCnDZoaNw/company-logo_200_200/0?e=2159024400&v=beta&t=kMEj3ZLaQ1RzT9MdcxHIbC2IOT3eyPFiKz3yRVrv5Fo)


This project aims to gather COVID-19 news data and create a dataset of tagged Reliable and Unreliable news based on machine learning models.

## Objectives 

Building upon existing works in the research area of fake news detection, which is already quite thoroughly explored, we aim to achieve better prediction accuracy than a generic-purpose fake news detector by exploiting the particularity of COVID-related news and employing state-of-the-art ML and deep learning (DL) models specifically tuned to work with COVID-related news articles. 

## Dataset / Data Wrangling

The primary raw dataset we will be processing before use is retrieved from **Aylien**, containing over 1,600,000 COVID-related news articles with numerous enriched features. Since **the raw Aylien dataset is not labeled.**, we manually labeled part of the **Aylien** dataset in the data wrangling process. 

**FNIR**, a secondary dataset will also be considered. This dataset comes reliably fact-checked and labeled, and will be merged with the labeled portion of the **Aylien** dataset to form our final dataset/

For more details on the datasets and the wrangling process, see [**Resources**](#resources) section, and the file `data_wrangling/demo`.


## Repository Description

- `data_visualization`: directory containing code to visualize data.
    - `data_visualization`: script for cleaning up the our final dataset (generated in `data_wrangling`).
    - `demo`: Jupyter notebook that serves the same purpose as `data_visualization` but is designed for demonstration purposes.
    - `utils`: functions used in `data_visualization` and `demo`.

- `data_wrangling`:  directory containing code to clean up data.
    - `data_wrangling`: script for data wrangling procedures and can save intermediate results according to configurations.
    -  `demo`: Jupyter notebook that serves the same purpose as `data_wrangling` but is designed for demonstration purposes.
    - `utils`: functions used in `data_wrangling` and `demo`.

- `modeling`: directory containing code to generate models.
    - `BERT`: directory containing code to generate the BERT model.
        - `data_preprocessing`: read our dataset from the csv.
        - `demo`: Jupyter notebook that serves the same purpose as `ML_modeling` but is designed for demonstration purposes.
        - `FakeNewsDataset`: Wrapper class containing functions to tokenize data.
        - `main`: Driver program to execute all procedures in training and testing BERT.
        - `test`: Test the performance of BERT.
        - `train`: Train BERT.
        - `utils`: Helper functions used in other code.
        - `validate`: Validate and ensure robustness of BERT with real news.
    - `ML_models`: directory containing code to generate traditional machine learning models.
        - `misclassified_data`: directory containing `.csv` files of misclassified data by different models.
        - `model_instances`: pickle files of the saved models, which have been fine-tuned, trained, and finalized. *Note that due to GitHub Repo file size limits, we are not able to upload our saved SVM model in this folder.*
        - `ML_modeling`: script for deploying Logistic Regression, Support Vector Machine and Naive Bayes models and for generating misclassified data.
        - `demo`: Jupyter notebook that serves the same purpose as `ML_modeling` but is designed for deonstration purposes.
        - `utils`: functions used in `ML_modeling` and `demo`.
        - `vectorizer`: pickle file of the TF-IDF vectorizer used in the modeling process. It has been fitted on our dataset.
    - `BERT`: directory containing code to deploy state-of-the-art deep learning model - Bidirectional Encoder Representations from Transformers (BERT).

- `tools`:  directory containing news-credibility-check tools used in `data_wrangling`.
    - `aylien_reliable_sources` : a manually compiled list of highly credible news sources present in **Aylien**.

- `app_demo_for_users`: a short Python application that allows users to input a given COVID news title and outputs a prediction (by our Logistic Regression model) on whether the input entry is fake or not.

- `config.py`: contains several options for running the code.

Note that the data needed to run the code in the `data_wrangling` directory are not available in this GitHub repository. However, they are accessible on our Google Drive. For more details, see the [**Resources**](#resources) section. Procedures regarding running the code will be discussed in the [**Running the Code**](#running-the-code) section.


## Running the code

If you wish to run the code together with the data, you may follow these steps for the correct directory setup.

1. Download the following items:
    - the folders available in this GitHub repository,
    - the `data` folder on our Google Drive (See [**Resources**](#resources)),
    - the raw **Aylien** dataset (See [**Resources**](#resources)).
2. Put the downloaded raw **Aylien** dataset file (`aylien_covid_news_data.jsonl`) inside the directory `data/raw/`.
3. Move the `data` directory so that it is in the **same parent directory** as the folders available in this GitHub repository.
4. Run the code.

## Resources

### **Our Google Drive**: 
Our Google Drive can be accessed via https://drive.google.com/drive/folders/1u6bHzkLcPs0myTNW5O_Ev8jFCzOsmg9E?usp=sharing, and its structure is as described below:

- `Check-ins` : Containing the weekly check-in presentations to report our progress.
- ``data`` : Containing both raw and processed data used in our project.
    - `raw` : Containing raw data from three different sources.
        - `fakeNews.csv` : Fake news entries provided by the **FNIR** dataset. Some preliminary manual cleanups have been done on it (See the file `data_wrangling/demo`).
        - `aylien_preprocessed` : Pickle object containing a subset of size ~50,000 of the raw **Aylien** dataset. https://aylien.com/blog/free-coronavirus-news-dataset
        
    - `processed` : Containing three processed datasets in possibly different formats.
        - `aylien_true` : Processed subset of the **Aylien** dataset that has been labeled **TRUE**.
        - `aylien_unlabeled` : Processed subset of the **Aylien** dataset that has been left unlabeled.
        - `final`: Final dataset to be used.
        - `fnir_fake`: Processed **FALSE** portion of the **FNIR** dataset.
- `Models`: Containing model instances we generated throughout the project
- `Presentations`: Containing our Initial, Midterm, and Final Presentations.
- `Reports`: Containing our Initial, Midterm, and Final Reports.


### **Raw Aylien Dataset**: 
Although not posted on the GitHub repository or the Google Drive due to its large size, the raw **Aylien** dataset is available on https://aylien.com/blog/free-coronavirus-news-dataset. A full detailed description of the dataset can be found on the aforementioned website. 
