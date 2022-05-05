import warnings
import nltk
import argparse
import gensim
import pprint
import json
import math
import pandas as pd
import fitz
import os
import re
import glob
from nltk.corpus import stopwords
import spacy
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import collections
import numpy as np
warnings.filterwarnings("ignore")

nltk.download('stopwords')
nltk.download('vader_lexicon')


# --------------------------- Read a pdf into a large string of text ---------------------------
def read_pdf(file_path):
    pymupdf_text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            pymupdf_text += page.getText()
    return pymupdf_text


# --------------------------- Read a report and breaks it up into individual sentences ---------------------------
def convert_pdf_into_sentences(text):
    # Remove unnecessary spaces and line breaks
    text = re.sub(r'\x0c\x0c|\x0c', "", str(text))
    text = re.sub('\n ', '', str(text))
    text = re.sub('\n', ' ', str(text))
    text = ' '.join(text.split())
    text = " " + text + "  "
    text = text.replace("\n", " ")
    if "”" in text:
        text = text.replace(".”", "”.")
    if "\"" in text:
        text = text.replace(".\"", "\".")
    if "!" in text:
        text = text.replace("!\"", "\"!")
    if "?" in text:
        text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]

    # Filter for sentences with more than 100 characters
    sentences = [s.strip() for s in sentences if len(s) > 100]
    return sentences


# --------------------------- Retrieve the report name from the pdf ---------------------------
def reportName(path):
    name = path.split('/')[-1]
    company = name.split('.')[0]
    return company


# --------------------------- Sentiment Analysis ---------------------------
# This function calculates the sentiment score for the various sentences using VADER
# Sentence:
#   - The sentence to be inputted to the function, which will return the respective sentiment score
#   - If there are > 1 sentence, the average will be computed and returned
def averagedCompoundSentimentScore(sentences):
    sid = SentimentIntensityAnalyzer()
    score = 0
    for sentence in sentences:
        sentiment = sid.polarity_scores(sentence)
        score += sentiment['compound']
    try:
        return score / len(sentences)
    except ZeroDivisionError:
        return score


# --------------------------- Sentence Extraction ---------------------------
# This function extracts out the keywords from the given corpus
# corpus:
#   - This refers to a document (i.e one company)
# subFeatureKeywords:
#   - A list containing all the keywords which we would like to identify from our sentence bank
def keySentences(corpus, subFeatureKeywords):
    sentencesCaptured = []
    for word in subFeatureKeywords:
        sentencesCaptured.extend(
            [sentence for sentence in corpus if word in sentence])
    return sentencesCaptured


# --------------------------- Print all sentences (Debugging purposes only) ---------------------------
def printAllSentences(corpus, pillar, keywordBank):
    for subFeature, kewords in keywordBank[pillar].items():
        print('\n\n\n')
        print(f"======= Printing Sentences from: '{subFeature}' =======")
        sentences = keySentences(corpus, kewords)
        for sentence in sentences:
            print(sentence)
            print('\n\n')


# --------------------------- Subpillar Feature Statistics ---------------------------
# 4 options for pillar:
#   - 'Environment'
#   - 'Social'
#   - 'Governance'
#   - 'ESG phrases'
# corpus:
#   - A specific company report, and NOT the whole collection of reports from all companies!
# keywordBank:
#   - All the keywords from the subpillar
def subPillar_featureStats(corpus, pillar, keywordBank):
    data = {}

    # Calculate the sentences, frequency of sentence occurence, sentiment score etc
    def summaryStatistics(corpus, subFeatureKeywords):
        temp = {
            # "Sentences": None,
            "NumOfSentences": None,
            "FrequencyOfOccurence": None,
            "SentimentScore": None
        }
        # temp['Sentences'] = subpillar_sentences(corpus, keywordBank[pillar])
        sentences = keySentences(corpus, subFeatureKeywords)
        temp['NumOfSentences'] = len(sentences)
        temp['FrequencyOfOccurence'] = round(len(sentences) / len(corpus), 5)
        temp['SentimentScore'] = averagedCompoundSentimentScore(sentences)
        return temp

    for subFeature, subFeatureKeywords in keywordBank[pillar].items():
        data[subFeature] = summaryStatistics(corpus, subFeatureKeywords)

    return data


# --------------------------- Complete Feature Statistics ---------------------------
# This combines all the data across the 3 pillars into a dictionary
# esg_bank:
#   - Complete set of data processed from reading in all the companies
#   - Structure of esg_bank:
#       - Dictionary where
#           - key: company name
#           - value: [sentence1, sentence2, ..., sentenceN]
# companyName:
#   - The company we wish to explore
# keywordBank:
#   - Complete set of data from the keywords.json file
def featureStats(esg_bank, companyName, keywordBank):
    company = {
        companyName: []
    }
    for pillar in [*keywordBank][:-1]:
        temp = {}
        temp[pillar] = subPillar_featureStats(
            esg_bank[companyName], pillar, keywordBank)
        company[companyName].append(temp)
    return company


def processByLength(esg_bank, keywordBank, numberOfReports):
    print('\n\n === Generating feature statistic data from all companies === \n\n')
    companies = []

    def flatten_data(dictionary_data):
        new_data = {
            "Company Name": list(dictionary_data.keys())[0],
        }
        avgscore = []
        for subData in dictionary_data.values():
            for i in range(0, 3):
                for pillar, pillarValues in subData[i].items():
                    title = f"avg {pillar} Sentiment Score"
                    try:
                        avg = round(sum([data['SentimentScore'] for title, data in pillarValues.items(
                        )]) / len(pillarValues), 5)
                        new_data[title] = avg
                        avgscore.append(avg)
                    except ZeroDivisionError:
                        new_data[title] = 0
                        avgscore.append(0)
        new_data['avg ESG Sentiment'] = round(sum(avgscore) / len(avgscore), 5)
        return new_data

    counter = 0
    for company, data in esg_bank.items():
        if counter == numberOfReports:
            break
        else:
            print(f"Processing data from --- {company}")
            company_data = featureStats(esg_bank, company, keywordBank)
            companies.append(flatten_data(company_data))
            counter += 1

    return pd.DataFrame(companies)


""" # Read our database of ESG reports
E.g run python3 featureExtraction.py "/Users/kevintanyuejun/Desktop/Reports 2.0"
"""
parser = argparse.ArgumentParser()
parser.add_argument('text', action='store', default=None,
                    help='Path to files', type=str)
path = parser.parse_args()
esg_reports = glob.glob(path.text + '/*.pdf')
esg_corpus = {}

counter = 1
for report in esg_reports:
    esg_corpus[reportName(report)] = convert_pdf_into_sentences(
        read_pdf(report))
    print(f"Reading Report {counter}: '{reportName(report)}'")
    counter += 1

# Read the key words from our json file
f = open('keywords.json')
keywordBank = json.load(f)
f.close()

# pprint.pprint(keywordBank)
# pprint.pprint(featureStats(esg_corpus, 'samsung 2021', keywordBank))
# printAllSentences(esg_corpus['Izertis SA 2021'], 'Environment', keywordBank)

esgdata = processByLength(esg_corpus, keywordBank, 30)
print(esgdata)
