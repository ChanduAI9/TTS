import preprocessor as p
import re
import wordninja
import csv
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from utils import augment


# Data Loading
def load_data(filename):
    """
    Load and process data from a CSV file.

    Args:
        filename (str): Path to the dataset file.

    Returns:
        pd.DataFrame: Processed DataFrame with cleaned and combined data.
    """
    concat_text = pd.DataFrame()

    # Load individual columns
    raw_text = pd.read_csv(filename, usecols=[0], encoding='ISO-8859-1')
    raw_target = pd.read_csv(filename, usecols=[1], encoding='ISO-8859-1')
    raw_label = pd.read_csv(filename, usecols=[2], encoding='ISO-8859-1')
    seen = pd.read_csv(filename, usecols=[3], encoding='ISO-8859-1')

    # Replace string labels with numeric values
    label = raw_label.replace(['AGAINST', 'FAVOR', 'NONE'], [0, 1, 2])

    # Combine all columns into a single DataFrame
    concat_text = pd.concat([raw_text, label, raw_target, seen], axis=1)
    concat_text.columns = ['Tweet', 'Stance', 'Target', 'Seen']

    # Remove rows with 'Seen' labels for non-training datasets
    if 'train' not in filename:
        concat_text = concat_text[concat_text['Seen'] != 1]

    return concat_text


# Data Cleaning
def data_clean(strings, norm_dict):
    """
    Clean and normalize individual strings using a normalization dictionary.

    Args:
        strings (str): Input text string to clean.
        norm_dict (dict): Dictionary for normalizing slang and abbreviations.

    Returns:
        list: List of cleaned and normalized tokens.
    """
    # Remove URLs
    clean_data = re.sub(r'http\S+', '', strings)

    # Remove emojis (you can add more Unicode ranges for additional emojis)
    clean_data = re.sub(r'[😀-🙏]', '', clean_data)

    # Extract tokens: words, hashtags, mentions, punctuation, and numbers
    clean_data = re.findall(r"[A-Za-z#@]+|[,.!?&/\\<>=$]|[0-9]+", clean_data)

    # Convert to lowercase and normalize using norm_dict
    clean_data = [[token.lower()] for token in clean_data]
    for i in range(len(clean_data)):
        if clean_data[i][0] in norm_dict:
            clean_data[i] = norm_dict[clean_data[i][0]].split()

    return clean_data


# Clean All Data
def clean_all(filename, norm_dict):
    """
    Clean and process the entire dataset file.

    Args:
        filename (str): Path to the dataset file.
        norm_dict (dict): Dictionary for normalizing slang and abbreviations.

    Returns:
        tuple: A tuple containing cleaned tweets, labels, and targets.
    """
    # Load all data as a DataFrame
    concat_text = load_data(filename)

    # Extract required columns
    raw_data = concat_text['Tweet'].values.tolist()  # Tweets as a list of strings
    label = concat_text['Stance'].values.tolist()  # Stances (labels) as a list
    x_target = concat_text['Target'].values.tolist()  # Targets as a list of strings

    # Initialize containers for cleaned data
    clean_data = [None for _ in range(len(raw_data))]

    # Clean tweets and targets
    for i in range(len(raw_data)):
        clean_data[i] = data_clean(raw_data[i], norm_dict)  # Clean each tweet
        x_target[i] = data_clean(x_target[i], norm_dict)  # Clean each target

    # Compute average length of cleaned tweets
    avg_length = sum([len(x) for x in clean_data]) / len(clean_data)

    # Log statistics
    print("Average tweet length: ", avg_length)
    print("Number of samples: ", len(label))

    return clean_data, label, x_target
