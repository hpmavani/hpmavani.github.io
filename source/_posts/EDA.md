---
title: EDA
draft: 'OSNN:'
date: 2025-04-16 00:21:38
tags:
---

## Polyvore Dataset

The Polyvore Dataset's training data is contained in "data/train_no_dup.json"; It is a json file that contains x outfits with each outfit containing about 7-8 outfit items. 

Each outfit item has a name that contains a description of the item, a price, and the category ID which helps to map the item to a generic category of clothing or accessory type. For purpose of understanding the data, we will mainly look at these item name/descriptions and category IDs.

<img src = "" style = "width: 75%">

For this type of data, pandas multi-indexing will help to maintain the hierarchical nature between outfits and outfit items: 

`code` 

    import pandas as pd
    import numpy as np
    import json
    pd.options.mode.chained_assignment = None
    with open('data/train_no_dup.json') as file: 
        data = json.load(file)

    iterables = []
    item_prices = []
    item_names = [] 
    item_catids = []

    for index, d in enumerate(data): 
        items = [i for i in d["items"]]
        
        for i_index, i in enumerate(items): 
            iterables.append([index, i_index])
            item_names.append(i["name"])
            item_prices.append(i["price"])
            item_catids.append(i["categoryid"])

    multindex = pd.DataFrame(iterables, columns=["outfit_index", "item_index"])
    index = pd.MultiIndex.from_frame(multindex)

    df = pd.DataFrame({"item_name": item_names, "item_price": item_prices, "item_catid": item_catids}, index=index)

    df.head(25)

## Generating a Word Cloud of Outfit Items
A word cloud is a tool that gives a qualitative idea of the most frequent words found in the data. 

Before we generate a word cloud, the outfit items must be preprocessed to remove any empty strings or irrelevant words in the item descriptions. All words are made lowercase and punctuation is also removed. 

`code`

    import re 
    from wordcloud import STOPWORDS
    stopwords = set(STOPWORDS)
    text = ' '.join(df['item_name'].astype(str).tolist())
    text = re.sub(r"[^A-Za-z\s]", '', text)
    text = text.lower() 
    text = ' '.join(word for word in text.split() if word not in stopwords)
    df.replace(["", " ", "...", "Polyvore", ''], np.nan, inplace=True)
    df["item_name"] = df["item_name"].str.lower()
    df["item_name"] = df["item_name"].str.replace("t shirt", "t-shirt")
    df["item_name"] = df["item_name"].str.replace(r"[^A-Za-z\s\-]", "", regex=True)
    df = df.dropna()
    df[df["item_name"] == ""]["item_name"] = np.nan

`code`

    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Item Names Word Cloud')
    plt.show()

<br>

At glance, the most common words are women, black, shoulder bag, white, and ankle boot. The most common colors seem to be black, white, brown, silver, and gold. This implies that the training data lacks data on more vibrant clothing.

There's also a lot of brand names: dolce gabbana, micheal kors, saint laurent, miu miu, and jimmy choo. This reflects that the Polyvore dataset consists of higher-end products. 

## Distribution of Items Across CategoryID


## Co-Occurence Matrix Analysis 

A co-occurence matrix is basically a frequency table, but the rows and columns contain the same features. In NLP, this matrix is used to calculate how often pairs of words show up together, which offer insight into the similarity between words. 

For this specific application, we use co-occurence matrices to see how often two words show up together in an outfit. For example, how often do we see the word "black" and "gold" in one outfit? This could indicate that since black and gold are compatible colors, they show up together often. 

This helps to gain insight on word relationships across outfits, and we will analyze this more closely in terms of colors, fabrics, and patterns that co-occur in the training data. 

To accomplish this, we will follow these steps: 

1. Randomly sample n outfits from the training dataset
2. Create a vocab set that contains all words found in clothing item names.
3. Find the cartesian product of 
4. Loop through the pairs and increment the location (i, j) and (j, i) respective to the pair of items by 1. 

Once we have a general co-occurence matrix of the entire vocab set, we can subset this dataframe to find specific patterns in terms of colors, fabrics, and patterns.

For this, I have created 3 text files with an exhaustive list of colors, fabrics, and patterns which can be used filter the dataset and generate new co-occurence matrices. 
