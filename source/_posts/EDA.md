---
title: EDA
draft: 'OSNN:'
date: 2025-04-16 00:21:38
tags:
---

## Polyvore Dataset

The Polyvore Dataset's training data is contained in "data/train_no_dup.json"; It is a json file that contains x outfits with each outfit containing about 7-8 outfit items. 

Each outfit item has a name that contains a description of the item, a price, and the category ID which helps to map the item to a generic category of clothing or accessory type. For purpose of understanding the data, we will mainly look at these item name/descriptions and category IDs.

<img src = "https://github.com/hpmavani/hpmavani.github.io/blob/main/images/Json-Ex.png?raw=true" style = "width: 75%">

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

<img src = "https://github.com/hpmavani/hpmavani.github.io/blob/main/images/df1.png?raw=true" style = "width: 75%">

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
3. Find the cartesian product of the outfit items
4. Loop through the pairs and increment the location (i, j) and (j, i) respective to the pair of items by 1. 

Once we have a general co-occurence matrix of the entire vocab set, we can subset this dataframe to find specific patterns in terms of colors, fabrics, and patterns.

For this, I have created 3 text files with an exhaustive list of colors, fabrics, and patterns which can be used filter the dataset and generate new co-occurence matrices. 

The code for each of these parts is below: 

### 1. Random Sampling of Outfits
`code`

    import random 
    random.seed(42)
    num_outfits = 17315
    random_ints = random.sample(range(num_outfits), k = 100)
    outfit = df.loc[random_ints]

### 2. Vocabulary Set
`code`

    import itertools
    outfit_vocab = list(outfit["tokenized_item"])
    items_list = set(sorted(list(itertools.chain(*outfit_vocab))))

### 3. Cartesian Product of Outfit Items
`code`
    # Create a list of lists where each nested list contains combination of clothing items
    # e.g., [['victoria', 'velvet', 'gown'], ['rosie', 'long', 'a-line', 'coat']]

    item_product = []
    for i in outfit_vocab: 
        for j in outfit_vocab: 
            if i != j: 
                item_product.append([i, j])

    # We want to further expand this to pairs of words in those item descriptions

    pairs = []

    for item_combo in item_product: 
        for element in itertools.product(*item_combo): 
            pairs.append(element)
    
    # "Pairs" will have tuples that look like this: 
    # [('victoria', 'rosie'),
        ('victoria', 'long'),
        ('victoria', 'a-line'),
        ('victoria', 'coat'),
        ('velvet', 'rosie'),
        ('velvet', 'long'),
        ('velvet', 'a-line'),
        ('velvet', 'coat'),
        ('gown', 'rosie'),
        ('gown', 'long'),
        ('gown', 'a-line'),
        ('gown', 'coat')...]


### 4. Creating the co-occurrence matrix by incrementing frequencies 
`code`

    # Create a mapping from items to a matrix index
    item_indices = {item: i for i, item in enumerate(items_list)}

    # initialize matrix
    matrix = np.zeros((len(items_list), len(items_list)), dtype=int)

    for item1, item2 in pairs: 
        matrix[item_indices[item1], item_indices[item2]] += 1
        matrix[item_indices[item2], item_indices[item1]] += 1

    cooc_mat = pd.DataFrame(matrix)

After we have this co-occurence matrix of item-pair frequencies, we can extract the rows and columns that are colors, fabrics, and patterns to see the relationships between these categories -- how often two colors, fabrics, or patterns occur together in an outfit.

### Color Analysis 

A colors.txt file contains a list of colors that covers most of the colors found in the data set. We read this file into a python set and then loop through the cooc_mat columns, subsetting the columns that occur in the colors set. This new matrix is then normalized by dividing each cell by a row total. These normalized values give better insight into the distribution of the occurences of other colors given that row's color. This is the basic idea of a frequency table and builds off of the idea of conditional probability. 

To visualize these relationships better, we can view them in a heatmap. The differing cell colors for each row indicate that there is a correlation between colors and their likelihood of co-occuring. 

image here

`code`

    import seaborn as sns
    import matplotlib.pyplot as plt

    mat_colors = [i for i in cooc_mat.columns if i in colors]
    colors_mat = cooc_mat.loc[mat_colors,mat_colors]
    normalized_colors_mat = colors_mat.div(colors_mat.sum(axis=1), axis=0)


    plt.figure(figsize = (10, 5))
    sns.heatmap(normalized_colors_mat, annot=False, cmap="Blues", fmt=".2f", linewidths=0.5)
    plt.title("Color Co-Occurence Matrix Heatmap")
    plt.show()

We can also use an unsupervised technique called PCA to better understand how different colors cluster together when reduced to lower dimensions. PCA is a dimensionality reduction technique. Learn more here. 

image here 

`code`
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(normalized_colors_mat)  # Same length as your data
    colors_legend = plt.cm.tab10(np.linspace(0, 1, len(mat_colors)))

    # Create a color map for categories
    cat_color_map = {cat: color for cat, color in zip(mat_colors, colors_legend)}
    point_colors = [cat_color_map[cat] for cat in mat_colors]

    plt.scatter(reduced[:, 0], reduced[:, 1], c=point_colors)

    for cat in mat_colors: 
        plt.scatter([], [], c=cat_color_map[cat], label=cat)
        
    plt.legend(title = "Colors", fontsize = "x-small", loc = "lower left")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("PCA of Colors")
    plt.show()

### Fabric/Texture Analysis

`code`
    mat_fabrics = [i for i in cooc_mat.columns if i in fabrics]
    fabrics_mat = cooc_mat.loc[mat_fabrics, mat_fabrics]
    normalized_fabrics_mat = fabrics_mat.div(fabrics_mat.sum(axis=1), axis=0)
    plt.figure(figsize = (10, 5))
    sns.heatmap(normalized_fabrics_mat, annot=False, cmap="Blues", fmt=".2f", linewidths=0.5)
    plt.title("Fabric Co-Occurence Matrix Heatmap")
    plt.show()

`code`

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(normalized_fabrics_mat)  # Same length as your data
    colors_legend = plt.cm.tab10(np.linspace(0, 1, len(mat_fabrics)))

    # Create a color map for categories
    cat_color_map = {cat: color for cat, color in zip(mat_fabrics, colors_legend)}
    point_colors = [cat_color_map[cat] for cat in mat_fabrics]

    plt.scatter(reduced[:, 0], reduced[:, 1], c=point_colors)

    for cat in mat_fabrics: 
        plt.scatter([], [], c=cat_color_map[cat], label=cat)
        
    plt.legend(title = "Fabrics", fontsize = "x-small", loc = "lower left")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("PCA of Fabrics")
    plt.show()

### Patterns Analysis 

image here 

`code`

    mat_patterns = [i for i in cooc_mat.columns if i in patterns]
    patterns_mat = cooc_mat.loc[mat_patterns, mat_patterns]
    normalized_patterns_mat = patterns_mat.div(patterns_mat.sum(axis=1), axis=0)
    plt.figure(figsize = (10, 5))
    sns.heatmap(normalized_patterns_mat, annot=False, cmap="Blues", fmt=".2f", linewidths=0.5)
    plt.title("Pattern Co-Occurence Matrix Heatmap")
    plt.xlabel("Items")
    plt.show()

image here 

`code`

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(normalized_patterns_mat)  # Same length as your data
    colors_legend = plt.cm.tab10(np.linspace(0, 1, len(mat_patterns)))

    # Create a color map for categories
    cat_color_map = {cat: color for cat, color in zip(mat_patterns, colors_legend)}
    point_colors = [cat_color_map[cat] for cat in mat_patterns]

    plt.scatter(reduced[:, 0], reduced[:, 1], c=point_colors)

    for cat in mat_patterns: 
        plt.scatter([], [], c=cat_color_map[cat], label=cat)
        
    plt.legend(title = "Patterns", fontsize = "x-small", loc = "lower left")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("PCA of patterns matrix")
    plt.show()