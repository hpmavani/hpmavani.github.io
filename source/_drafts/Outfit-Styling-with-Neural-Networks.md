---
title: Outfit Styling with Neural Networks
tags:
---

## Project Motivation 

Outfit styling, although a creative process, has many different rules and patterns that can be automated using a neural network. Generating a compatible outfit can be boiled down to a next-token prediction task, where given a set of some clothing items, we can ask, "What's the next clothing item that will maximize this outfit's compatability?" 

The ideal vision for this project is that the model is able to pick up on the subtleties of color, pattern, and texture combinations as well as other implicit features that determine outfit compatability and use this to predict the next "token." 

To achieve this, the model will be fit with a training dataset of about 18,000 compatible outfits with over 100,000 clothing items. Each clothing item will have a textual description, a category ID, and price. However, for simplicity, we will only consider the textual description. With the diverse array of clothing item descriptions, we can make a massive vocabulary for our model with tokenized clothing items. The tokens will be embedded using embedding techniques and inputted to the GPT-2 base model. 

Over the course of several blog posts, I will detail the process of training a model that is capable of generating compatible outfits. This blog series will cover many different aspects of the ML pipeline such as: 

1. Exploratory Data Analysis (EDA)
    * This will involve using sophisticated techniques to identify qualitative patterns in the data as well as visualizing these patterns using charts and word clouds.
2. Data Preprocessing 
    * Cleaning textual data by removing punctuation and stop words (irrelevant words)
    * Removing null values
    * Creating test and validation sets that fit this task. 
3. Model training
    * Fitting the training data to a base model
    * Evaluating base-line performance
4. Hypertuning & Feature Engineering 
    * Revisiting the training data if there is "bad data"
    * Engineering new features to improve the model's performance
    * Tuning parameters such as learning rates, activation functions as well as utilizing regularization
5. Final Evaluation on Test Set
    * Analysis of error and results 
    * Future improvements

![next-token-flow](hpmavani.github.io\images\next-token-flow.png)