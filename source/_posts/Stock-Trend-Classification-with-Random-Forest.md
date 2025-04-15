---
title: Stock Trend Classification with Random Forest
date: 2025-04-01 23:46:38
tags:
---

## Overview 
Many individuals aim to invest their money in stocks and take advantage of this quick but also risky way of making money. The movement of stock prices are impacted by many factors such as crowd sentiment, world news, and macroeconomic and microeconomic movement. Stock analysis is divided into two main categories: fundamental and technical analysis. Fundamental analysis focuses on broader analysis of a company’s financial records and market capitalization, while technical analysis focuses on creating an almost scientific method of stock prediction that’s based on patterns, trends, and momentum of a stock’s price. Machine learning techniques can be used in identifying relevant patterns within indicators and signals while filtering out noise. This kind of classification offers investors a buy or sell signal from the technical perspective of stock prediction, which they can verify with real-world sentiments to help make decisions about entry and exit points for investments. In this exploration, I will detail how Random Forest Classification can be applied to prediction of financial market conditions to make well-informed investment decisions.

## Random Forest Classification
### Why Random Forest for Stock Trend Classification?
Random Forest is a machine learning technique that can be used for regression and classification tasks and is considered an extension of bagging. It incorporates an ensemble of decision trees that enables it to be accurate with large datasets and robust to overfitting.

Random Forest Classification is an intuitive choice for stock bullish/bearish analysis because the strategies that many traders use can be represented as a decision tree. When analysts make decisions about a security, they filter through different signals and indicators, making often binary decisions about which technical or market conditions indicate a bullish or bearish move for a stock, before coming to a final conclusion (buy or sell). For example, a trader might first look at the RSI indicator and see that it has dipped below the 30-line indicating that it is oversold. They may also see that the ADX is greater than 15 and DMI+ is above the DMI- line, indicating that there is an uptrend forming. After looking at several other indicators, they come to a conclusion of a certain confidence that there is a strong chance that the stock will go up. 

Decision trees capture these decision points in a quantitative way, determining the best way to split each decision. If we go one step further, we can imagine that evaluating the opinion of 100 traders each with their own conclusion would result in a more reliable estimation of the "correct" classification of a stock. The intent behind Random Forest is exactly this, where 100 different decision trees each produce an outcome and the final prediction is determined on a majority vote basis.

Before discussing the process of using Random Forest for the purpose of Stock Trend Classification, it's crucial to understand the basic functionality of the Random Forest algorithm. 

### Decision Trees (Node Purity & Tree Independence)
Before, I briefly went over why Random Forest is an intuitive choice for stock trend classification using the following example: 
"...a trader might first look at the RSI indicator and see that it has dipped below the 30-line indicating that it is oversold. They may also see that the ADX is greater than 15 and DMI+ is above the DMI- line, indicating that there is an uptrend forming. After looking at several other indicators, they come to a conclusion of a certain confidence that there is a strong chance that the stock will go up."

Notice how this line of reasoning can be easily fit into a (overly simplifed) decision tree. An even more complex line of reasoning can be represented this way.

![image](/images/Decision_Tree.png)

Additionally, the numerical splits 30, 15, and 0 in the image above are not arbitrarily assigned but are selected after maximizing the node purity at each split. Each point of decision in a decision tree aims to make every node as homogenous as possible (data points of mostly one label). Intuitively, it is deciding the factors that contribute to class labels being the most different. There are many algorithms to determine the best split, but in this article, I will focus on the Gini Index, which is also the algorithm used in the stock trend classification model. 

![image](https://github.com/user-attachments/assets/d6ffefaa-3aee-400f-a1d6-3cb0750a7f8a)

The Gini Index is a measure of impurity (non-homogeneity of nodes). It ranges from 0 to 1, with 0 representing a pure dataset and 1 representing an impure dataset. A decision tree aims to minimize the gini index at each decision point until it reaches its maximum depth. <br>

![image](https://github.com/user-attachments/assets/c6ad290f-7ae4-4aa3-bd5f-32e6050c415f)

p(j) represents the probability of classifying as a class j, so adding these probabilities and subtracting from 1 give the probability of misclassifying a datapoint when it is randomly assigned to a class. You can extend this to feature selection in a decision tree. If we take appropriate test values (a, b, c, ..., z) for features X, Y, Z, we can calculate P(X = a), P(X = b), ..., for each feature and calculate the Gini Impurity for each value. If feature Y has the smallest Gini Impurity, we know that under random selection of a datapoint, it has the smallest chance of misclassification, making it the best feature for the decision tree. 

Another important advantage of using Random Forest over just one decision tree besides having more than one "opinion" on a classification is how each tree is independent to one another because of feature subsetting and bagging. The decorrelation of trees is imperative in terms of avoid overfitting, so the model doesn't memorize the noise in the data.

### Feature Subsetting
Each tree in the Random Forest gets the full set of features available, but if it just went by the best feature to split by at every node, we would end up with a forest filled with identical decision trees. This isn't very useful! To address this, Random Forest takes the best feature among a random subset of features while splitting a node. For example, if we had features {RSI, MACD, ADX, Volume} and the sample size of the subset was 2, at split 1, we might get the subset {RSI, ADX} and determine that the best feature to split by is the RSI. At the subsequent split, we take another subset of the original set of features with replacement, {RSI, Volume} and repeat the process. The number of features sampled at each node is denoted by "m" and usually set to the square root of the total number of features n for classification tasks. However, this is a hyperparameter that can be tuned to see what best fits each model. Feature subsetting ensures that each tree is diverse and not dominated by a select few stronger predictors.

### Bagging 
Bagging is used to further decorrelate each tree by taking a random bootstrapped sample of data points to train each tree. Bootstrapping means that the original set of data points is sampled with replacement so each tree may be trained on repeated points. Random Forest avoid overfitting through this means of shuffling the data. 

This raises an important question of how Random Forest can be used with time series data if the training process involves random sampling from the data. Wouldn't the sequentiality of the data be lost? The answer is yes if you used random forest on raw time series data, sequentiality would be lost and look-ahead bias would be reflected in the results. This is why the original features of the dataset are engineered into lagged features. This way time dependence is encoded into each of the features and offsets the bootstrapping issue.

### Disadvantages of Random Forest
Random Forest is computationally expensive. The more estimators (number of trees), the longer it takes for the algorithm to run, especially if each tree maintains its max depth. This can potentially be avoided by reducing how deep each tree can go and adjusting the number of estimators, but there's usually a tradeoff between computation and accuracy.

Random Forest is also black box. It's not feasible to see the different branches of each of 100s of decision trees, so a lot of this information isn't shown in detail while working with the Random Forest model. This makes it harder to tune and adjust the model. However, feature importance scores give more insight into the features that contributed the most to the algorithm's accuracy.

Now that we've covered the basics of how Random Forest works, I can discuss the specifics of this implementation.

## Dataset
The dataset used to train and test the model was sourced from Python's YFinance library with historical and most recent data on open, high, low, close, volume, and adjusted close prices for the S&P500 index to capture patterns consistent with the top 500 traded stocks in the United States. This included 9909 rows and 7 columns. 

The training data is labeled with a 5-day lagged close price to simulate conditions for the moderately active home investor. If P(i) represents the most recent close on the i-th day then:

P(i) - P([i-5]) > 0 => label -> 1 indicates that the price has increased within the past 5 days. <br>
P(i) - P([i-5]) <= 0 => label -> 0 indicates that the price has decreased within the past 5 days.

Using this 5-day lag also helps the model filter out noise in the data that occurs from a 1-day lag close price.

The engineered features in consideration are several different indicators such as RSI, MACD, ADX, SMAs, and Bollinger Bands. The main algorithm used was Random Forest, which is one of the industry standards for classification in terms of accuracy and prevention of overfitting. This project makes use of many different data science skills such as wrangling and pre-processing, feature engineering, data exploration through plots and correlation matrices, and model analysis using metrics such as accuracy, precision, specificity, sensitivity, ROC curve, f1-score, and K-Fold cross-validation. 

## Features

### Simple Moving Average & Bollinger Bands
The Simple Moving Average (SMA) is calculated with the formula (n is typically 14): 

![image](https://github.com/user-attachments/assets/8e575462-b063-45da-b699-f826593cef35)

Where p(i) indicates the close-price on the i-th day of the n-day period. 
This indicates the averaged price movement of a stock and it’s particularly useful as a measure of central tendency. For instance, with the SMA, a Bollinger band can be created to mark out the approximate trading range of a stock’s prices. By taking 2 standard deviations above and below the SMA, a region of price variety is created, and a stock price that travels outside of the region – above the upper boundary of the Bollinger band or below the lower boundary of the Bollinger band – indicates an unusual fluctuation that can signal that it’s due for a correction that brings the price back up or down. 

### RSI
The RSI or the relative strength index is a momentum indicator that is calculated with the formula: <br>

![image](https://github.com/user-attachments/assets/4c62e65e-50d9-4a07-bfcb-9eeee716e8a0)


The average gain and loss are calculated over an n-day period, where n is usually 14 days. 
The concept behind the RSI is that every stock has a trading range, and when a stock price reaches the bottom of its range, there will be a support force that pushes the stock price back. Contrarily, when the stock price reaches the top of its range, there’s a resistance force that pulls the stock price back down. The resistance level and support level of a stock’s trading range are measured by an RSI index of either 70 and 30 or 80 and 20 respectively. If a stock price touches the 70/80 mark, it indicates that the stock is overbought, and market forces will naturally bring the stock back down (trend reversal). If a stock price touches the 30/20 mark, it indicates that the stock is oversold, and market forces will naturally bring the stock back up. In other words, the RSI measures the likelihood of a trend reversal in context of the stock's current price momentum. 

Instead of using static values such as 70/30 or 80/20, using a z-score for the RSI with rolling means and rolling standard deviations creates dynamic markers for when a stock price is oversold/overbought.

### MACD
MACD or the Moving Average Convergence Divergence Oscillator is another popular indicator used in technical analysis of the stock market. Essentially, it measures the momentum (the speed and direction) of price movements. The MACD indicator has two parts: the MACD and the signal. 
The MACD value is calculated by subtracting the 12-Period and 26-Period EMA (exponential moving average) of the historical close prices. The former is a short-term moving average, and the latter is a long-term moving average, so the difference of the two provides more insight on the direction of the market. The exponential moving average is like the simple moving average, but it places more weight on the more recent data points in the timeseries data. 

MACD = EMA (12) – EMA (26)

![image](https://github.com/user-attachments/assets/e5e90025-97e5-4146-89f9-f9b4784315e6)

The signal value is the 9-Period simple-moving average of the MACD line, and it’s used as a standard of comparison for the MACD. 

The main points of interest with the MACD indicator are crossover points, where the MACD and Signal lines cross over each other, as well as the magnitude of the momentum at those points. Generally, when the MACD line is above the Signal line, this is a buy signal, and when the MACD line is below the Signal line, this is a sell signal. 

### ADX, DMI+, DMI-, DI

Some indicators that are used in conjunction with MACD are the average directional index (ADX), directional movement index plus and directional movement index minus. ADX identifies the existence of a stock trend, however it only provides the magnitude of the strength of a trend and not the direction. The directional movement (positive line DMI+) and directional movement (negative line DMI-) provide information about the direction of the trend. The overall directional index (DI) is calculated by taking the difference between the DMI+ and DMI-. When the DMI+ is above DMI-, the price is moving up. Conversely, when the DMI+ is below the DMI-, the price is moving down. 
The ADX can be evaluated with these notions: 
* 0-20	Absent or Weak Trend
* 25-50	Strong Trend
* 50-75	Very Strong Trend
* 75-100	Extremely Strong Trend

For example, a positive directional index (DMI+ > DMI-) and an ADX value of 45 indicates a strong uptrend. A negative directional index (DMI+ < DMI-) and an ADX value of 15 indicates a weak downtrend. 
In this stock trend classification, the ADX and directional index are represented in one feature by multiplying the ADX by -1 if the directional index is negative (the DMI+ < DMI-); the ADX is kept positive if the directional index is positive (the DMI+ > DMI-). 

## Results

### Confusion Matrix
Confusion matrices summarize the performance of a classification algorithm by showing the outcomes of predicting each class. In this situation, the positive class is an uptrend (indicating a stock is predicted to rise). The negative class is a downtrend (indicating a stock is predicted to fall). 

* True Negatives = This is the number of correct downtrend predictions, where the algorithm correctly identified that the stock price would fall. (741)
* True Positives = This is the number of correct uptrend predictions, where the algorithm correctly identified that the stock price would rise. (936)
  
* False Positives = This is the number of incorrect uptrend predictions. (112)
* False Negatives = This is the number of incorrect downtrend predictions. (184)

![image](https://github.com/user-attachments/assets/39691d77-789e-4822-acf4-4fd074375f97)

### Accuracy, F1-Score, Precision, and Recall 
The results of the Random Forest model's performance were evaluated using accuracy, f1-score, precision, and recall. These are directly calculated from the outcomes of the confusion matrix above.

![image](https://github.com/user-attachments/assets/9d8aae84-e9a6-41b7-b9b7-8f0f2aa84407)

### Cross-Validation and Hyperparameter Tuning 
The cv (Cross-Validation) score is also considered, which indicates a mean accuracy of 86.4% indicating that there likely isn't much overfitting. However, we can try to increase this score further through hyperparameter tuning using gridsearchCV, specifically with the number of estimators (number of trees) and the maximum amount of features to select from at each split. 

For the number of estimators, the best number of estimators is 880 estimators. However, this specific finding doesn't change the cv score by a signficant amount. 

![image](https://github.com/user-attachments/assets/f35d4035-73fe-4661-a07a-792f8d1392ca)

For the number of max features, the best parameter is found to be 8 features. This means that at every split, a subset of 8 features should be taken. 

![image](https://github.com/user-attachments/assets/d1dfc06a-bcb8-46c3-a455-e17faa4ffea4)

Re-fitting the model with the tuned parameters (880 estimators and max features of 8) gives a new cv score of 86.7%, a 0.3% from the original cv score of 86.4. This different isn't very significant, so the time trade-off that would be required for training the random forest with 880 estimators isn't worth the small increase in cv. 

### Feature Importances 
The feature importances reveal that RSI Z-Score, Z-Score, RSI, and ADX contributed the most to the model's accuracy, while Bollinger Bands lower bound, SMA, and Bollinger Bands upper bound contributed the least. This also means that the most prominent features reduced the Gini Index (promoted node purity) relatively more than the other features. 

<p align="middle">
  <img src="https://github.com/user-attachments/assets/d41328fd-ce1c-48ce-a9c2-e0eed87a00d2">
  <img src="https://github.com/user-attachments/assets/5a853d97-e6d6-404c-8584-5335ae0a9ed0" width="600px">
</p>

## Conclusion
The overall performance of the Random Forest Classification is ideal with a cv score of about 82.5%. Compared to other algorithmic models such as Support Vector Machines that can reach accuracies of up to 90%, Random Forest performs adequately well considering that it's highly interpetable and takes less time to train. 

As the model was trained on S&P500 data, the feature importance reflects that RSI and ADX features are highly indicative of stock market patterns and trends when generalized to a diverse range of stocks from different sectors. This makes intuitive sense because RSI is primarily a momentum indicator, providing information on the direction and extent of price movement, and ADX provides information on the quantified strength of a trend in any direction. In practice, these two indicators are often used in conjunction to verify potentially true signals or identify false signals. However, it is surprising that MACD features didn't contribute more to the overall accuracy of the model considering its popularity as a technical indicator. With even more nuanced feature engineering, these results could definitely evolve and better the model's performance.