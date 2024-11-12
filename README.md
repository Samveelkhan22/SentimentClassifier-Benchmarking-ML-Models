# SentimentClassifier-Benchmarking-ML-Models
In this project, we aim to analyze customer reviews to predict the sentiment of each review—whether it is positive or negative using text based features and associated numerical ratings. These reviews can help businesses gauge customer satisfaction and improve their products or services. The problem involves applying machine learning techniques to classify textual data (customer reviews) based on sentiment labels derived from the review ratings.

## Dataset Description

The dataset used in this project consists of customer reviews, each containing a piece of text along with a numerical rating. This rating represents the customer's opinion of the product or service and is used as an indicator of sentiment. The dataset is structured with two primary columns:

•	review_text: The actual textual content of the review.
•	customer_review_rating: A numerical value, typically on a scale of 1 to 5, indicating the sentiment. For this project, a rating of 3 or 
higher indicates positive sentiment, and a rating below 3 is classified as negative sentiment.

The dataset contains thousands of reviews, which provides a large volume of data for building a robust sentiment classification model.

## Problem Statement

The primary objective of this project is to develop a sentiment classification model that can predict the sentiment of a customer review based on its textual content and rating. The project will explore several machine learning models and evaluate their performance in classifying reviews into two categories:

•	Positive: Reviews with ratings of 3 or higher.
•	Negative: Reviews with ratings below 3.

By the end of the project, we aim to identify the best-performing model, demonstrate its ability to predict sentiment accurately, and discuss potential future improvements.

## Models and Methodology

### Preprocessing

Text preprocessing techniques include:
- Tokenization
- Lowercasing
- Removal of stop words
- TF-IDF vectorization

### Models

1. **Logistic Regression**: Optimized with grid search for parameters `C=10`, `max_iter=1000`, and `solver=saga`.
2. **Naive Bayes**: With optimal smoothing parameter `alpha=0.5`.
3. **Support Vector Machine (SVM)**: Trained with linear kernel for sentiment classification.

## Results

### Logistic Regression

- Best Parameters: {'C': 10, 'max_iter': 1000, 'solver': 'saga'}
- Performance:
- Accuracy: 63%
- F1-Score (Positive): 0.73

### Naive Bayes

- Best Parameters: {'alpha': 0.5}
- Performance:
- Accuracy: 61%
- F1-Score (Positive): 0.70

### SVM

- Performance:
- Accuracy: 62%
- F1-Score (Positive): 0.71

## Comparison of Models

The results indicate that Logistic Regression achieved the highest F1-score (0.73) for the positive class, followed by the SVM model (0.71). Naive Bayes had a slightly lower performance (0.70 for the positive class) but performed relatively well, especially in terms of recall for negative sentiment (0.59).

In terms of precision, Naive Bayes showed the highest performance for the positive class (0.79), whereas Logistic Regression and SVM both performed similarly (0.77).

## Model Summary:

•	Logistic Regression:

o	Best for balanced precision and recall for the positive class.

o	Overall accuracy: 63%.

•	Naive Bayes:

o	Best for precision for positive sentiment, but lower recall.

o	Overall accuracy: 61%.

•	SVM:

o	Best recall for the positive class among all models, with a trade-off in precision.

o	Overall accuracy: 62%.

## Future Work

Future efforts could focus on improving the recall of all models, especially for the negative sentiment class, which was a weakness for most of the models. Additionally, integrating advanced techniques like LSTM and BERT models for deeper contextual understanding might improve performance across all metrics, especially in challenging sentiment scenarios like irony or sarcasm.


