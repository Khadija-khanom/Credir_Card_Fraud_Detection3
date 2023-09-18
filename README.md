# Comparative Analysis of Machine Learning and Deep Learning Algorithms for Credit Card Fraud Detection on 3rd Dataset
![image](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/assets/138976722/49370c87-61f4-4bbf-b543-9bbdaea5589e)

**Abstract Data Set for Credit Card Fraud Detection**
This dataset contains information related to credit card transactions, with a focus on fraud detection. The dataset has a total of 12 columns, each providing different attributes associated with credit card transactions. The goal of this dataset is to develop a predictive model that can accurately identify fraudulent credit card transactions based on these attributes.

**Dataset Columns:**

**- Merchant_id:** An identifier for the merchant involved in the transaction.

**- Transaction date:** The date of the transaction. (Note: All entries in this column are missing values, so this column might not be useful for analysis.)

**- Average Amount/transaction/day:** The average amount of transactions made per day.

**- Transaction_amount:** The amount of the transaction.

**- Is declined:** Indicates whether the transaction was declined.

**- Total Number of declines/day:** The total number of transaction declines per day.

**- isForeignTransaction:** Indicates whether the transaction is a foreign transaction (made in a different country).

**- isHighRiskCountry:** Indicates whether the transaction involves a high-risk country.

**- Daily_chargeback_avg_amt:** The average amount of chargebacks per day.

**- 6_month_avg_chbk_amt:** The average chargeback amount over the last six months.

**- 6-month_chbk_freq:** The frequency of chargebacks over the last six months.

**- isFradulent:** The target variable, indicating whether the transaction is fraudulent or not.

**Data Characteristics:** 
• The dataset contains a total of 3075 entries (rows).

• Most columns have non-null values, except for the "Transaction date" column.

• The data types in the dataset include integers (int64), floating-point numbers (float64), and categorical values (object).

Table of Contents
=================

[Data Visualization](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/blob/main/README.md#data-visualization)
[Implementation process Of Machine learning models](https://github.com/Khadijakhanom/Credir_Card_Fraud_Detection3/blob/main/README.md#implementation-process-of-machine-learning-models)

[Evaluating the performance of machine learning models](https://github.com/Khadijakhanom/Credir_Card_Fraud_Detection3/blob/main/README.md#evaluating-the-performance-of-machine-learning-models)
 * [Random forest Model](https://github.com/Khadijakhanom/Credir_Card_Fraud_Detection3/blob/main/README.md#random-forest-model)
 * [Decision Tree Model](https://github.com/Khadijakhanom/Credir_Card_Fraud_Detection3/blob/main/README.md#decision-tree-model)
 * [Logistic Regression Model](https://github.com/Khadijakhanom/Credir_Card_Fraud_Detection3/blob/main/README.md#logistic-regression-model)
 * [K-Nearest Neighbors (KNN) Model](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/blob/main/README.md#k-nearest-neighbors-knn-model)
[Implementation process Of deep learning models](https://github.com/Khadijakhanom/Credir_Card_Fraud_Detection3/blob/main/README.md#implementation-process-of-deep-learning-models)
 * [Convolutional Neural Network (CNN) Model]https://github.com/Khadijakhanom/Credir_Card_Fraud_Detection3/blob/main/README.md#convolutional-neural-network-cnn-model
 * [Recurrent Neural Network (RNN) Model]( https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/blob/main/README.md#recurrent-neural-network-rnn-model)
[Evaluating the performance of Deep learning models]( https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/blob/main/README.md#evaluating-the-performance-of-deep-learning-models)
 * [Convolutional Neural Network (CNN) Model]( https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/blob/main/README.md#convolutional-neural-network-cnn-model-1)
 * [Recurrent Neural Network (RNN) Model]( https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/blob/main/README.md#recurrent-neural-network-rnn-model-1)
[Comparative analysis between machine learning and deep learning models]( https://github.com/Khadijakhanom/Credir_Card_Fraud_Detection3/blob/main/README.md#comparative-analysis-between-machine-learning-and-deep-learning-models)



#  Data Visualization

**Visualisation of fradulent and legitimate data**
![image](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/assets/138976722/f8376655-3119-44f8-97e4-cd516705d5af)

**correlation Matrix**

![image](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/assets/138976722/e322ab74-b884-45bf-b044-6aec0c426b36)

**% of fraud transaction declined**

![image](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/assets/138976722/cfd9407d-8ae3-4107-9e4e-51b5e9d2b88a)

**Distribution of the target variable**

![image](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/assets/138976722/d5bb1cea-563f-4dca-bdc0-922e860c27fc)

**Boxplot of Transaction Amount by Fraudulence**

![image](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/assets/138976722/77e3d9c6-acbc-4486-b8fe-c2d3618e147c)

**Count of Total Number of Declines by Fraudulence**

![image](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/assets/138976722/e48b4190-44f5-486c-8ab3-6710b2e5cfe5)

![image](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/assets/138976722/5ab9a82d-4838-45b4-84e5-b3860d823e39)

![image](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/assets/138976722/c6ca1455-7e18-415d-9190-8c02b7166eb3)

**Distribution**

![image](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/assets/138976722/5c538629-dfe4-438c-a1f9-ac9d2bd48bd3)

![image](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/assets/138976722/a5bc72f7-f357-4eb7-b250-59d58b0dc5ac)

![image](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/assets/138976722/e2660352-9aba-4b72-bba0-29ab8d984582)

![image](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/assets/138976722/d1c4f714-c9a0-4b0a-83e7-20c0d87e84a3)

**Scatter Plot of Average Transaction Amount vs. 6-month Avg Chargeback Amount**

![image](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/assets/138976722/88948a2e-cd48-4957-b805-7ed632e840a6)

# Implementation process Of Machine learning models

Demonstrating the process of building and evaluating machine learning models, including Random Forest, Decision Tree, Logistic Regression, and K-Nearest Neighbors (KNN). Here's an overall idea of how these models are built:

**Data Preparation:**

- The dataset is loaded and features (X) are separated from the target variable (y).
- The dataset is split into training and testing sets using train_test_split.

**Balancing the Dataset (SMOTE):**

To address class imbalance, the training data is resampled using SMOTE, creating balanced training data (X_train_resampled, y_train_resampled).

**Hyperparameter Tuning using GridSearchCV:**

- For each model (Random Forest, Decision Tree, Logistic Regression, KNN), a parameter grid is defined to search through different hyperparameters.
- A classifier object (rf_classifier, dt_classifier, lr_classifier, knn_classifier) is created with specified settings.
- An instance of GridSearchCV is created for each classifier, with the specified parameter grid, cross-validation, and accuracy as the scoring metric.
- The models are fitted to the resampled training data using fit.

**Best Model Estimation:**

- The best hyperparameters are determined by the grid search for each model.
- The best estimator (best_rf_classifier, best_dt_classifier, best_lr_classifier, best_knn_classifier) is obtained.

**Model Evaluation:**

- For each model, predictions are made on the test data using the best estimator.
- Accuracy scores for the test and train sets are calculated using accuracy_score.
- The overall accuracy is calculated as the average of test and train accuracies.
- Classification reports and confusion matrices are generated using classification_report and confusion_matrix to assess model performance.
- Cross-validation scores are calculated using cross_val_score with 5-fold cross-validation.
  
**Results Display:**
The script prints test accuracy, train accuracy, total accuracy, classification reports, confusion matrices, and cross-validation scores for each model.

**Comparison and Analysis:**

- By comparing the evaluation results for each model, you can assess their performance in terms of accuracy, precision, recall, and F1-score.
- This analysis helps you identify which model performs better for the specific task of credit card fraud detection.

Overall, the process involves data preparation, model training with hyperparameter tuning, evaluation of model performance using various metrics, and comparison to determine the best-performing model among Random Forest, Decision Tree, Logistic Regression, and KNN for the given dataset.

# Evaluating the performance of machine learning models

## Random forest Model

![image](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/assets/138976722/cced1da8-9876-4b39-9735-3648509507ec)

**Description:**

- **Test Accuracy:** The model achieved an accuracy of approximately 98.05% on the test set, meaning that it correctly classified around 98.05% of the transactions.

- **Train Accuracy:** The model achieved a perfect accuracy of 100% on the training set. While this might indicate strong performance, it's important to consider potential overfitting.

- **Total Accuracy:** The average of test and train accuracies, which is approximately 99.02%. This metric provides an overall idea of performance but doesn't consider overfitting.

**Classification Report:** The classification report provides metrics for each class (0 and 1) separately:

- **Precision:** The proportion of positive identifications that were actually correct.
- **Recall:** The proportion of actual positives that were correctly identified.
- **F1-Score:** The harmonic mean of precision and recall.
- **Support:** The number of occurrences of each class in the test set.
  
**Confusion Matrix:** Summarizes the model's performance in terms of true positives, false positives, true negatives, and false negatives.

**Cross-Validation Scores:** The accuracy scores obtained from cross-validation, indicating how well the model generalizes across different subsets of the data.

Overall, the model shows good performance with high accuracy, precision, and recall. However, it's essential to consider factors such as class imbalance and potential overfitting when interpreting these results. Cross-validation provides additional insight into model stability across different data subsets.

## Decision Tree Model

![image](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/assets/138976722/102f2910-350b-45ee-bcac-a3a2e25cf49f)

**Description:**

- **Test Accuracy:** The Decision Tree model achieved an accuracy of approximately 97.07% on the test set.

- **Train Accuracy:** The model achieved an accuracy of approximately 99.88% on the training set.

- **Total Accuracy:** The average of test and train accuracies, which is approximately 98.48%.

- **Classification Report:** Provides metrics for each class (0 and 1) separately, including precision, recall, and F1-score. The model shows slightly lower performance for class 1 (fraudulent transactions) compared to class 0.

- **Confusion Matrix:** Summarizes the model's performance in terms of true positives, false positives, true negatives, and false negatives.

- **Cross-Validation Scores:** Accuracy scores obtained from cross-validation, indicating how well the model generalizes across different subsets of the data.

The Decision Tree model demonstrates good performance, but its precision and recall for class 1 are slightly lower compared to the Random Forest model. The cross-validation scores also indicate stability in performance across different data subsets.

## Logistic Regression Model

![image](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/assets/138976722/29f3c6b4-6ac1-4525-88b9-14100489a738)

**Description:**

- **Test Accuracy:** The Logistic Regression model achieved an accuracy of approximately 94.63% on the test set.

- **Train Accuracy:** The model achieved an accuracy of approximately 97.99% on the training set.

- **Total Accuracy:** The average of test and train accuracies, which is approximately 96.31%.

- **Classification Report:** Provides metrics for each class (0 and 1) separately, including precision, recall, and F1-score. The model shows lower precision and recall for class 1 (fraudulent transactions) compared to class 0.

- **Confusion Matrix:** Summarizes the model's performance in terms of true positives, false positives, true negatives, and false negatives.

- **Cross-Validation Scores:** Accuracy scores obtained from cross-validation, indicating how well the model generalizes across different subsets of the data.

The Logistic Regression model performs reasonably well, but its precision and recall for class 1 are lower compared to both the Random Forest and Decision Tree models. The cross-validation scores suggest relatively consistent performance across different data subsets.

## K-Nearest Neighbors (KNN) Model

![image](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/assets/138976722/49896c7a-3fea-4a41-ac8b-d85a8266641f)

**Description:**

- **Test Accuracy:** The K-Nearest Neighbors (KNN) model achieved an accuracy of approximately 89.76% on the test set.

- **Train Accuracy:** The model achieved perfect accuracy (100%) on the training set, which could indicate overfitting.

- **Total Accuracy:** The average of test and train accuracies, which is approximately 94.88%.

- **Classification Report:** The KNN model exhibits lower precision and recall for class 1 (fraudulent transactions) compared to class 0. This indicates that the model has more difficulty correctly classifying fraudulent transactions.

- **Confusion Matrix:** The model has a higher number of false negatives (26) compared to other models, which means that it tends to miss identifying fraudulent transactions.

- **Cross-Validation Scores:** The accuracy scores obtained from cross-validation show some variability across different subsets of the data.

The K-Nearest Neighbors model achieves relatively good accuracy, but its performance is not as balanced as the Random Forest model. It exhibits a higher number of false negatives, indicating that it could benefit from further tuning or other algorithm choices to improve its performance on classifying fraudulent transactions.


# Implementation process Of deep learning models

##  Convolutional Neural Network (CNN) Model 

The below code snippet demonstrates the process of building and evaluating a Convolutional Neural Network (CNN) model for credit card fraud detection. Here's a breakdown of the model building process:

**1. Data Preparation:**

- The dataset is split into training and testing sets using the train_test_split function.
- SMOTE (Synthetic Minority Over-sampling Technique) is applied to balance the dataset by oversampling the minority class (fraudulent transactions).
- The features are converted to NumPy arrays and reshaped to fit the CNN model's input shape.
  
**2. Model Architecture:**

- The CNN model is created using the Sequential API from TensorFlow Keras.
- The model starts with a 1D convolutional layer with 32 filters and a kernel size of 3, using the ReLU activation function.
- A MaxPooling1D layer with a pool size of 2 is added to reduce the spatial dimensions.
- The output of the MaxPooling layer is flattened and connected to a dense layer with 64 units and a ReLU activation function.
- A dropout layer with a dropout rate of 0.2 is introduced to prevent overfitting.
- The final output layer with a single neuron and sigmoid activation function is used for binary classification (fraud or not fraud).

**3. Model Compilation:**

- The model is compiled using binary cross-entropy loss (suitable for binary classification) and the Adam optimizer.
- The accuracy metric is used to monitor the model's performance during training.

**4. Cross-Validation:**

- A StratifiedKFold cross-validation strategy with 5 splits is defined using StratifiedKFold.
- Lists are created to store cross-validation scores, training accuracies, and testing accuracies for each fold.

**5. Cross-Validation Loop:**

- The loop iterates through each fold of the cross-validation.
- For each fold:
       - The training set is trained on the CNN model for 10 epochs with a batch size of 32.
       - The validation set is used to calculate the accuracy of the model's predictions.
       - The training and testing accuracies are computed using the training and testing sets, respectively.
       - The calculated values are appended to their respective lists.

**Results:**
- Cross-validation scores, training accuracies, and testing accuracies are printed to assess the model's performance.

The provided code demonstrates how the CNN model is constructed, trained, and evaluated using cross-validation. This process helps in assessing the model's generalization performance and making comparisons to other models. Remember that tuning hyperparameters and exploring different architectures could further enhance the performance of the CNN model.

## Recurrent Neural Network (RNN) Model

The below code snippet demonstrates the process of building and evaluating a Recurrent Neural Network (RNN) model for credit card fraud detection. Here's a breakdown of the model building process:

**1. Data Preparation:**

- The dataset is split into training and testing sets using the train_test_split function.
- SMOTE (Synthetic Minority Over-sampling Technique) is applied to balance the dataset by oversampling the minority class (fraudulent transactions).
- The features are converted to NumPy arrays and reshaped to fit the RNN model's input shape.
  
**2. Model Architecture:**

- The RNN model is created using the Sequential API from TensorFlow Keras.
- An LSTM layer with 64 units and ReLU activation function is added to the model. The input shape is specified based on the reshaped data.
- A dense layer with 64 units and a ReLU activation function is added after the LSTM layer.
- A dropout layer with a dropout rate of 0.2 is introduced to prevent overfitting.
- The final output layer with a single neuron and sigmoid activation function is used for binary classification (fraud or not fraud).
  
**3. Model Compilation:**

- The model is compiled using binary cross-entropy loss (suitable for binary classification) and the Adam optimizer.
- The accuracy metric is used to monitor the model's performance during training.

**4. Cross-Validation:**

- A StratifiedKFold cross-validation strategy with 5 splits is defined using StratifiedKFold.
- Lists are created to store cross-validation scores, training accuracies, and testing accuracies for each fold.

**5. Cross-Validation Loop:**

- The loop iterates through each fold of the cross-validation.
- For each fold:
     - The training set is trained on the RNN model for 10 epochs with a batch size of 32.
    - The validation set is used to calculate the accuracy of the model's predictions.
    - The training and testing accuracies are computed using the training and testing sets, respectively.
    - The calculated values are appended to their respective lists.

**6. Results:**

Cross-validation scores, training accuracies, and testing accuracies are printed to assess the model's performance.

The provided code demonstrates how the RNN model is constructed, trained, and evaluated using cross-validation. The use of LSTM layers allows the model to capture sequential patterns in the data, making it suitable for time-series data like credit card transactions. Like with the CNN model, hyperparameter tuning and architecture exploration could further enhance the performance of the RNN model.


# Evaluating the performance of Deep learning models

##  Convolutional Neural Network (CNN) Model 

![image](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/assets/138976722/4d094bd1-177d-476a-b7c6-a9a81df5ad7e)

**Description:**

- **Training Accuracy:** The CNN model achieved a training accuracy of approximately 98.16%.

- **Testing Accuracy:** The CNN model achieved a testing accuracy of approximately 95.28% on the test set.

- **Total Accuracy:** The average of training and testing accuracies, which is approximately 96.72%.

- **Classification Report (CNN):** The CNN model demonstrates good precision and recall for both classes, although precision for class 1 (fraudulent transactions) is slightly lower.

- **Confusion Matrix (CNN):** The CNN model exhibits a relatively low number of false positives and false negatives, indicating a balanced performance in classifying both non-fraudulent and fraudulent transactions.

The CNN model shows promising results for credit card fraud detection. It achieves high accuracy and provides a balanced classification performance, making it a viable option for this task. The precision, recall, and F1-score metrics suggest that the CNN model is effective in identifying both non-fraudulent and fraudulent transactions.

## Recurrent Neural Network (RNN) Model

![image](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/assets/138976722/98999eb2-20f0-41d7-b51b-6469747795b3)

**Description:**

- **Training Accuracy:** The RNN model achieved a training accuracy of approximately 96.45%.

- **Testing Accuracy:** The RNN model achieved a testing accuracy of approximately 93.50% on the test set.

- **Total Accuracy:** The average of training and testing accuracies, which is approximately 94.97%.

- **Classification Report (RNN):** The RNN model demonstrates good precision and recall for class 0 (non-fraudulent transactions) but slightly lower precision for class 1 (fraudulent transactions).

- **Confusion Matrix (RNN):** The RNN model shows a reasonable balance between false positives and false negatives, suggesting effective classification performance for both classes.

The RNN model also shows promising results for credit card fraud detection. It achieves a high level of accuracy and provides a balanced classification performance, indicating its potential utility in identifying both non-fraudulent and fraudulent transactions. The precision, recall, and F1-score metrics provide insights into the RNN model's capability to classify transactions accurately.

# Comparative analysis between machine learning and deep learning models

Compiling all the results of the different models, both machine learning and deep learning, and analyze their performance based on various metrics:

![image](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/assets/138976722/fe72df79-101c-4d5a-b395-fd3ed6b1d60a)

![image](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/assets/138976722/99550423-0e90-4a58-85b5-c8d3cecbbf1b)

![image](https://github.com/Khadija-khanom/Credir_Card_Fraud_Detection3/assets/138976722/36ebe854-61cb-47d1-865c-ee5e030a13f7)

Based on the comparison of these results, the Random Forest model appears to perform the best overall. It achieves high testing accuracy, total accuracy, and performs well in terms of precision, recall, and F1-score for class 1 (fraudulent transactions). It's worth noting that the Random Forest model avoids overfitting and captures the complexity of the data.

While both the CNN and RNN models demonstrate competitive performance, they fall slightly behind the Random Forest model in terms of overall accuracy and precision-recall metrics. The machine learning models (Random Forest and Decision Tree) outperform the deep learning models (CNN and RNN) in this specific scenario of credit card fraud detection with the given dataset.

