# Credit Applicant Classification with TensorFlow Neural Networks

## Current todo:
- add cross validation (fold)
- retrain models on new dataset composition

## An Application of Artificial Neural Networks for the Classification of Credit Applicants with Comparisons to Traditional Methods

This is the documentation for my final project of the MSc Financial Technology course. This project focuses on the application of TensorFlow neural networks for the classification of credit applicants, and compares the results with traditional traditional methods of classification: linear decision analysis and logistic regression. 

The project uses various Python libraries such as TensorFlow and SKlearn (full list in requirements.txt).

In this project, the data is preprocessed, cleansed, trained and ultimately evaluated. Multiple techniques are used such as, an algorithm I developed to encode the dataset as well as, Keras-Tuner to aid in hyperparameter tuning of the neural network.

### Project Overview
The goal of this project is to develop a credit applicant classification system using TensorFlow neural networks. The project uses a supervised learning approach to train the models using historical credit application data. The trained models will then be used to predict the creditworthiness of new applicants. To conclude, we compare the neural network to traditional models that are also trained on the same dataset. 

### Why This Topic
I choose this topic as I am very interested in financial risk. I think the whole area is very interesting as ultimately we are quantifying something that isn't inherently quantifiable due to the uncertainties involved - nevertheless, we have been able to do it successfully. Plus, with the advancements in technology over the past decade I feel artificial intelligence is starting to position itself at the forefront of risk management. 

### Installation
To run this project, you need to install Python libraries contained in the requirements folder:

You can install the libraries with the command: <pre><code>pip install -r requirements.txt</code></pre>

### Data Preprocessing
Before the data is even used in the model, it needs to be cleaned. Firstly, the data needs to be encoded to integers as there are multiple string columns. Also, there are numerous NaN's in the columns of the dataset, plus, the dataset is imbalanced roughly 11:1.

Therefore, I created an algorithm to go through the dataset and populate a dictionary containing a key. This key is then used to encode the dataset to numbers. We replace NaN's with 0's, and use the imblearn library to balance the dataset.

I wanted to note that I did attempt to use an undersampled dataset however, there just wasn't enough data for the neural network to converge to the global minima. In fact, it found it hard to progress at all. Therefore, I had to oversample the dataset.  

### Balancing the Dataset
The imblearn library enabled me to use Synthetic Minority Oversampling Technique to equally balance the dataset. Interestingly, it uses K-nearest neighbor to populate the dataset with synthetic values. 

Balancing the dataset helps us to prevent the models from being biased towards the majority class. If the data wasn't balanced, the model wouldn't ever really learn how to deal with the minority class. This essentially means that the model doesn't generalise well when it encounters a minority case.  

### Neural Network Model
The neural network used in this project was built using TensorFlow. The model architecture is shown below: 

<p align="center">
  <img src="https://github.com/JackGreenaway/final_project/blob/main/misc/nn_model.png"/>
</p>

The model was found using hyperparameter tuning. The tuner was BayesianSearch as I found that although, a lengthily searching process, it produced the best results for the model. I made sure to include regulation techniques to avoid the chances of overfitting. Through hyperparameter tuning these were also picked.

Although, the output layer used a sigmoid activation, I still needed to transfer the continuous data into binary. I decide a 50/50 split for either 0:1. Though, in a real world example, a company could choose their tolerance levels.

### Linear Discriminant Analysis & Logistic Regression Models
A linear discriminant analysis (LDA) model is built and trained using the credit application data. LDA is a linear classification technique that seeks to find the optimal linear combination of features that best separates the classes. The model is evaluated using accuracy, precision, recall, and F1-score.

A logistic regression model is also built and trained using the credit application data. Logistic regression is a traditional statistical method used for binary classification tasks. The model is evaluated using accuracy, precision, recall, and F1-score.

These models were picked as per my research repeatedly named these two models as traditional models used for the classification of credit applicants:

- Bekhet, H. A., & Eletter, S. F. K. (2014). Credit risk assessment model for Jordanian commercial banks: Neural scoring approach. Review of Development Finance, 4(1), 20-28. 

- Ince, H., & Aktan, B. (2009). A comparison of data mining techniques for credit scoring in banking: A managerial perspective. Journal of Business Economics and Management, 10(3),233-240. 

### Models Evaluation
To evaluate the models, I used accuracy, precision, recall, F1-score, and AUC-score. I also used a AUC curve (for the neural network), and a heatmap to visualise the predictions. Lastly, arguably the most important metric: the false-positive, was calculated for each model.   

NOTE: in regards to the binary classification, 1 = default, 0 = non-default

### Deploying the Model
I created a FastAPI to allow others to access the created model. There is also a demo file in the directory to display how the API works. I am thinking of hosting the file live when my grading comes around. I am also attempting to develop a pipeline that can periodically retrain the model if there was feature drift or ground truth drift. 

I have implemented MLFlow into the API to log inputs into the API. Also, there is an SQL database that stores the inputs and corresponding predictions. I have started to investigate how I can detect feature drift using statistical testing however, it's hard to implement accurately/meaningfully as the dataset is ultimately static - though, I am using it as a practising case to improve my understanding and practical ability in MLOps

### Results and Conclusions
The preliminary results show that my neural network outperforms traditional methods. However, this project is still a work in progress therefore, I cannot make conclusions yet about the effectiveness of my model. 

This project has demonstrated my ability to take a real-world problem and ultimately, create a meaningful solution. I have used TensorFlow, SKlearn, data preprocessing techniques, visulisation libraries, and metrics to create a solution that hopefully could decrease the risk levels of a lender. 
