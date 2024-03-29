# An Application and Evaluation of Artificial Neural Networks in the Classification of Credit Applicants with Comparisons to Traditional Methods with the Additional Use of Hyperparameter Tuning

## An Application of Artificial Neural Networks for the Classification of Credit Applicants with Comparisons to Traditional Methods

This is the documentation for my final project of the MSc Financial Technology course. This project focuses on the application of TensorFlow neural networks for the classification of credit applicants, and compares the results with traditional traditional methods of classification: linear decision analysis, logistic, and random forest. 

I came up with the hypothesis that previous researchers failed to use hyperparameter tuning in their research. Therefore, could we say for certainty that certain models are better than others in this classification task? After all, a logistic regression could be set up for failure given it's base parameters. Therefore, I wanted to investigate how hyperparameter tuning affects the models performance/ranking. Furthermore, some researchers called for more investigations into the application of neural networks for credit default prediction

The project uses various Python libraries such as TensorFlow and SKlearn (full list in requirements.txt).

In this project, the data is preprocessed, cleansed, trained and ultimately evaluated. Multiple techniques are used such as, one hot encoding, a linear regression to fill NaN value, Keras-Tuner to aid in hyperparameter tuning of the neural network, and FastAPI to enable deployment.

### Project Overview
The goal of this project is to develop a credit applicant classification system using TensorFlow neural networks. The project uses a supervised learning approach to train the models using historical credit application data. The trained models will then be used to predict the creditworthiness of new applicants. To conclude, we compare the neural network to traditional models that are also trained on the same dataset.

Past the research of the paper, I have also gone the next step and implemented a FastAPI to enable the deployment of a model

### Why This Topic
I choose this topic as I am very interested in financial risk. I think the whole area is very interesting as ultimately we are quantifying something that isn't inherently quantifiable due to the uncertainties involved - nevertheless, we have been able to do it successfully. Plus, with the advancements in technology over the past decade I feel artificial intelligence is starting to position itself at the forefront of risk management. 

### Installation
To run this project, you need to install Python libraries contained in the requirements file:

Python Version:
<pre><code>Python 3.9.13</code></pre>
You can install the libraries with the command: <pre><code>pip install -r requirements.txt</code></pre>

### Data Preprocessing
Before the data is even used in the model, it needs to be cleaned. Firstly, the data needs to be encoded to integers as there are multiple string columns. Also, there are numerous NaN's in the columns of the dataset, plus, the dataset is imbalanced roughly 11:1.

Therefore, I used one hot encoding to encode the dataset. If there was any ordinal features I would have also used a label encoder

The NaN's were replaced using a linear regression as I felt this is the most reproducible method for a research paper

I wanted to note that I did attempt to use an undersampled dataset however, there just wasn't enough data for the neural network to converge to the global minima. In fact, it found it hard to progress at all. Therefore, I had to oversample the dataset using SMOTE (as seen below).  

### Balancing the Dataset
The imblearn library enabled me to use Synthetic Minority Oversampling Technique (SMOTE) to equally balance the dataset. Interestingly, it uses K-nearest neighbor to populate the dataset with synthetic values. 

Balancing the dataset helps us to prevent the models from being biased towards the majority class. If the data wasn't balanced, the model wouldn't ever really learn how to deal with the minority class. This essentially means that the model doesn't generalise well when it encounters a minority case.

This was used in previous research but also as a best practice to enable an even playing field for the models as it is known for example that a logistic regression does not handle imbalanced datasets well

### Splits
I took an unusual route to tuning my models. Due to the limited computing power, I split the dataset 50/50. 50% of the dataset would be used to tune for hyperparameters - and only hyperparameters. I would then take these hyperparameters and run a K-fold cross-validation (K = 5) on the remaining 50% of the data to get an evaluation for the models performance

In an ideal world, I would be able just to run a straight nested K-fold where, I retune hyperparameters for each fold. However, the breadth of this paper isn't large enough to enable me to do this. I could reduce the dataset in size however, I already saw with undersampling that a reduced sample size significantly impacted the performance of the model

### Neural Network Model
The neural network used in this project was built using TensorFlow. The model architecture is shown below: 

<p align="center">
  <img src="https://github.com/JackGreenaway/final_project/blob/main/misc/nn_model.png"/>
</p>

The model was found using hyperparameter tuning. The tuner was BayesianSearch as I found that although, a lengthily searching process, it produced the best results for the model. I made sure to include regularisation techniques to avoid the chances of overfitting. Through hyperparameter tuning these were also picked.

Although, the output layer used a sigmoid activation, I still needed to transfer the continuous data into binary. I decide a 50/50 split for either 0:1. Though, in a real world example, a company could choose their tolerance levels.

It is important to note that as a neural network provides a continuous output, it actually provides a probability of default which in some cases is far more useful to an institution.

### Linear Discriminant Analysis & Logistic Regression Models & Random Forest
A linear discriminant analysis (LDA) model is built and trained using the credit application data. LDA is a linear classification technique that seeks to find the optimal linear combination of features that best separates the classes.

A logistic regression model is also built and trained using the credit application data. Logistic regression is a traditional statistical method used for binary classification tasks.

A random forest was also investigated.

These models were picked as per my research repeatedly named these two models as traditional models used for the classification of credit applicants:

- Bekhet, H. A., & Eletter, S. F. K. (2014). Credit risk assessment model for Jordanian commercial banks: Neural scoring approach. Review of Development Finance, 4(1), 20-28. 

- Ince, H., & Aktan, B. (2009). A comparison of data mining techniques for credit scoring in banking: A managerial perspective. Journal of Business Economics and Management, 10(3),233-240. 

- Loan default prediction using decision trees and random forest: A comparative study, M. Madaan, A. Kumar, C. Keshri, R. Jain and P. Nagrath, IOP Conference Series: Materials Science and Engineering 2021, Publisher: IOP Publishing Pages: 012042


I did look into using a SVM however, the quadratic time complexity meant that I didn't really have enough computing power to easily/quickly train the model. I could have used a linear SVM however it still took ages to do anything. I tried to incorporate PCA but, the training still took forever due to the number of samples.

### Models Evaluation
To evaluate the models, I used accuracy, precision, recall, F1-score, and AUC-score. I also used a AUC curve (for the neural network), and a heatmap to visualise the predictions. Lastly, arguably the most important metric: the false-negative, was calculated for each model.   

The false negative is often seen as more important than the accuracy as it represents a significant financial penalty for an organisation if too many people who shouldn't get credit, get credit - in an ideal world, none of them would get credit

NOTE: in regards to the binary classification, 1 = default, 0 = non-default

### Deploying the Model
I created a FastAPI to allow others to access the created model. There is also a demo file in the directory to display how the API works. I am thinking of hosting the file live when my grading comes around. I am also attempting to develop a pipeline that can periodically retrain the model if there was feature drift or ground truth drift. 

I have implemented MLFlow into the API to log inputs into the API. Also, there is an SQL database that stores the inputs and corresponding predictions. I have started to investigate how I can detect feature drift using statistical testing however, it's hard to implement accurately/meaningfully as the dataset is ultimately static - though, I am using it as a practising case to improve my understanding and practical ability in MLOps

I've also set it up to be ready to be placed into a docker container

### Results and Conclusions
The results of the study find that the neural network has the best overall accuracy, recall, F1, AUC and second-best FpR. However, it has the second-worst FNR which is arguably the most important metric when lending. The results are as follows:

<p align="center">
  <img src="https://github.com/JackGreenaway/final_project/blob/main/misc/results_table.png"/>
</p>

This project has demonstrated my ability to take a real-world problem and ultimately, create a meaningful solution. I have used TensorFlow, SKlearn, data preprocessing techniques, visulisation libraries, and metrics to create a solution that hopefully could decrease the risk levels of a lender. 

### Notes
Please note that due to the size of the trained random forest models, they have not been uploaded to GitHub
