# MODELLING HEALTH INSURANCE CLAIMS FRAUD USING LOGISTIC REGRESSION: AN APPLICATION OF MACHINE LEARNING


## Abstract

The insurance industry is plagued by health insurance fraud in Kenya. Health insurance fraud involves deception and misrepresentation to get illicit health insurance benefits. This approach reduces affordable medical care for all socioeconomic groups. The analysis highlights the rising frequency of fraudulent claims and insurance and public healthcare program losses. It stresses how important it is to resolve these issues to preserve the healthcare system. This research seeks to detect fraudulent health insurance claims using logistic regression and stresses the importance of detecting and eradicating fraud to protect policyholders, minimize costs, and retain insurance sector trust. The research approach includes data collection, preparation, model fitting, and performance evaluation. The model's performance is measured by accuracy, precision, recall, fall-out, and AUR-ROC. The study fills gaps in fraud detection, data validation, and fraudulent intent research. It uses machine learning to detect health insurance fraud to improve healthcare efficiency and cut costs aiming to help the Kenyan and international insurance industry comprehend the rise in health insurance fraud.

## TABLE OF CONTENTS


- [Abstract](#abstract)
- [CHAPTER ONE: INTRODUCTION](#chapter-one:-introduction)
- [Background of the Study](#background-of-the-study)
- [Statement of the Problem](#statement-of-the-problem)
- [Research Objectives](#research-objectives)
- [Significance of the Study](#significance-of-the-study)
- [Scope](#scope)
- [CHAPTER TWO: LITERATURE REVIEW](#chapter-two:-literature-review)
- [Introduction](#introduction)
- [Application of Machine Learning in Health Insurance Fraud Detection](#application-of-machine-learning-in-health-insurance-fraud-detection)
- [Generalized Linear Models](#generalized-linear-models)
- [Advantages of Logistic Regression in Detecting Health Insurance Fraud](#advantages-of-logistic-regression-in-detecting-health-insurance-fraud)
- [Research gap](#research-gap)
- [CHAPTER THREE: METHODOLOGY](#chapter-three:-methodology)
- [Introduction](#introduction)
- [Data Source](#data-source)
- [Sampling Design](#sampling-design)
- [Data Assumptions](#data-assumptions)
- [Model Specifications](#model-specifications)
- [Model Assumptions](#model-assumptions)
- [Methods for achieving Objective 1 and 2](#methods-for-achieving-objective-1-and-2)
- [Methods for achieving Objective 3](#methods-for-achieving-objective-3)
- [CHAPTER FOUR: DATA ANALYSIS, PRESENTATION, AND INTERPRETATION](#chapter-four:-data-analysis,-presentation,-and-interpretation)
- [Introduction](#introduction)
- [ Data Analysis](#data-analysis)
- [Binary logistic model](#binary-logistic-model)
- [Testing for significance](#testing-for-significance)
- [Performance Metrics](#performance-metrics)
- [CHAPTER FIVE: CONCLUSION](#chapter-five:-conclusion)
- [Conclusion](#conclusion)
- [Recommendations](#recommendations)
- [GLOSSARY](#glossary)
- [REFERENCES](#references)
- [States and their Codes](#states-and-their-codes)
- [Top 10 diagnosis involved in health fraud and their codes](#top-10-diagnosis-involved-in-health-fraud-and-their-codes)
- [Top 10 procedures involved in health care frauds and their codes](#top-10-procedures-involved-in-health-care-frauds-and-their-codes)
- [Races and their Codes](#races-and-their-codes)
  


## TABLE OF FIGURES
[Figure 4.2.1: Distribution of claim records](#figure-4.2.1:-distribution-of-claim-records)
[Figure 4.2.2: Potential fraud among providers](#figure-4.2.2:-potential-fraud-among-providers)
[Figure 4.2.3: State-wise beneficiary distribution](#figure-4.2.3:-state-wise-beneficiary-distribution)
[Figure 4.2.4: Race-wise Beneficiary Distribution](#figure-4.2.4-race-wise-beneficiary-distribution)
[Figure 4.2.5: Top 10 procedures involved in healthcare fraud](#figure-4.2.5:-top-10-procedures-involved-in-healthcare-fraud)
[Figure 4.2.6: Top10 Diagnosis involved in Healthcare Fraud](#figure-4.2.6:-top-10-diagnosis-involved-in-healthcare-fraud)
[Figure 4.4.1: Receiver Operating Characteristic](#figure-4.4.1:-receiver-operating-characteristic)
[Figure 4.4.2: True Positive Rate vs True negative Rate](#figure-4.4.2:-true-positive-rate-vs-true-negative-rate)



## CHAPTER ONE: INTRODUCTION


### Background of the Study


One of the primary challenges confronting the insurance sector pertains to the prevalence of fraudulent claims and instances of abuse. Health insurance fraud is an act of deception or intentional misrepresentation to obtain illegal benefits concerning the coverage provided by a health insurance company (J Villegas-Ortega et.al, 2021). The Insurance Fraud Investigation unit reported a rise in fraudulent cases in Kenya from 83 in 2019 to 123 in 2020 (IRA, 2020). Fraudulent activity within the health sector has a direct impact on the availability of affordable healthcare for individuals across all socioeconomic strata. Fraudulent activities by contracted hospitals are reported to cost The National Health Insurance Fund (NHIF) up to 16.5 billion shillings annually (Alushula, 2021). Various surveys have reported that about 10% of health insurance claims are fraudulent (Mitic, 2023). The high reported losses show the severe underreporting of fraudulent health insurance claims.

Health insurance fraud has undergone significant transformations in the 21st century, exhibiting a diverse range of profiles that span from rudimentary fraudulent schemes to intricate networks. These fraudulent claims are resulting in financial losses for companies, imposing significant medical expenses on ordinary citizens, and contributing to the decline of the Kenyan economy. NHIF reported that some of its contracted hospitals exaggerated their bed numbers by up to five times to defraud it (Alushula, 2021). Despite the efforts made by the government and other relevant stakeholders to address this criminal activity, there is a pressing need for additional measures to effectively close the existing loopholes exploited for fraudulent purposes. The government created The Insurance Fraud Investigation Unit (IFIU) in 2011 as a measure to examine instances of fraudulent activity (Insurance Fraud Investigations Unit (IFIU), 2020). The unit receives a limited number of cases annually, and because of thorough investigative procedures, only a small portion of these cases are ultimately concluded, with appropriate advice provided to the complainants.

The past few decades have seen an immense growth in Artificial Intelligence (AI). Most notably, machine learning techniques have been incorporated by insurance firms to detect fraudulent claims (Bhatore, 2020). Researchers have explored and still are exploring various machine learning techniques to improve the accuracy of fraud detection, especially in the medical insurance sector.

### Statement of the Problem

The prevalence of misuse and fraud in health insurance is currently causing an increase in anxiety within the insurance sector. Insurers face a substantial issue in effectively assessing and pricing risks due to the rising costs connected with fraudulent activity (NAIC, 2022). As a result, these insurers' returns on equity have decreased and premiums have increased. Due to fraudulent actions, public programs, such as the National Health Insurance Fund (NHIF), suffer severe negative repercussions. Therefore, modelling health insurance fraud is important for reducing the burden and costs of compliance for law-abiding healthcare institutions, practitioners, and their patients. 

 ### Research Objectives
 
#### General Objective
To model health insurance claims fraud using logistic regression
#### Specific Objectives
1.	To fit a logistic regression model on health insurance claims fraud data.
2.	To train the logistic regression algorithm to generate an appropriate model for assessing health insurance claims.
3.	To evaluate the performance of the model using AUR-ROC and confusion matrix metrics.
	
### Significance of the Study

This report aims to substantiate the significance of fraud detection as a primary function of health insurers and as a domain for prospective investigation. Undoubtedly, fraud imposes a financial burden on the health care system without yielding any advantages. Therefore, this study will play a crucial role in fostering a more streamlined health care system.

The implementation of the research proposal aims to facilitate a reduction in insurance claim fraud within the health insurance system, resulting in cost savings.
The research also helps in knowing factors that lead to rise on insurance health claims fraud.

### Scope

The research project will focus on how the medical insurance industry can model fraudulent claims and therefore identify potential fraud cases that will cost them a lot of money. They then can employ ways to seal loopholes that lead to fraud. Consequently, the cost of healthcare in a developing country like Kenya can be manageable since fraud inflates the cost of healthcare.

 

## CHAPTER TWO: LITERATURE REVIEW


### Introduction 


Fraud and abuse take place at many points in the healthcare system. Doctors, hospitals, nursing homes, diagnostic facilities, medical equipment suppliers, and attorneys have been cited in scams to defraud the system. The National Health Care Anti-Fraud Association (NHCAA) estimates that the financial losses due to health care fraud are $68 billion, or as high as $300 billion (III, 2022). Health insurance fraud committed by patients may be an increasing problem given the number of underinsured and uninsured people.

According to the research done by the Insurance Information Institute, common types of Health Care frauds include Kickback schemes where actors might improperly pay for or waive the client’s out-of-pocket expense to make up for that cost in additional business. Unbundling and fragmentation of procedures where separate claims are created for services or supplies that should be grouped together. When it comes to submitting claims not only improper coding practices can be fraudulent, but also care providers can try to submit the same claim multiple times, to get paid two times for performing one action. Further, phantom billing for services not rendered to clients takes place and this is also a form of fraud alongside billing unnecessary services that could lead to client harm. Claims are submitted for a service provided based on a stated diagnosis. These diagnoses can also be manipulated i.e., a patient can get a certain diagnosis while that diagnosis is not actually true. This type of fraud can be done to falsely prescribe certain medicines to a patient, for example.

Fraud detection and prevention are crucial in the health insurance industry to control costs, protect policyholders, and ensure the financial stability of insurers. These practices promote fairness and equity in insurance benefits, comply with regulations, and safeguard sensitive data. By stopping fraudulent activities, insurers can streamline claim processing, improve healthcare quality, and maintain trust with policyholders and stakeholders, ultimately contributing to a sustainable and reliable healthcare system. 

Fraud and abuse are widespread and very costly to Kenya's health-care system. Even though the medical business premiums have more than doubled in the last 4 years, the industry registered an average loss of Kshs.33,394,812 per year (33%) over the 4-year period (AKI, Health Insurance Fraud Survey Report, 2018). According to the Insurance Industry Annual Report given by IRA, 2020, IFIU reported a rise in medical fraudulent claims from 9 cases in 2019 to 12 fraud cases in 2020 which costed a total of Ksh.1511290. 

The Insurance Industry Annual Report (2018) by AKI identifies prevalent forms of insurance fraud in Kenya, including manipulating diagnoses, substituting branded drugs with generic ones, excessive servicing, swapping memberships, splitting fees, altering invoices or claims, and failing to disclose pre-existing medical conditions. Health frauds among other factors have contributed to the collapse of some health Insurance companies in Kenya.

Many health insurance companies in Kenya have put up various technologies and strategies to fight fraud such as planned, targeted audits, random audits, whistleblowing, biometric systems, application of data mining such as classification algorithms (Naïve Bayes, Decision Tree and K-Nearest Neighbour) (Mambo, S., & Moturi, C. (2022)). It is estimated that 25% of insurance industry income is fraudulently claimed and 35% of medical claims are fraudulent according to AKI report produced in 2018. The implementation of biometric based technology for authentication of insured patients has been adopted to curb rising cases of fraud in medical insurance with 60-80% coverage.  

### Application of Machine Learning in Health Insurance Fraud Detection

It has proved a necessity for insurance firms to implement effective measures to detect any cases of fraudulent claims before claims are settled. Some of the notable methods identified to help detect fraud in medical insurance includes Decision Trees (DTs), Random Forest (RF), Neural Network (NN), Support Vector Machine (SVM), and k-nearest neighbors (KNN). Traditionally speaking, in the healthcare anomaly detection area, machine learning applications can be divided into supervised learning and unsupervised learning (Zhang C, Xiao X & Wu C., 2020). Basically, supervised learning needs label of data for training, while unsupervised learning does not. Classification trees and decision trees (DTs) are binary trees that classify data into predetermined groups. The tree is constructed using splitting rules in a recursive manner from top to bottom. These rules are often univariate, but they can employ the same variable in zero, one, or several splitting rules.

The fact that DTs do not output the likelihood of classifying a claim as valid or fraudulent, (and as a result do not distinguish across claims in the same classification), is a significant drawback (Maher, 2020). There is no evidence to support the claim that decision trees developing algorithms will reduce risk by not using previous rules when determining future rules but there is no evidence to suggest that this will reduce classiﬁcation or prediction accuracy ability.

DT technique is the foundation of the ML-based random forest approach, which is frequently employed to tackle various regression and classification issues. It helps in highly accurate output prediction in datasets which are large. Random forest aids in forecasting the average mean of other trees' output. The precision of the result increases as the number of trees increases. The decision tree algorithm has a number of disadvantages that the random forest approach helps to overcome (Darwish, 2019). Additionally, it reduces dataset lifting, boosting precision. The RF approach is quick and efficient for managing unbalanced and big volumes of datasets (Sulaiman et al., 2022). However, the random forest has drawbacks when it comes to training different datasets, particularly for regression issues. Although the random forest algorithms are quite good at identifying the class of regression issues, Bin Sulaiman, (2022) claims that they have a number of drawbacks when it comes to real-time medical insurance fraud detection. In real-time applications, random forest algorithms perform less well, making predictions take longer since the training process is slower. Consequently, in order to effectively detect fraud in real-world datasets, a large volume of data is needed, and random forest algorithms lack the capability of training the datasets effectively and making predictions. 

Artificial Neural Network (ANN) is an ML algorithm which functions similarly to the human brain. ANN is based on two types of methods: supervised method and unsupervised method. The Unsupervised Neural Network is widely used in detecting fraud cases since it has an accuracy rate of 95% (Maher, P. 2020). The unsupervised neural network attempts to find similar patterns between fraudulent claims settled previously and the current claim reported. Suppose the details found in the current claims are correlated with the previous claims, then a fraud case is likely to be detected. ANN methods are highly fault tolerant and is a viable option for the identification of medical insurance fraud due to its high processing speed and efficiency. However, when combined with a variety of algorithms and functions, it has generated fewer good results. Each of these functions is deficient in some way.

KNN is an example of a supervised machine learning technique useful for problem classification and regression analysis. It is a useful technique for supervised learning. It aids in enhancing detection and lowering the rate of false positives. In order to prove the existence of medical insurance fraud, it employs a supervised technique (Alam et al., 2021). The memory-intensive algorithm KNN scales up properties of non-essential data. When a big amount of data is entered, KNN algorithm's performance declines as data volume increases (Bin Sulaiman et al., 2022). These limitations consequently affect the accuracy and recall matrix in the identification of insurance fraud. Linear models have been previously used to predict and model the medical expenses for new customers. This can be in turn be used in future to detect anomalies in the expense using linear regression models and machine learning algorithms that are trained to estimate possible medical expense (Makkar et al., 2023). These algorithms through deep learning and statistical linear models offer transparency since they are free from human interventions. They mainly depend on large data that is used to primarily train the algorithm before it is tasked to come up with the said expense.

Multiple (linear) regression models calculate the expected value E (*) of one variable Y (the 'dependent' or 'outcome' variable) from a linear combination of several observed covariates X1; Xn (the 'independent' or 'explanatory' variables, or just 'predictors').

### Generalized Linear Models

A class of statistical models known as generalized linear models (GLMs) expands the classic linear regression model to consider a larger variety of data distributions and response types. The response variable in GLMs may have one of several distributions, including the Gaussian (normal), binomial, Poisson, or gamma distributions. The response variable's predicted value and the predictors are connected by the link function. It converts the linear combination of predictors into the proper scale of the distribution of the response variable. 

By including a link function and a particular distribution assumption, GLMs offer a flexible framework for modeling the relationship between the dependent variable and one or more predictors (Arnold et al., 2020). In a GLM, parameter estimates, and their standard errors give information about the effect, uncertainty, and statistical relevance of all variables. Which in this case GLM outperforms other machine learning techniques such as decision trees (Henckaerts et al., 2021). The coefficients βhat₀, βhat₁, βhat₂, ..., βhatn for a given GLM are typically obtained via a statistical process known as maximum likelihood estimation. Although GLMs are conceptually straightforward to comprehend and apply, as the number of covariates increases, estimation of their parameters without the aid of computational equipment quickly becomes impracticable. Therefore, a revolution in data analytics was made possible by the development of programmable desktop computers in the 1980s and 1990s since it was now possible to do the intricate matrix inversions necessary quickly and automatically for generalized linear modeling.

  Fraud results in the creation of outliers due to non-uniformity. It is a crucial component of fraud prediction for prediction models to determine the likely value (or risk) of an outcome given information from one or more 'predictors'. The GLM model is therefore paramount in modelling health insurance fraud by narrowing down to only factors relevant to healthcare insurance. It is also important to evaluate GLM from a business point by using business metrics used in insurance tariffs evaluation.
  
Generally, these supervised machine learning algorithms are heavily dependent on training datasets. Because of the complicated nature of the healthcare sector, a dataset is usually not comprehensive, this therefore causes the result to be seriously over-fitted in real-world scenarios hence the GLM models are preferred. Types of GLMs commonly used in fraud modelling are logistic regression, Poisson regression, Negative Binomial Regression and Gamma Regression. Logistic regression is typically used when the response variable is binary (fraud vs. non-fraud). The logistic link function transforms the linear combination of predictors into the likelihood of fraud (Maher, 2020). This model calculates the odds ratio for each predictor and displays how the probability of fraud changes when the predictor is altered by one unit. Poisson regression is appropriate when the response variable contains count data, such as the number of erroneous claims. The Poisson distribution and log link function are used to model the relationship between the predictors and the anticipated number of fraud cases. Poisson regression assumes that the mean and variance of the response variable are both equal. In the case of the Negative Binomial regression, the assumption of equal mean and variance is relaxed in this extension of Poisson regression. It is employed when over dispersion, or when the variance is greater than the mean, is present in the count data. To account for the additional variation, negative binomial regression models the link between the predictors and the anticipated fraud count. Gamma regression can be used to model continuous, positively skewed response variables with a gamma distribution, such as medical expenses or claim amounts. Gamma regression takes into consideration the unique distributional characteristics of these variables, enabling more precise modeling and forecasting of the costs associated with fraud.

### Advantages of Logistic Regression in Detecting Health Insurance Fraud

Logistic regression models can handle binary response variable distributions, making them suitable for modeling data encountered in health insurance fraud detection, such as fraud vs. non-fraud.

 Logistic regression models also offer interpretable coefficients that enable analysts to comprehend the degree and direction of each predictor variable's impact on the response variable. Finding the causes of health insurance fraud requires being able to understand data. The direction of the link between the predictor factors and the response variable (in this case, health insurance fraud) is represented by the coefficients provided by the model. A positive coefficient indicates a positive association, indicating that the likelihood of fraud increases as the predictor variable rises. A negative coefficient, on the other hand, denotes a negative association, where a rise in the predictor variable is connected to a fall in the risk of fraud. It is essential to comprehend the direction of the influence when figuring out what causes or discourages health insurance fraud. Therefore, Logistic regression models reveal not just the direction but also the size or intensity of each predictor variable's influence on the response variable (Maher, 2020). The size of the coefficient represents how much the likelihood of fraud is affected by a unit change in the predictor variable. Smaller coefficients represent a relatively weaker link, while larger coefficients show a more significant influence of the predictor variable on fraud. Analysts can determine the most important causes of health insurance fraud by looking at the magnitude of the coefficients.
 
Analysts can learn more about the elements that have a big impact on health insurance fraud by having coefficients that are easy to interpret. This knowledge can help with decision-making procedures like creating fraud detection techniques, effectively allocating resources, and putting in place targeted measures to reduce fraud risks. Additionally, the interpretability of the model’s coefficients makes it easier to validate and replicate study findings in identifying health insurance fraud and to communicate findings more clearly.

Logistic Regression Models enable testing of hypotheses regarding the importance of predictor factors (Henckaerts et al., 2021). These aids in identifying the factors that statistically significantly predict fraud and evaluating the relative contributions of each factor. Additionally, the model can incorporate categorical information in the fraud detection model.

### Research gap

 There is often insufficient detail on the underlying methodologies and data samples that lead to fraud estimates, which may be due to different purposes of these reports or the need to obscure the details of fraud detection methods to prevent fraudulent operators from responding to existing methods. The main gaps identified are validation of existing methods and proof of intent to commit fraud in the studies analyzed. The lack of automation in medical processing within the Kenyan medical insurance sector creates an opening for exploring the implementation of machine learning for local medical claim processing using available data. Therefore, this research project is intended to provide suitable methodology on supervised machine learning on detecting health insurance claim fraud.



## CHAPTER THREE: METHODOLOGY

### Introduction

This chapter elaborates the methodology that was used to accomplish the already established research objectives sequentially. The research design, data collection and analysis, are briefly illustrated.

### Data Source

The data that was used in this study is Medicare insurance data posted on Kaggle, an online portal for community of data scientists. The dataset captures individuals from age 26 to 100 and was specific whether they suffered from chronic diseases, and if so, which ones. The study used this data as the people within this age range are more likely to have health insurance. The claim behavior patterns were studied for a period of two years.

### Sampling Design

Data used in this study was from a secondary data source. This data was sampled to be a representative of Healthcare Insurance holders. The research specifically focused on outpatient data for patients who visited hospital to receive treatment but were not admitted in them, inpatient data for patients who visited hospital to receive treatment and were admitted, beneficiary data for claim beneficiaries and their identifying characteristics such as race, religion, region and whether they suffered from chronic illnesses such as heart failure.

- Outpatient data
  
The data here is for patients who visited hospital to receive treatment but were not admitted in them. The datasets were split into train and test datasets, each with elaborate claim features, describing claim details. This dataset had 517,737 records and 27 feature columns.

- Beneficiary data
   
The data here was for beneficiaries of each claim and their identifying characteristics such as race, religion, region and whether they suffered from chronic illnesses such as heart failure. This data was also split into train and test datasets with 138,556 records and 25 feature columns. For sensitivity of the information provided, most of the identifying features were coded for data privacy of the involved individuals for example, Race is a discrete random variable taking the values 1 to 5.

- Train data
  
This file contained a unique provider identifier and the class label, potential fraud, labelled Yes and No, coded as 1 and 0 respectively. For a Fraudulent claim, the claim is reviewed and if it is a No then the claim is correct to be paid. Data was aggregated at Unique claim ID for each claim made by the provider. The unique providers handling each claim were 5410.

### Data Assumptions

There were various assumptions taken in the model and these include:
1.	Claims can be filed more than once by a client.
2.	All claim payments are fulfilled by insurance firms shortly after they have been filed.
3.	All claims result in some payments.

### Model Specifications

After data preprocessing, a binary logistic regression model was fitted to the data using statistical software, since the dependent variable was binary in nature i.e., presence or absence of claim fraud makes binary logistic regression applicable. Also, since the research assumed predictor variables were not correlated, binary logistic regression became a better predictor since it also assumes absence of multicollinearity. This model was suitable for modeling health care fraud since it helped in modeling the probability that a rising claim is fraudulent. Further, the research was able to use both continuous and categorical variables in determining the probability of an upcoming claim being fraudulent. The probability values of a claim being fraudulent, π(x) ranges from 0 to 1.

The logistic regression model which models the log of odds ratio is given by:

logit(x)=βo+β_1 X_1+β_2 X_2+⋯+β_n X_n           …………                        Equation 1

The values for the dependent variable were; Y=1 if the claim is fraudulent and Y=0 if the claim is not fraudulent. This implies that this will be a binomial model with probability of success of π, which is the mean of binomial distribution.

Logit(x)=log⁡(π/(1-π))                  ………………..                                              Equation 2

Making π the subject of the formula, (which is the probability Y=1/X1,  X2,…,Xn) the probability of a claim being fraudulent will be given by:

π(x)=e^(βo+β_1 X_1+β_2 X_2+⋯+β_n X_n )/(1+e^(βo+β_1 X_1+β_2 X_2+⋯+β_n X_n ) )           …………                            Equation 3

Where the vector (β_(0),β_1 〖,…β〗_n)  represents the regression coefficient and X can take the predictor variables.

### Model Assumptions

The model does not require normally distributed data.
The model assumed that predictor variables were not correlated. 

### Methods for achieving Objective 1 and 2

### Data Preprocessing

The study began by checking for missing values in the train dataset and found that there were none. The beneficiary data set was also checked and there were 63,394 missing values in the Date of Death column. This prompted the calculation of an additional column that was named ‘Whether Dead’ that indicated whether an individual was alive (No substituted by 0) or dead (Yes substituted by 1). The Age column was calculated by subtracting the data collection year and the Date of Birth of the beneficiaries. Data types for the beneficiary data were checked and replaced with 1 and 2 with 0 and 1 for No and Yes respectively in the columns showing presence of a chronic condition. The inpatient and outpatient had a lot of variables with missing data. However, another column was added for the number of days a patient was admitted for the inpatient data.
     
Outpatient and inpatient data sets were merged using the outer join (union) as they had many similar columns. The new data set was then merged with beneficiary data details using the inner join with Beneficiary ID as the joining key. The final dataset was then merged with train data that contained fraudulent providers’ details using provider ID as the key for the inner join (intersection). The final data set consisted of 558,218 entries with 57 variables.

For familiarity with the data, frequencies of fraud and non-fraud transactions were checked. 

Numerical columns were imputed with 0 for the missing values then eliminated categories that would not be necessary for prediction such as Dates, diagnosis codes and IDs. Race and Gender were converted to categorical variables. The remaining dataset was aggregated to unique providers and the result was a Tibble of 5,410 entries and 30 variables which was suitable for modelling. The dataset was then normalized using a Scaler preprocessing called the MinMaxScaler method.

### Model Fitting

The combined train data was split into a training set and a validation set with a probability of 0.7 and 0.3. The training set was used to fit a logistic regression model, The dataset was rebalanced to reduce the bias towards non fraudulent claims. 

### Methods for achieving Objective 3

#### Performance Metrics

After training the data using the training set, the validation set was used to test the capability of the Logistic regression model in data prediction by using it as a prediction sample. The probability threshold for fraud was set at 0.5 to improve its sensitivity. In assessing the medical claims, the data was classified as either fraudulent or non-fraudulent. For the logistic predicted model, the study deployed a confusion matrix to compare actual counts against the predicted counts. The confusion matrix is a 2*2 matrix with true negatives (negative values that are predicted as negative), true positives (positive values that are predicted as positive), false negatives (positive values that are predicted as negative) and false positives (negative values that are predicted as positive). The metrics used in performance evaluation included accuracy, that is the sum of true positives and true negatives divided by the total number of predictions, precision which is the number of true positives divided by the sum of true positives and false positives, recall also known as model sensitivity and is calculated by the number of true positives divided by the sum of true positives and false negatives, fall-out which is calculated by false positives divided by the sum of false positives and true negatives, and the AUR-ROC which stands for the area under the receiver operating characteristic curve. The receiver operating curve is the graph of fall-out against recall, that is, the graph of false positives against true positives. The higher the area under the curve the better the model will be at predicting fraud.

An illustration of the confusion matrix is as follows:



|        |                 | Predicted       |                 |
|--------|-----------------|-----------------|-----------------|
|        |                 | Non- Fraudulent | Fraudulent      |
| Actual | Non- Fraudulent | True Negatives  | False Positives |
|        | Fraudulent      | False Negatives | True Positives  |




 ## CHAPTER FOUR: DATA ANALYSIS, PRESENTATION, AND INTERPRETATION


 
 ### Introduction


This section outlines the research findings, showcasing the performance metric outcomes obtained from employing binary logistic regression technique. It demonstrates the model's credibility and feasibility, leading to appropriate conclusions.


## Data Analysis


From the graph in figure 4.2.1, our data, the fraudulent cases were less than the non-fraudulent cases. Out of the total number of claims made, the percentage of fraudulent cases was 38.12% while that of non-fraudulent cases was 61.88%.



![image](https://github.com/user-attachments/assets/8042674c-7567-4695-96ae-43e0734577b4)

Figure 4.2.1: Distribution of claim records



In the figure 4.2.2 below, it was noted that only 9.4% of the service providers were fraudulent while the other 90.6% were non-fraudulent.



![image](https://github.com/user-attachments/assets/a5007a9f-c923-43a6-9b0f-b3d2ac14bc05)

Figure 4.2.2: Potential fraud among providers


  
State-wise distribution of beneficiaries was as follows: Arkansas had the maximum number of beneficiaries while Alaska had the least number of beneficiaries as shown in figure 



![image](https://github.com/user-attachments/assets/c6dad6f3-d7e0-4087-9a95-fda5f53df63d)

Figure 4.2.3: State-wise beneficiary distribution



According to figure 4.2.4 below, Race 1 (Whites/Caucasian) had the highest percentage of beneficiaries and Race 5 (Australoid) had the least percentage of beneficiaries.



![image](https://github.com/user-attachments/assets/a6bcf38a-a894-43ab-a885-b3fdec9bc514)

Figure 4.2.4: Race-wise Beneficiary Distribution



According to figure 4.2.5 below, it is shown that the fraudulent cases were high compared to the non-fraudulent cases in all the procedures. The procedure transfusion of packed cells had the most fraudulent claim counts.



![image](https://github.com/user-attachments/assets/caf5a14d-d1ba-4884-ad7f-bd2edee8ab54)

Figure 4.2.5: Top 10 procedures involved in healthcare fraud



Figure 4.2.6 shows that claim diagnosis 4019 (Unspecified essential hypertension) was highly fraudulent followed by diagnosis 4011 (Benign essential hypertension).


![image](https://github.com/user-attachments/assets/e996601a-654d-4db1-9e65-7906ad0695b2)

Figure 4.2.6: Top 10 Diagnosis involved in Healthcare Fraud



### Binary logistic model
The general fitted binary logistic regression model is as shown below.


 Optimization was terminated with the value at 0.1766880 after 8 iterations



| Dependence variable              | Potential fraud |
|----------------------------------|-----------------|
| Model                            | Logit           |
| Method                           | MLE             |
| Number of observations           | 3787            |
| Degrees of freedom for residual  | 3758            |
| Degrees of freedom for the model | 28              |
| Pseudo R-squared                 | 0.4310          |
| Log-Likelihood                   | -669.09         |
| True LL-Null                     | -1175.9         |
| LLR p-value                      | 1.87e-195       |
| Covariance type                  | Non-robust      |


| Variable                          | Coefficient | Std. Error | Z score   | P-value  |
|-----------------------------------|-------------|------------|-----------|----------|
| Intercept                         | -0.766993   | 0.578034   | 5.852817  | 4.83E-09 |
| Insurance Claim Amount Reimbursed | 3.392273    | 0.66024    | -4.520282 | 6.18E-06 |
| Deductible Amount Paid            | -2.367368   | 0.607397   | 3.320912  | 8.97E-04 |
| Days of Admission                 | 2.75583     | 0.140668   | 0.447706  | 6.54E-01 |
| State                             | 0.097752    | 0.166542   | 1.372727  | 1.70E-01 |
| County                            | 0.392772    | 13.690409  | 0.79903   | 4.24E-01 |
| Number of Months Covered(A)       | -0.102887   | 13.922157  | -0.259819 | 7.95E-01 |
| Number of Months Covered (B)      | -0.125375   | 0.822234   | -0.440217 | 6.60E-01 |
| Alzheimer                         | -0.411113   | 1.366685   | 0.06123   | 9.51E-01 |
| Heart failure                     | 0.799395    | 0.995665   | 0.698964  | 4.85E-01 |
| Kidney Disease                    | -0.169096   | 0.426027   | -1.025596 | 3.05E-01 |
| Cancer                            | -0.742661   | 0.713785   | -0.314605 | 7.53E-01 |
| Obstructive Pulmonary             | 0.426464    | 0.903044   | 0.805802  | 4.20E-01 |
| Depression                        | 1.200192    | 2.060881   | -2.592433 | 9.53E-03 |
| Diabetes                          | -1.188673   | 2.14549    | 0.45342   | 6.50E-01 |
| Ischemic Heart                    | 0.245455    | 0.739077   | -0.640414 | 5.22E-01 |
| Osteoporosis                      | -0.280937   | 0.720675   | 0.094939  | 9.24E-01 |
| Rheumatoid arthritis              | -0.704476   | 0.318729   | 2.632536  | 8.48E-03 |
| Stroke                            | 0.515285    | 0.566608   | -1.025811 | 3.05E-01 |
| IP Annual Reimbursement Amount    | -0.498351   | 0.544542   | 0.145737  | 8.84E-01 |
| Annual Deductible Amount          | -0.859198   | 1.054572   | -0.428809 | 6.68E-01 |
| Annual Reimbursement Amount       | 0.435262    | 1.080475   | 1.312376  | 1.89E-01 |
| OP Annual Deductible Amount       | 0.661865    | 4.613631   | -1.437928 | 1.50E-01 |
| Age                               | -0.843984   | 0.10915    | 0.171069  | 8.64E-01 |
| Whether Dead                      | 0.08648     | 1.016708   | 2.029992  | 4.24E-02 |
| Female                            | 1.052072    | 0.158847   | -1.81977  | 6.88E-02 |
| Mongoloid/Asian                   | -0.109052   | 0.209757   | 1.8527    | 6.39E-02 |
| Negroid/Black                     | 0.380787    | 0.150796   | -1.831006 | 6.71E-02 |
|                                   |             |            |           |          |



### Testing for significance


Using the p-value approach (that is, when the p-value is less than the level of confidence, the variable is significant), our significant variables were Insurance claim amount reimbursed, the deductible amount paid, whether the individual was alive or dead, as well as the chronic conditions Depression and rheumatoid arthritis.

Insurance claim amount reimbursed has a co-efficient of 3.392273 implying an increase in the reimbursement amount will lead to an increase in fraudulent cases. Similarly, variable X_13 (Chronic condition depression) has a co-efficient of 1.200192 implying it positively affects the model and so do variable X_24 (Whether dead or alive). The other variables, X_1 (Deductible amount paid) and X_17 (Chronic condition rheumatoid arthritis), negatively affect the model hence a unit increase in any of these variables implies a decrease in potential fraudulent claims. 

The fitted model was:

logit(x)= -0.766993+3.392273X_1-2.367368X_2+1.200192X_13  -0.704476X_17  + 0.08648 X_24

logit(x)=log⁡(π/(1-π)) 

log⁡(π/(1-π)) = -0.766993+3.392273X_1-2.367368X_2+1.200192X_13  -0.704476X_17  + 0.08648 X_24

π(x)=〖e 〗^(-0.766993+3.392273X_1-2.367368X_2+1.200192X_13-0.704476X_17+ 0.08648X_24 )/(1+〖e 〗^(-0.766993+3.392273X_1-2.367368X_2+1.200192X_13-0.704476X_17+ 0.08648X_24 ) ) 


### Performance Metrics


The Area under the Curve was high with a value of 92% after training the model.



![image](https://github.com/user-attachments/assets/eb4beab4-dc7f-43b9-8184-9edfb94cace1)

Figure 4.4.1: Receiver Operating Characteristic



The confusion matrix was given by:



![image](https://github.com/user-attachments/assets/dd02fb43-c5ba-425d-8292-00c6e38f4e03)




The model predicted 1361 claims as non-fraudulent when they were non-fraudulent and correctly predicted 95 fraudulent claims.

The accuracy of the model was 89.71%. This means that the model had 89.71% correct predictions. The model’s specificity was 92.52%. The model had high specificity meaning that it was successful at avoiding false alarms or false positives when the true label was negative. The model’s sensitivity was 62.5% which suggested that the model is not as strong at correctly identifying instances of fraudulent claims as compared to the non-fraudulent claims as shown in figure 4.4.2 below.



![image](https://github.com/user-attachments/assets/1cd37eb3-6116-484d-945e-af233fdbb008)

Figure 4.4.2: True Positive Rate vs True negative Rate



 The F1-score was 53.22% suggesting that the model maintains a reasonable balance between precision and recall, but it was skewed toward specificity. Overall, the model performed well.





## CHAPTER FIVE: CONCLUSION



### Conclusion


The insurance industry faces health insurance fraud, which affects medical pricing and access across all socioeconomic groups. The regulatory bodies have reported more fraudulent health insurance claims, emphasizing the need to solve this issue quickly. Fraudulent activities cost insurance companies a lot of money and burden public healthcare programs, which hurts patients. The research sought to illuminate the issue's magnitude, its effects, and the need for effective diagnosis and prevention.

Logistic regression was used to model health insurance claims fraud in the research. Available data was used to create a logistic regression model, trained the algorithm, and assessed its performance using key metrics. This study showed promise for detecting fraudulent claims using contemporary data analytics and machine learning. An approach which could greatly benefit the insurance industry.

Ultimately, the fight against health insurance fraud is constant and ever-changing. The methods and technologies used to detect and prevent fraud must evolve. The Kenyan insurance industry can protect policyholders, limit costs, and preserve the healthcare system by following these rules and remaining alert. To provide affordable healthcare for all is a moral and economic obligation.


### Recommendations


Health insurance fraud remains to be a challenge that requires a varied approach to detection and prevention. Insurance companies, government agencies, and healthcare providers must collaborate better. This cooperative effort should involve fraud data exchange to improve fraud detection and investigation.

Second, insurance companies must urgently invest in data analytics and machine learning. These systems may detect subtle trends and abnormalities that signal fraud, greatly improving fraudulent claim detection. By adopting these changes, insurance companies can combat rising fraud.

Moreover, insurance professionals, claims adjusters, and investigators need ongoing training. These apps should keep users up to date on the latest fraud detection and prevention methods. Fraud prevention requires seeing red flags and conducting thorough investigations.

Additionally, real-time monitoring mechanisms must be implemented. These technologies can detect irregularities and fraud in claim data as it is processed. Real-time monitoring reduces false claim costs by acting quickly.

Public awareness initiatives should inform policyholders about insurance fraud and encourage them to report suspicious conduct. Informing the public can help deter false claims. These measures can instill policyholders' responsibility and participation in insurance system integrity.

 
## GLOSSARY
IRA-Insurance Regulatory Authority

NHIF-National Hospital Insurance Fund

AI-Artificial Intelligence

IFIU-Insurance Fraud Investigation Unit

NAIC-National Association of Insurance Commissioners

AUROC-Area Under the Receiver Operating Characteristic Curve

NHCAA-National Health Care Anti-Fraud Association

AKI- Association of Kenya Insurers

DT- Decision Trees

SVN-Support Vector Machine

KNN- K-Nearest Neighbor

RF- Random Forest

NN-Nearest Neighbor

ML-Machine Learning

ANN-Artificial Neural Network

GLM- Generalized Linear Model

MLE- Maximum likelihood Estimator



## REFERENCES


Alam MN, Podder P, Bharati S, Mondal MRH (2021). Effective machine learning approaches for credit card fraud detection. Cham: Springer.

Alushula, P. (2021, August 23). NHIF audit report uncovers fraudulent hospital claims. Business Daily.

Arnold, K. F., Davies, V., de Kamps, M., Tennant, P. W. G., Mbotwa, J., & Gilthorpe, M. S. (2020). Reflection on modern methods: generalized linear models for prognosis and intervention—theory, practice, and implications for machine learning. International Journal of Epidemiology, 49(6), 2074–2082. 

Association of Kenyan Insurers. (2021). Information Paper on Insurance Fraud. 

Bhatore, S., Mohan, L. & Reddy, Y.R. Machine learning techniques for credit risk evaluation: a systematic literature review. J BANK FINANC TECHNOL 4, 111–138 (2020).

Bin Sulaiman, R., Schetinin, V. & Sant, P. (2022) Review of Machine Learning Approach on Credit Card Fraud Detection. 

Carmi, G., & Segal, S. Y. (2014). Mobile Security: A Review of New Advanced Technologies to Detect and Prevent E-Payment Mobile Frauds. Int. J. Comput. Syst, 292(4), 2394-1065.

Darwish S. M. (2019): An intelligent credit card fraud detection approach based on semantic fussion of two classifiers. Soft Compute

Henckaerts, R., Côté, M. P., Antonio, K., & Verbelen, R. (2021). Boosting insights in insurance tariff plans with tree-based machine learning methods. North American Actuarial Journal, 25(2), 255-285.

International Social Security Association (ISSA), 4 July 2022. “Detecting fraud in health care through emerging technologies.”

IRA. (2020). INSURANCE INDUSTRY ANNUAL REPORT 2020. 

Kumaraswamy N, Markey MK, Ekin T, Barner JC, Rascati K. 2022 Jan 1; 19. Healthcare Fraud Data Mining Methods: A Look Back and Look Ahead. Perspect Health Inf Manag. 

Lu, Y. C. (2021). Relational Outlier Detection: Techniques and Applications (Doctoral dissertation, Virginia Tech).

Maher, P. (2020). The Seven Most Popular Machine Learning Algorithms for Online Fraud Detection and Their Use in SAS®. 

Mambo, S., & Moturi, C. (2022). Towards Better Detection of Fraud in Health Insurance Claims in Kenya: Use of Naïve Bayes Classification Algorithm. East African Journal of Information Technology, 5(1), 244-255.

Makkar, R. R., Saraswat, B. K., & Pandey (2023). M. Predicting Health Expenses Using Linear Regression. 

Mitic, I. (2023). The Fraudster Next Door: 30 Insurance Fraud Statistics. Fortunly.

NAIC. (2022, December 19). Insurance Fraud. Content.naic.org.

Samantha Rohn. (June 23, 2022). “The Future of Insurance Fraud Detection is Predictive Analytics”. whatfix.com. 

Sriram Sasank JVV, Sahith GR, Abhinav K, Belwal M (2019). Credit card fraud detection using various classification and sampling techniques: a comparative study.

Taitsman, J. K., Grimm, C. M., & Agrawal, S. (2019). Protecting patient privacy and data security. New England Journal of Medicine, 368(11), 977-979.

Villegas-Ortega, J., Bellido-Boza, L. & Mauricio, D. Health Justice 9, 26 (2021). Fourteen years of manifestations and factors of health insurance fraud, 2006–2020: a scoping review.

Yadav, V. (2021). Prediction and detection analysis of bank credit card fraud using regression model: novel approach. Prediction and detection analysis of bank credit card fraud using regression model: novel approach. Turkish Online Journal of Qualitative Inquiry (TOJQI), 12(7), 14085–14098. 

Zhang C, Xiao X, Wu C, (2020): Medical Fraud and Abuse Detection System Based on Machine Learning.  International Journal of Environmental Research and Public Health.



## APPENDICES


### States and their Codes





| State                     | Code | State             | Code |
|---------------------------|------|-------------------|------|
| Alabama                   | 01   | Nebraska          | 31   | 
| Alaska                    | 02   | Nevada            | 32   |
| Arizona                   | 04   | New Hampshire     | 33   |      
| Arkansas                  | 05   | New Jersey        | 34   | 
| California                | 06   | New Mexico        | 35   |
| Colorado                  | 08   | New York          | 36   |
| Connecticut               | 09   | North Carolina    | 37   | 
| Delaware                  | 10   | North Dakota      | 38   | 
| District of Columbia      | 11   | Ohio              | 39   | 
| Florida                   | 12   | Oklahoma          | 40   | 
| Georgia                   | 13   | Oregon            | 41   |   
| Hawaii                    | 15   | Pennsylvania      | 42   | 
| Idaho                     | 16   | Puerto Rico       | 72   |
| Illinois                  | 17   | Rhode Island      | 44   |
| Indiana                   | 18   | South Carolina    | 45   |
| Iowa                      | 19   | South Dakota      | 46   |
| Kansas   KS               | 20   | Tennessee         | 47   | 
| Kentucky                  | 21   | Texas             | 48   |
| Louisiana                 | 22   | Utah              | 49   | 
| Maine       ME            | 23   | Vermont           | 50   | 
| Maryland                  | 24   | Virginia          | 51   | 
| Massachusetts             | 25   | Virgin Islands    | 78   | 
| Michigan                  | 26   | Washington        | 53   | 
| Minnesota                 | 27   | West Virginia     | 54   | 
| Mississippi               | 28   | Wisconsin         | 55   | 
| Missouri                  | 29   | Wyoming           | 56   | 
| Montana                   | 30   |                   |      |
|                           |      |                   |      |


### Top 10 diagnosis involved in health fraud and their codes


4019- Unspecified essential hypertension

4011- Benign essential hypertension

2720- Pure hypercholesterolemia

2724- Other and unspecified hyperlipidemia

2722- Mixed hyperlipidemia

2721- Pure hyperglyceridemia

2723- Hyperchylomicronemia

78659- Other chest pain

42731- Atrial fibrillation

78650- Chest pain, unspecified



### Top 10 procedures involved in health care frauds and their codes


9904.0- Transfusion of packed cells

8154.0- Total knee replacement

66.0- Percutaneous transluminal coronary angioplasty [PTCA]

3893.0- Venous catheterization, not elsewhere classified

3995.0- Hemodialysis

4516.0- Esophagogastroduodenoscopy [EGD] with closed biopsy

3722.0- Left heart cardiac catheterization

8151.0- Total hip replacement

8872.0- Diagnostic ultrasound of heart

9671.0- Continuous invasive mechanical ventilation for less than 96 consecutive hours




### Races and their Codes
1-	White/Caucasian

2-	Mongoloid/Asian

3-	Negroid/Black

5-	Australoid
