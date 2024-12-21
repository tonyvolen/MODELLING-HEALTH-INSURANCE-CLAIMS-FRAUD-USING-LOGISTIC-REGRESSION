# MODELLING HEALTH INSURANCE CLAIMS FRAUD USING LOGISTIC REGRESSION: AN APPLICATION OF MACHINE LEARNING


### Abstract

The insurance industry is plagued by health insurance fraud in Kenya. Health insurance fraud involves deception and misrepresentation to get illicit health insurance benefits. This approach reduces affordable medical care for all socioeconomic groups. The analysis highlights the rising frequency of fraudulent claims and insurance and public healthcare program losses. It stresses how important it is to resolve these issues to preserve the healthcare system. This research seeks to detect fraudulent health insurance claims using logistic regression and stresses the importance of detecting and eradicating fraud to protect policyholders, minimize costs, and retain insurance sector trust. The research approach includes data collection, preparation, model fitting, and performance evaluation. The model's performance is measured by accuracy, precision, recall, fall-out, and AUR-ROC. The study fills gaps in fraud detection, data validation, and fraudulent intent research. It uses machine learning to detect health insurance fraud to improve healthcare efficiency and cut costs aiming to help the Kenyan and international insurance industry comprehend the rise in health insurance fraud.

## TABLE OF CONTENTS



## TABLE OF FIGURES




## CHAPTER ONE: INTRODUCTION


### Background of the Study


One of the primary challenges confronting the insurance sector pertains to the prevalence of fraudulent claims and instances of abuse. Health insurance fraud is an act of deception or intentional misrepresentation to obtain illegal benefits concerning the coverage provided by a health insurance company (J Villegas-Ortega et.al, 2021). The Insurance Fraud Investigation unit reported a rise in fraudulent cases in Kenya from 83 in 2019 to 123 in 2020 (IRA, 2020). Fraudulent activity within the health sector has a direct impact on the availability of affordable healthcare for individuals across all socioeconomic strata. Fraudulent activities by contracted hospitals are reported to cost The National Health Insurance Fund (NHIF) up to 16.5 billion shillings annually (Alushula, 2021). Various surveys have reported that about 10% of health insurance claims are fraudulent (Mitic, 2023). The high reported losses show the severe underreporting of fraudulent health insurance claims.

Health insurance fraud has undergone significant transformations in the 21st century, exhibiting a diverse range of profiles that span from rudimentary fraudulent schemes to intricate networks. These fraudulent claims are resulting in financial losses for companies, imposing significant medical expenses on ordinary citizens, and contributing to the decline of the Kenyan economy. NHIF reported that some of its contracted hospitals exaggerated their bed numbers by up to five times to defraud it (Alushula, 2021). Despite the efforts made by the government and other relevant stakeholders to address this criminal activity, there is a pressing need for additional measures to effectively close the existing loopholes exploited for fraudulent purposes. The government created The Insurance Fraud Investigation Unit (IFIU) in 2011 as a measure to examine instances of fraudulent activity (Insurance Fraud Investigations Unit (IFIU), 2020). The unit receives a limited number of cases annually, and because of thorough investigative procedures, only a small portion of these cases are ultimately concluded, with appropriate advice provided to the complainants.

The past few decades have seen an immense growth in Artificial Intelligence (AI). Most notably, machine learning techniques have been incorporated by insurance firms to detect fraudulent claims (Bhatore, 2020). Researchers have explored and still are exploring various machine learning techniques to improve the accuracy of fraud detection, especially in the medical insurance sector.

#### Statement of the Problem

The prevalence of misuse and fraud in health insurance is currently causing an increase in anxiety within the insurance sector. Insurers face a substantial issue in effectively assessing and pricing risks due to the rising costs connected with fraudulent activity (NAIC, 2022). As a result, these insurers' returns on equity have decreased and premiums have increased. Due to fraudulent actions, public programs, such as the National Health Insurance Fund (NHIF), suffer severe negative repercussions. Therefore, modelling health insurance fraud is important for reducing the burden and costs of compliance for law-abiding healthcare institutions, practitioners, and their patients. 

 #### Research Objectives
 
##### General Objective
To model health insurance claims fraud using logistic regression
##### Specific Objectives
1.	To fit a logistic regression model on health insurance claims fraud data.
2.	To train the logistic regression algorithm to generate an appropriate model for assessing health insurance claims.
3.	To evaluate the performance of the model using AUR-ROC and confusion matrix metrics.
	
#### Significance of the Study

This report aims to substantiate the significance of fraud detection as a primary function of health insurers and as a domain for prospective investigation. Undoubtedly, fraud imposes a financial burden on the health care system without yielding any advantages. Therefore, this study will play a crucial role in fostering a more streamlined health care system.

The implementation of the research proposal aims to facilitate a reduction in insurance claim fraud within the health insurance system, resulting in cost savings.
The research also helps in knowing factors that lead to rise on insurance health claims fraud.

#### Scope

The research project will focus on how the medical insurance industry can model fraudulent claims and therefore identify potential fraud cases that will cost them a lot of money. They then can employ ways to seal loopholes that lead to fraud. Consequently, the cost of healthcare in a developing country like Kenya can be manageable since fraud inflates the cost of healthcare.

 

## CHAPTER TWO: LITERATURE REVIEW


### Introduction 


Fraud and abuse take place at many points in the healthcare system. Doctors, hospitals, nursing homes, diagnostic facilities, medical equipment suppliers, and attorneys have been cited in scams to defraud the system. The National Health Care Anti-Fraud Association (NHCAA) estimates that the financial losses due to health care fraud are $68 billion, or as high as $300 billion (III, 2022). Health insurance fraud committed by patients may be an increasing problem given the number of underinsured and uninsured people.

According to the research done by the Insurance Information Institute, common types of Health Care frauds include Kickback schemes where actors might improperly pay for or waive the client’s out-of-pocket expense to make up for that cost in additional business. Unbundling and fragmentation of procedures where separate claims are created for services or supplies that should be grouped together. When it comes to submitting claims not only improper coding practices can be fraudulent, but also care providers can try to submit the same claim multiple times, to get paid two times for performing one action. Further, phantom billing for services not rendered to clients takes place and this is also a form of fraud alongside billing unnecessary services that could lead to client harm. Claims are submitted for a service provided based on a stated diagnosis. These diagnoses can also be manipulated i.e., a patient can get a certain diagnosis while that diagnosis is not actually true. This type of fraud can be done to falsely prescribe certain medicines to a patient, for example.

Fraud detection and prevention are crucial in the health insurance industry to control costs, protect policyholders, and ensure the financial stability of insurers. These practices promote fairness and equity in insurance benefits, comply with regulations, and safeguard sensitive data. By stopping fraudulent activities, insurers can streamline claim processing, improve healthcare quality, and maintain trust with policyholders and stakeholders, ultimately contributing to a sustainable and reliable healthcare system. 

Fraud and abuse are widespread and very costly to Kenya's health-care system. Even though the medical business premiums have more than doubled in the last 4 years, the industry registered an average loss of Kshs.33,394,812 per year (33%) over the 4-year period (AKI, Health Insurance Fraud Survey Report, 2018). According to the Insurance Industry Annual Report given by IRA, 2020, IFIU reported a rise in medical fraudulent claims from 9 cases in 2019 to 12 fraud cases in 2020 which costed a total of Ksh.1511290. 

The Insurance Industry Annual Report (2018) by AKI identifies prevalent forms of insurance fraud in Kenya, including manipulating diagnoses, substituting branded drugs with generic ones, excessive servicing, swapping memberships, splitting fees, altering invoices or claims, and failing to disclose pre-existing medical conditions. Health frauds among other factors have contributed to the collapse of some health Insurance companies in Kenya.

Many health insurance companies in Kenya have put up various technologies and strategies to fight fraud such as planned, targeted audits, random audits, whistleblowing, biometric systems, application of data mining such as classification algorithms (Naïve Bayes, Decision Tree and K-Nearest Neighbour) (Mambo, S., & Moturi, C. (2022)). It is estimated that 25% of insurance industry income is fraudulently claimed and 35% of medical claims are fraudulent according to AKI report produced in 2018. The implementation of biometric based technology for authentication of insured patients has been adopted to curb rising cases of fraud in medical insurance with 60-80% coverage.  

#### Application of Machine Learning in Health Insurance Fraud Detection

It has proved a necessity for insurance firms to implement effective measures to detect any cases of fraudulent claims before claims are settled. Some of the notable methods identified to help detect fraud in medical insurance includes Decision Trees (DTs), Random Forest (RF), Neural Network (NN), Support Vector Machine (SVM), and k-nearest neighbors (KNN). Traditionally speaking, in the healthcare anomaly detection area, machine learning applications can be divided into supervised learning and unsupervised learning (Zhang C, Xiao X & Wu C., 2020). Basically, supervised learning needs label of data for training, while unsupervised learning does not. Classification trees and decision trees (DTs) are binary trees that classify data into predetermined groups. The tree is constructed using splitting rules in a recursive manner from top to bottom. These rules are often univariate, but they can employ the same variable in zero, one, or several splitting rules.

The fact that DTs do not output the likelihood of classifying a claim as valid or fraudulent, (and as a result do not distinguish across claims in the same classification), is a significant drawback (Maher, 2020). There is no evidence to support the claim that decision trees developing algorithms will reduce risk by not using previous rules when determining future rules but there is no evidence to suggest that this will reduce classiﬁcation or prediction accuracy ability.

DT technique is the foundation of the ML-based random forest approach, which is frequently employed to tackle various regression and classification issues. It helps in highly accurate output prediction in datasets which are large. Random forest aids in forecasting the average mean of other trees' output. The precision of the result increases as the number of trees increases. The decision tree algorithm has a number of disadvantages that the random forest approach helps to overcome (Darwish, 2019). Additionally, it reduces dataset lifting, boosting precision. The RF approach is quick and efficient for managing unbalanced and big volumes of datasets (Sulaiman et al., 2022). However, the random forest has drawbacks when it comes to training different datasets, particularly for regression issues. Although the random forest algorithms are quite good at identifying the class of regression issues, Bin Sulaiman, (2022) claims that they have a number of drawbacks when it comes to real-time medical insurance fraud detection. In real-time applications, random forest algorithms perform less well, making predictions take longer since the training process is slower. Consequently, in order to effectively detect fraud in real-world datasets, a large volume of data is needed, and random forest algorithms lack the capability of training the datasets effectively and making predictions. 

Artificial Neural Network (ANN) is an ML algorithm which functions similarly to the human brain. ANN is based on two types of methods: supervised method and unsupervised method. The Unsupervised Neural Network is widely used in detecting fraud cases since it has an accuracy rate of 95% (Maher, P. 2020). The unsupervised neural network attempts to find similar patterns between fraudulent claims settled previously and the current claim reported. Suppose the details found in the current claims are correlated with the previous claims, then a fraud case is likely to be detected. ANN methods are highly fault tolerant and is a viable option for the identification of medical insurance fraud due to its high processing speed and efficiency. However, when combined with a variety of algorithms and functions, it has generated fewer good results. Each of these functions is deficient in some way.

KNN is an example of a supervised machine learning technique useful for problem classification and regression analysis. It is a useful technique for supervised learning. It aids in enhancing detection and lowering the rate of false positives. In order to prove the existence of medical insurance fraud, it employs a supervised technique (Alam et al., 2021). The memory-intensive algorithm KNN scales up properties of non-essential data. When a big amount of data is entered, KNN algorithm's performance declines as data volume increases (Bin Sulaiman et al., 2022). These limitations consequently affect the accuracy and recall matrix in the identification of insurance fraud. Linear models have been previously used to predict and model the medical expenses for new customers. This can be in turn be used in future to detect anomalies in the expense using linear regression models and machine learning algorithms that are trained to estimate possible medical expense (Makkar et al., 2023). These algorithms through deep learning and statistical linear models offer transparency since they are free from human interventions. They mainly depend on large data that is used to primarily train the algorithm before it is tasked to come up with the said expense.

Multiple (linear) regression models calculate the expected value E (*) of one variable Y (the 'dependent' or 'outcome' variable) from a linear combination of several observed covariates X1; Xn (the 'independent' or 'explanatory' variables, or just 'predictors').

#### Generalized Linear Models

A class of statistical models known as generalized linear models (GLMs) expands the classic linear regression model to consider a larger variety of data distributions and response types. The response variable in GLMs may have one of several distributions, including the Gaussian (normal), binomial, Poisson, or gamma distributions. The response variable's predicted value and the predictors are connected by the link function. It converts the linear combination of predictors into the proper scale of the distribution of the response variable. 

By including a link function and a particular distribution assumption, GLMs offer a flexible framework for modeling the relationship between the dependent variable and one or more predictors (Arnold et al., 2020). In a GLM, parameter estimates, and their standard errors give information about the effect, uncertainty, and statistical relevance of all variables. Which in this case GLM outperforms other machine learning techniques such as decision trees (Henckaerts et al., 2021). The coefficients βhat₀, βhat₁, βhat₂, ..., βhatn for a given GLM are typically obtained via a statistical process known as maximum likelihood estimation. Although GLMs are conceptually straightforward to comprehend and apply, as the number of covariates increases, estimation of their parameters without the aid of computational equipment quickly becomes impracticable. Therefore, a revolution in data analytics was made possible by the development of programmable desktop computers in the 1980s and 1990s since it was now possible to do the intricate matrix inversions necessary quickly and automatically for generalized linear modeling.

  Fraud results in the creation of outliers due to non-uniformity. It is a crucial component of fraud prediction for prediction models to determine the likely value (or risk) of an outcome given information from one or more 'predictors'. The GLM model is therefore paramount in modelling health insurance fraud by narrowing down to only factors relevant to healthcare insurance. It is also important to evaluate GLM from a business point by using business metrics used in insurance tariffs evaluation.
  
Generally, these supervised machine learning algorithms are heavily dependent on training datasets. Because of the complicated nature of the healthcare sector, a dataset is usually not comprehensive, this therefore causes the result to be seriously over-fitted in real-world scenarios hence the GLM models are preferred. Types of GLMs commonly used in fraud modelling are logistic regression, Poisson regression, Negative Binomial Regression and Gamma Regression. Logistic regression is typically used when the response variable is binary (fraud vs. non-fraud). The logistic link function transforms the linear combination of predictors into the likelihood of fraud (Maher, 2020). This model calculates the odds ratio for each predictor and displays how the probability of fraud changes when the predictor is altered by one unit. Poisson regression is appropriate when the response variable contains count data, such as the number of erroneous claims. The Poisson distribution and log link function are used to model the relationship between the predictors and the anticipated number of fraud cases. Poisson regression assumes that the mean and variance of the response variable are both equal. In the case of the Negative Binomial regression, the assumption of equal mean and variance is relaxed in this extension of Poisson regression. It is employed when over dispersion, or when the variance is greater than the mean, is present in the count data. To account for the additional variation, negative binomial regression models the link between the predictors and the anticipated fraud count. Gamma regression can be used to model continuous, positively skewed response variables with a gamma distribution, such as medical expenses or claim amounts. Gamma regression takes into consideration the unique distributional characteristics of these variables, enabling more precise modeling and forecasting of the costs associated with fraud.

#### Advantages of Logistic Regression in Detecting Health Insurance Fraud

Logistic regression models can handle binary response variable distributions, making them suitable for modeling data encountered in health insurance fraud detection, such as fraud vs. non-fraud.

 Logistic regression models also offer interpretable coefficients that enable analysts to comprehend the degree and direction of each predictor variable's impact on the response variable. Finding the causes of health insurance fraud requires being able to understand data. The direction of the link between the predictor factors and the response variable (in this case, health insurance fraud) is represented by the coefficients provided by the model. A positive coefficient indicates a positive association, indicating that the likelihood of fraud increases as the predictor variable rises. A negative coefficient, on the other hand, denotes a negative association, where a rise in the predictor variable is connected to a fall in the risk of fraud. It is essential to comprehend the direction of the influence when figuring out what causes or discourages health insurance fraud. Therefore, Logistic regression models reveal not just the direction but also the size or intensity of each predictor variable's influence on the response variable (Maher, 2020). The size of the coefficient represents how much the likelihood of fraud is affected by a unit change in the predictor variable. Smaller coefficients represent a relatively weaker link, while larger coefficients show a more significant influence of the predictor variable on fraud. Analysts can determine the most important causes of health insurance fraud by looking at the magnitude of the coefficients.
 
Analysts can learn more about the elements that have a big impact on health insurance fraud by having coefficients that are easy to interpret. This knowledge can help with decision-making procedures like creating fraud detection techniques, effectively allocating resources, and putting in place targeted measures to reduce fraud risks. Additionally, the interpretability of the model’s coefficients makes it easier to validate and replicate study findings in identifying health insurance fraud and to communicate findings more clearly.

Logistic Regression Models enable testing of hypotheses regarding the importance of predictor factors (Henckaerts et al., 2021). These aids in identifying the factors that statistically significantly predict fraud and evaluating the relative contributions of each factor. Additionally, the model can incorporate categorical information in the fraud detection model.

### Research gap

 There is often insufficient detail on the underlying methodologies and data samples that lead to fraud estimates, which may be due to different purposes of these reports or the need to obscure the details of fraud detection methods to prevent fraudulent operators from responding to existing methods. The main gaps identified are validation of existing methods and proof of intent to commit fraud in the studies analyzed. The lack of automation in medical processing within the Kenyan medical insurance sector creates an opening for exploring the implementation of machine learning for local medical claim processing using available data. Therefore, this research project is intended to provide suitable methodology on supervised machine learning on detecting health insurance claim fraud.



## CHAPTER THREE: METHODOLOGY

### Introduction

This chapter elaborates the methodology that was used to accomplish the already established research objectives sequentially. The research design, data collection and analysis, are briefly illustrated.

#### Data Source

The data that was used in this study is Medicare insurance data posted on Kaggle, an online portal for community of data scientists. The dataset captures individuals from age 26 to 100 and was specific whether they suffered from chronic diseases, and if so, which ones. The study used this data as the people within this age range are more likely to have health insurance. The claim behavior patterns were studied for a period of two years.

#### Sampling Design

Data used in this study was from a secondary data source. This data was sampled to be a representative of Healthcare Insurance holders. The research specifically focused on outpatient data for patients who visited hospital to receive treatment but were not admitted in them, inpatient data for patients who visited hospital to receive treatment and were admitted, beneficiary data for claim beneficiaries and their identifying characteristics such as race, religion, region and whether they suffered from chronic illnesses such as heart failure.

- Outpatient data
  
The data here is for patients who visited hospital to receive treatment but were not admitted in them. The datasets were split into train and test datasets, each with elaborate claim features, describing claim details. This dataset had 517,737 records and 27 feature columns.

- Beneficiary data
   
The data here was for beneficiaries of each claim and their identifying characteristics such as race, religion, region and whether they suffered from chronic illnesses such as heart failure. This data was also split into train and test datasets with 138,556 records and 25 feature columns. For sensitivity of the information provided, most of the identifying features were coded for data privacy of the involved individuals for example, Race is a discrete random variable taking the values 1 to 5.

- Train data
  
This file contained a unique provider identifier and the class label, potential fraud, labelled Yes and No, coded as 1 and 0 respectively. For a Fraudulent claim, the claim is reviewed and if it is a No then the claim is correct to be paid. Data was aggregated at Unique claim ID for each claim made by the provider. The unique providers handling each claim were 5410.

#### Data Assumptions

There were various assumptions taken in the model and these include:
1.	Claims can be filed more than once by a client.
2.	All claim payments are fulfilled by insurance firms shortly after they have been filed.
3.	All claims result in some payments.

#### Model Specifications

After data preprocessing, a binary logistic regression model was fitted to the data using statistical software, since the dependent variable was binary in nature i.e., presence or absence of claim fraud makes binary logistic regression applicable. Also, since the research assumed predictor variables were not correlated, binary logistic regression became a better predictor since it also assumes absence of multicollinearity. This model was suitable for modeling health care fraud since it helped in modeling the probability that a rising claim is fraudulent. Further, the research was able to use both continuous and categorical variables in determining the probability of an upcoming claim being fraudulent. The probability values of a claim being fraudulent, π(x) ranges from 0 to 1.

The logistic regression model which models the log of odds ratio is given by:

logit(x)=βo+β_1 X_1+β_2 X_2+⋯+β_n X_n           …………                        Equation 1

The values for the dependent variable were; Y=1 if the claim is fraudulent and Y=0 if the claim is not fraudulent. This implies that this will be a binomial model with probability of success of π, which is the mean of binomial distribution.

Logit(x)=log⁡(π/(1-π))                  ………………..                                              Equation 2

Making π the subject of the formula, (which is the probability Y=1/X1,  X2,…,Xn) the probability of a claim being fraudulent will be given by:

π(x)=e^(βo+β_1 X_1+β_2 X_2+⋯+β_n X_n )/(1+e^(βo+β_1 X_1+β_2 X_2+⋯+β_n X_n ) )           …………                            Equation 3

Where the vector (β_(0),β_1 〖,…β〗_n)  represents the regression coefficient and X can take the predictor variables.

#### Model Assumptions

The model does not require normally distributed data.
The model assumed that predictor variables were not correlated. 

#### Methods for achieving Objective 1 and 2

##### Data Preprocessing

     The study began by checking for missing values in the train dataset and found that there were none. The beneficiary data set was also checked and there were 63,394 missing values in the Date of Death column. This prompted the calculation of an additional column that was named ‘Whether Dead’ that indicated whether an individual was alive (No substituted by 0) or dead (Yes substituted by 1). The Age column was calculated by subtracting the data collection year and the Date of Birth of the beneficiaries. Data types for the beneficiary data were checked and replaced with 1 and 2 with 0 and 1 for No and Yes respectively in the columns showing presence of a chronic condition. The inpatient and outpatient had a lot of variables with missing data. However, another column was added for the number of days a patient was admitted for the inpatient data.
     
Outpatient and inpatient data sets were merged using the outer join (union) as they had many similar columns. The new data set was then merged with beneficiary data details using the inner join with Beneficiary ID as the joining key. The final dataset was then merged with train data that contained fraudulent providers’ details using provider ID as the key for the inner join (intersection). The final data set consisted of 558,218 entries with 57 variables.

For familiarity with the data, frequencies of fraud and non-fraud transactions were checked. 

Numerical columns were imputed with 0 for the missing values then eliminated categories that would not be necessary for prediction such as Dates, diagnosis codes and IDs. Race and Gender were converted to categorical variables. The remaining dataset was aggregated to unique providers and the result was a Tibble of 5,410 entries and 30 variables which was suitable for modelling. The dataset was then normalized using a Scaler preprocessing called the MinMaxScaler method.

#### Model Fitting

The combined train data was split into a training set and a validation set with a probability of 0.7 and 0.3. The training set was used to fit a logistic regression model, The dataset was rebalanced to reduce the bias towards non fraudulent claims. 

#### Methods for achieving Objective 3

##### Performance Metrics

After training the data using the training set, the validation set was used to test the capability of the Logistic regression model in data prediction by using it as a prediction sample. The probability threshold for fraud was set at 0.5 to improve its sensitivity. In assessing the medical claims, the data was classified as either fraudulent or non-fraudulent. For the logistic predicted model, the study deployed a confusion matrix to compare actual counts against the predicted counts. The confusion matrix is a 2*2 matrix with true negatives (negative values that are predicted as negative), true positives (positive values that are predicted as positive), false negatives (positive values that are predicted as negative) and false positives (negative values that are predicted as positive). The metrics used in performance evaluation included accuracy, that is the sum of true positives and true negatives divided by the total number of predictions, precision which is the number of true positives divided by the sum of true positives and false positives, recall also known as model sensitivity and is calculated by the number of true positives divided by the sum of true positives and false negatives, fall-out which is calculated by false positives divided by the sum of false positives and true negatives, and the AUR-ROC which stands for the area under the receiver operating characteristic curve. The receiver operating curve is the graph of fall-out against recall, that is, the graph of false positives against true positives. The higher the area under the curve the better the model will be at predicting fraud.

An illustration of the confusion matrix is as follows:

