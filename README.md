# **Heart Attack Predictive Analysis**

---

Heart disease is thee leading cause of death in the United States, according to the Centers for Disease Control and Prevention (CDC). Up to 800,000 people die from the condition each year. That means almost 1 in every 3 deaths in the United States is a result of heart disease. There are several types of heart disease with Coronary Artery Disease being the most common one. Coronary Artery Disease occurs as a result of atherosclerosis; this condition develops as a result of plaque buildup in the walls of the arteries. This buildup narrows the arteries making blood flow difficult and resulting in clots which then leads to heart attack or stroke.

Early detection is vital in heart disease treatment; according to the CDC 200,000 heart disease deaths could be prevented each year if caught early on. Cordonary Artery Disease can be attributed to preventable factors like obesity, poor physical activity, heavy drinking, eating unhealthy foods and not keeping blood pressure and cholesterol under control. When a person has risk factors, their doctor can refer them to a cardiologist for further testing. These tests include a coronary calcium scan—a CT scan that takes pictures of the arteries to check for calcified plaque in the arteries, and a CT angiogram, which uses X-rays to provide detailed pictures of the heart and the blood vessels to look for disease. 

---

## **Overview:**

---

In this project I will perform an exploratory data analysis as well as implement machine learning (ML) approaches on the Cleveland Clinic Heart Disease Dataset. I will then compare the models and see which one is best for classifying whether a patient has heart disease or not. This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The "goal" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4. Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0).

The target output in this dataset has two unique values, 0 indicating no heart disease and 1 indiciating heart disease. We will therefore use binary classfication models to predict whether individuals have heart disease based on a series of attributes.

</br>

> *To veiw the project in its entirety, see 'Project Files' below for veiwing options. Only project output images are shown in this README file.*

---

## Table of Contents:

---

- [Project Files](#Project-Files)
- [Data Sources](#Data-Sources)
- [Steps in Analysis](#Steps-in-Analysis)
- [Data Sources](#Data-Sources)
- [Data Gathering and Cleaning](#Data-Gathering-and-Cleaning)
  * [Global Data](#Global-Data)
  * [United States Data](#United-States-Data)
- [Visualization](#Visualization)
  * [Bar Subplots](#Bar-Subplots)
  * [Pie Subplots](#Pie-Subplots)
  * [Pearson Correlation Heatmaps](#Pearson-Correlation-Heatmaps)
  * [Bubble Plots](#Bubble-Plots)
  * [Scatter Plots](#Scatter-Plots)
  * [Pair Plot](#Pair-Plot)
- [Geomapping](#Geomapping)
  * [Global Data](#Global-Data)
  * [United States Data](#United-States-Data)

---

## Project Files:

---

- [Analysis.pdf](https://github.com/OmkarShivaprasad/Covid19_DataExploration/blob/main/Analysis.pdf)- Project PDF </br>

	>*This is the most user friendly option to veiw the code and outputs of this project. However, interactive plots will be shown as static images only.* 
	>
	>*If you are unable to veiw the PDF in GitHub, click 'Download' to save the file to your computer; if you are on a mobile device, click the three horizontal dots (...) to the right of 'Stored with Git LFS', and then click 'Download' to save to your device.*
</br>

- [Analysis.ipynb](https://github.com/OmkarShivaprasad/Covid19_DataExploration/blob/main/Analysis.ipynb)- Jupyter Notebook
- [Analysis.py](https://github.com/OmkarShivaprasad/Covid19_DataExploration/blob/main/Analysis.py)- Source Code and Markdown
- [environments.yml](https://github.com/OmkarShivaprasad/Covid19_DataExploration/tree/main/binder) - Dependencies 
- [global_covid_final_data.csv](https://github.com/OmkarShivaprasad/Covid19_DataExploration/blob/main/global_covid_final_data.csv)- Final Cleaned Global Dataset
- [us_covid_final_data.csv](https://github.com/OmkarShivaprasad/Covid19_DataExploration/blob/main/us_covid_final_data.csv)- Final Cleaned US Dataset
- [Images](https://github.com/OmkarShivaprasad/Covid19_DataExploration/tree/main/images)- Folder containing images of plots, cell outputs, and geomaps

---

## **Data Sources**

---

**"Cleveland Clinic Heart Disease Dataset"**

Creators:

1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

UCI Machine Learning Repository [link](https://archive.ics.uci.edu/ml/datasets/heart+disease)

---

## **Evaluation Metrics:**

---

**Confusion Matrix:** 

A confusion matrix is a performance measurement for machine learning classification problems where the target output is two or more classes. This will be used for measuring Recall, Precision, Specificity, Accuracy, and AUC-ROC curves for our binary classification problem.

The confusion matrix will assess the number of correct and incorrect predictions, summarized with count values and broken down by each class in a two by two matrix:  True Positive(TP), False Positive(FP), False Negative(FN), and True Negative(TN). 

- True Positive(TP): The model correctly predicts the positive class
- True Negative(TN): The model correctly predicts the negative class
- False Positive(FP): This is called a type 1 error. The model incorrectly predicts the positive class when it is actually negative
- False Negative(FN): This is called a type 2 error. The model incorrectly predicts the negative class when it is actually positive.

With these outputs we will calculate the following for our analysis:

**Accuracy:** This is equal to the proportion of predictions that the model classified correctly.

> Accuracy = *no. correct predictions / total no. of predictions* = (TP + TN) / (TP + TN + FP + FN)

**Precision:** Also known as 'positive predictive value', this is the proportion of relevant instances among retrieved instances. In other words, it is the ratio between the True Positives and all the Positives. The aim is always to reduce false positives and false positives. However, if our target data was imbalanced precision in the context of this analysis would be less important than recall. Suppose the number of false positives were high for the model; even if the prediction results in a false positive, the patient can undergo further tests which will show that it was in fact a false positive.

> Precision = TP / (TP + FP)

**Recall** Also known as 'sensitivity, hit rate, or true positive rate (TPR)', this is the proportion of total amount of relevant instances that were actually retrieved. In other words, it is the measure of our model correctly identifying True Positives. If our target data was imbalanced, paying attention to recall would be more important. Suppose the number of false negatives was high for the model; if the patient actually has heart disease, and the model predicts that the patient does not have heart disease, it could be disastrous. That is why we want to increase recall by lowering the number of false negatives.

> Recall = TP / (TP + FN)

**Specificity** Or 'true negative rate (TNR), measures proportion of actual negatives that are correctly identified as negatives

> Specificity = TN / (TN +FP)

**F1-Score** Measure of test accuracy(or precision and robustness of the model) using harmonic mean of precision and recall. A maximum of 1 signifies perfect precision and recall, with a minimum of 0. If we had a severely imbalanced target dataset we would be using an F-beta(2.0) which places less weight on precision and more weight on recall. However, because the target data is fairly balanced, we will stick with F-beta(1.0).

> F1 Score = (2 * (precision * recall))/ (precision + recall) = 2TP / (2TP + FP + FN)

</br>

**ROC Curve**

We will be using an ROC curve to summarize the trade-off between the true positive rate and false positive rate for each model using different probability thresholds values between 0.0 and 1.0. ROC curves are appropriate when the observations are balanced between each class. If our data had a large imbalance between output values, we would employ a Precision Recall Curve. The ROC curves of the different models can be compared for different thresholds. The shape of the curve contains important information about the false positive rate, and false negative rates; smaller values on the x-axis indicate lower false positives and higher true negatives, whereas, larger values on the y-axis indicates higher true positives and lower false negatives.  For the sake of comparison, we will also be measuring and plotting the Precision Recall Curve which is the ratio of the number of true positives divided by the sum of the true positives and false positives. It describes how good a model is at predicting the positive class. Precision is referred to as the positive predictive value.

---

## **Variable Descriptions**

---

**Age** : Age of the patient

**Sex** : Sex of the patient (0 = female, 1 = male)

**cp**: Chest Pain type chest pain type

  - Value 1: typical angina

  - Value 2: atypical angina

  - Value 3: non-anginal pain
  
  - Value 4: asymptomatic

**trtbps**: resting blood pressure (in mm Hg on admission to the hospital)

**chol**: cholestoral in mg/dl fetched via BMI sensor

**fbs**: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

**rest_ecg**: resting electrocardiographic results

  - Value 0: normal

  - Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
  
  - Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria 
  
**thalachh**: maximum heart rate achieved

**exng**: exercise induced angina (1 = yes; 0 = no)

**oldpeak**: ST depression induced by exercise relative to rest

**slp**: the slope of the peak exercise ST segment (1 = upsloping; 2 = flat; 3 = downsloping)

**caa**: number of major vessels (0-3)

**thall**: Thalium Stress Test result ~ (0-3)

**output**: *target variable* 0 = less chance of heart attack 1 = more chance of heart attack

---

## Steps in Analysis:
1. Import required libraries

2. Gather data

	- Webscraping with BeautifulSoup and Selenium
	- Reading in CSV files
	- Saving information into Pandas dataframes

3. Clean data

	- Manage null and missing values
	- Drop unwanted rows and columns
	- Rename columns and observations
	- Format datatypes
	- Add columns with calculations
	- Concatenate dataframes

4. Visualization

	- Seaborn | MatPlotLib | Plotly
	- Pearson Correlation Heatmaps
	- Bar Plots
	- Pair Plots
	- Pie Plots
	- Bubble Plots

5. Global GDP Exploration

	- Scatter Plots
	- Pearson Correlation

6. GeoMapping

	- Plotly Library
	- Choropleth
	- ScatterGeo

---

# PREVIEW

---

## Data Gathering and Cleaning:
</br>

### *GLOBAL DATA* 
</br>

**Global Covid-19 Raw Dataset**

![](images/globaldataraw.png)

**Global Covid-19 Initial Cleaned Dataset**

![](images/globaldataclean.png)

**Global GDP Per Capita Raw Dataset**

![](images/countrygdpraw.png)

**Global GDP Per Capita Cleaned Dataset**

![](images/countrygdpclean.png)

**Country Area Raw Dataset**

![](images/countryarearaw.png)

**Country Area Cleaned Dataset**

![](images/countryareaclean.png)

**Global Data Concatenated with Global GDP Per Capita and Country Area**

![](images/globaldataconcat.png)

**FINAL GLOBAL DATASET**

![](images/geoglobaldata.png)

**All Null Values Managed for Final Global Dataset**

![](images/geoglobaldataisnull.png)

</br>

### *UNITED STATES DATA* 
</br>

**United States Raw Dataset**

![](images/usdataraw.png)

**United States Cleaned Dataset**

![](images/usdataclean.png)

**FINAL US DATASET**

![](images/usgeodata.png)

**Null Values Managed for Final US Dataset**

![](images/usdataisnull.png)

---

## Visualization

</br>

##### **Bar Subplots**

![](images/toptenbarplot.png)

</br>

![](images/contbar.png)

</br></br> 

##### **Pie Subplots**

![](images/pie.png)

</br></br>

##### **Pearson Correlation Heatmaps**

![](images/toptenheat.png)

</br>

![](images/usheat.png)

</br>

![](images/worldpearson.png)

</br></br>

##### **Bubble Plots**

![](images/bubble.png)

</br></br>

##### **Scatter Plots**

![](images/popdenscatter.png)

</br>

![](images/totaltestscatter.png)

</br></br>

##### **Pair Plot**

![](images/pair.png)

---

## **StatsModel Summary**

</br>

### *TOTAL CASES PER MILLION VS GDP*

</br>

**With Constant**

![](images/output1.jpg)

**Without Constant**

![](images/output2.jpg)

</br>

![](images/gdpscatter.png)

</br>

### *DEATHS PER MILLION VS GDP*

</br>

**With Constant**

![](images/output3.jpg)

**Without Constant**

![](images/output4.jpg)

</br>

![](images/gdpdeathscatter.png)

</br>

### *TESTS PER MILLION VS GDP*

</br>

**With Constant**

![](images/output5.jpg)

**Without Constant**

![](images/output6.jpg)

</br>

![](images/gdptestscatter.png)

</br>

---

## Geomapping

</br>

##### **Bubble Plot Geomap**

![](images/countrypopbubble.png)

</br>

##### **Chloropleth Maps**

![](images/totcasechloro.png)

</br>

![](images/totdeathchloro.png)

</br>

![](images/statetotchloro.png)

</br>

![](images/statedeathchloro.png)






