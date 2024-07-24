# Credit-Card-Risk-Prediction
In this project, we have two datasets: one for customer data and the other for financial indicators (Cibil score). The client data set consists of 62 columns and 51336 rows, whereas the finance data set has 26 columns and 51336 rows. 
We use the trigger column "Approved_Flag." to build a machine learning model that predicts whether or not the consumer is qualified for a credit card loan.
## Banking Related Treams:
1. Assets = Profitable Products for Banks(Likes any kind of loans)
2. Liability = losable Products for Banks(Likes kind of accounts current account, saving account,RD etc..)
3. NAP = Non Performing Asset
4. OSP = Outstanding Principle
5. DPD = Days Past Due
6. PAR = Portfolio At Risk
7. Disbured Amount = Loan Amount
   **If DPD<90 then it is PAR, if DPD<90 then it is NPA and DPD=0 then it is NDA(Non Delinquint Account)**
   1. DPD is 0 to 30 it is SMA1(Standard Mointoring Account)
   2. DPD is 31 to 60 it is SMA2
   3. DPD is 61 to 90 it is SMA3
   4. DPD is 90 to 180 it is NPA
   5. DPD > 180 it is Writen-off

  Data set do not have Null values but Some Rows and Columns have -99999 values which mean is null so we delete that rows and column.
  There is a no duplicate values and outliers.
  Merge both data set using Innor join
  Apply chi-square test and ANOVA test.
  Check Mutlicollinearity. 
  Apply Lable Encoding and One Hot Encoding.
  Create 3 models Random Forest Classifier, Decision Tree and XGBoot Classifier.
  Apply Hyperparameter Tuning on XGboost because XGBoost's accuracy is highest.
  Test the model with unseen dataset.
  Create projects .exe file for deployment using CMD.
