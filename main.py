
import pandas as pd 

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from joblib import dump


telecom_cust = pd.read_csv('Telco_Customer_Churn.csv') 
                           
telecom_cust ['TotalCharges'] = pd.to_numeric(telecom_cust['TotalCharges'], errors='coerce')
telecom_cust['TotalCharges'].fillna(0, inplace=True)


#binary labels
label_encoder = LabelEncoder()
telecom_cust['Churn'] = label_encoder.fit_transform(telecom_cust['Churn'])

#Internet service
telecom_cust['InternetService'] = label_encoder.fit_transform(telecom_cust['InternetService'])
telecom_cust['Contract'] = label_encoder.fit_transform(telecom_cust['Contract'])

#select features
selected_features = ['tenure', 'InternetService', 'MonthlyCharges', 'TotalCharges']
X = telecom_cust[selected_features]
y = telecom_cust['Churn']

#train random forest mode
model = RandomForestClassifier(n_estimators=100, random_state=101)
model.fit(X,y)

# save train model
dump(model, 'random_forest_model.joblib')
