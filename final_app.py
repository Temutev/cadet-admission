import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the pre-trained model
model = joblib.load('model.pkl')

# Load the data
df = pd.read_excel('System_Data_selected.xlsx')
# Replace 'no' with 'No' in the 'Selected' column
df['Selected'] = df['Selected'].replace('no', 'No')

# Add a title to your Streamlit app
st.title('Project Showcase')

# Introduction
st.header('1. Introduction')
st.write('This Streamlit application showcases a machine learning project.')

# Data Analysis
st.header('2. Data Analysis')
st.write('Let\'s explore the data and its analysis.')

# Display the first few rows of the DataFrame
st.subheader('First Few Rows of the Data')
st.write(df.head())

# Show value counts of the 'Selected' column
st.subheader('Value Counts of the "Selected" Column')
st.write(df['Selected'].value_counts())

df.rename(columns = {'Height (ft)':'Height',
                    'Weight (kg)':'Weight',
                    ' 3 mile run':'3_mile_run',
                    'Sit ups ':'Sit_ups ',
                    'Push ups':'Push_ups',
                    'Aptitude Test':'Aptitude_Test',
                    'Oral Interview':'Oral_Interview',
                    'Blood pressure':'Blood_pressure',
                    'Underlying condition':'Underlying_condition'}, inplace = True)

df[['Gender', 'Eyes', 'Nose', 'Teeth', 'Scars', 'Limbs', 'Blood_pressure', 'Toxicology', 'Underlying_condition', 'Selected']] = df[['Gender', 'Eyes', 'Nose', 'Teeth', 'Scars', 'Limbs', 'Blood_pressure', 'Toxicology', 'Underlying_condition', 'Selected']].astype(str)

# Encode categorical variables
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == type(object):
        df[column] = le.fit_transform(df[column])

# Fill missing values with the median of the column
df.fillna(df.median(), inplace=True)

# Split the data into training and testing sets
X = df.drop('Selected', axis=1)
y = df['Selected']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Show a heatmap of correlations
#st.subheader('Correlation Heatmap')
#plt.figure(figsize=(18, 15))
#sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
#st.pyplot()


# Build the model
model = RandomForestClassifier()
model.fit(X_train, y_train)


# Get feature importances
importances = model.feature_importances_

# Convert the importances into a DataFrame
feature_importances = pd.DataFrame({"feature": X.columns, "importance": importances})



# Display feature importances
st.subheader('Feature Importances')
st.write(feature_importances)

new_df = df [['Blood_pressure','Scars',
              'Underlying_condition','Toxicology','Teeth','Height','Weight']]

from imblearn.over_sampling import SMOTE
# transform the dataset
oversample = SMOTE()
new_df, y = oversample.fit_resample(new_df, y)
new = RandomForestClassifier(random_state=23)
X_train, X_test, y_train, y_test = train_test_split(new_df, y, test_size=0.2, random_state=42)

new.fit(X_train, y_train)
y_pred = new.predict(X_test)


scores = cross_val_score(new, new_df, y, cv=5)
# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Cross-validation scores
st.subheader('Cross-validation Scores')
st.write(f"Cross-validation scores: {scores}")
st.write(f"Average cross-validation score: {scores.mean()}")

# Confusion Matrix
st.subheader('Confusion Matrix')
st.pyplot(plt.figure(figsize=(10, 7)))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')



# Model Evaluation
st.header('3. Try the Model')
st.write('You can use the following interface to try out the trained model.')

# Sidebar for user input
st.sidebar.header('User Input')
bp = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=1, value=1)
uc = st.sidebar.number_input('Underlying Condition', min_value=0, max_value=1, value=1)
height = st.sidebar.number_input('Height (ft)', min_value=0, max_value=10, value=5)
weight = st.sidebar.number_input('Weight (kg)', min_value=0, max_value=300, value=70)
toxicology = st.sidebar.number_input('Toxicology', min_value=0, max_value=1, value=0)
scars = st.sidebar.number_input('Toxicology', min_value=0, max_value=3, value=1)
teeth = st.sidebar.number_input('Teeth', min_value=0, max_value=4, value=1)
# Add more inputs for other features as needed

input_data = {
    'Blood_pressure': [bp],'Scars': [scars],
    'Underlying_condition': [uc],
    'Toxicology': [toxicology],
    'Teeth': [teeth],
    'Height': [height],
    'Weight': [weight],
    
    
    # Add more features here
}

input_df = pd.DataFrame(input_data)

# Make predictions
if st.sidebar.button('Predict'):
    prediction = new.predict(input_df)
    if prediction[0] == 1:
        st.sidebar.write('Selected: Yes')
    else:
        st.sidebar.write('Selected: No')


