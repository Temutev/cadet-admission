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

# Set Streamlit configurations
st.set_page_config(page_title='Cadet Admission Prediction Engine', layout='wide' ,page_icon='')
st.set_option('deprecation.showPyplotGlobalUse', False)
# Custom CSS to enhance the app's style
st.markdown(
    """
        <style>
            .stApp {
                background-color: #006847; /* Dark green background, representing Army Cadet green */
            }
            .st-eb {
                background-color: #AA8A63; /* Light brown error box */
                color: white; /* White text for error boxes */
                font-weight: bold;
            }
            .st-bd {
                background-color: #4B5320; /* Olive green info box */
            }
            body {
                font-size: 16px; /* Increase the font size to 16px */
                color: #006847; /* Army Cadet green text color */
            }
        </style>
    """,
    unsafe_allow_html=True
)

# Load the pre-trained model
model = joblib.load('model.pkl')

# Load the data
df = pd.read_excel('System_Data_selected.xlsx')
# Replace 'no' with 'No' in the 'Selected' column
df['Selected'] = df['Selected'].replace('no', 'No')
df['Eyes'] = df['Eyes'].replace('Normal ','Normal')
df['Nose'] = df['Nose'].replace('Normal ','Normal')
df['Teeth'] = df['Teeth'].replace('Normal ','Normal')
df['Teeth'] = df['Teeth'].replace('normal','Normal')
df['Teeth'] = df['Teeth'].replace('Abnormal ','Abnormal')
df['Scars'] = df['Scars'].replace('Normal ','Normal')
df['Scars'] = df['Scars'].replace('Abnormal ','Abnormal')
df['Limbs'] = df['Limbs'].replace('Normal ','Normal')
df['Limbs'] = df['Limbs'].replace('Abnormal ','Abnormal')
df['Blood pressure'] = df['Blood pressure'].replace('normal','Normal')


# Add a title to your Streamlit app
st.title('Cadet Admission Prediction Engine')

# Introduction
st.header('1. Introduction')
st.write('Welcome to the Machine Learning Showcase! This Streamlit application showcases a machine learning project for Cadet Admission Prediction Engine. The goal of this project is to predict whether a cadet will be selected or not based on the given data. The data was collected from the cadet admission system. The data was cleaned and preprocessed before training the model. The model was trained using the Random Forest Classifier algorithm. The model was evaluated using the cross-validation score and confusion matrix. The model was deployed using Streamlit. The model can be used to predict whether a cadet will be selected or not based on the given data.')

# Data Analysis
st.header('2. Data Analysis')
st.write('Let\'s explore the data and its analysis.')

# Display the first few rows of the DataFrame
st.subheader('First Few Rows of the Data')
st.dataframe(df.head())

# Show value counts of the 'Selected' column using a bar chart
st.subheader('Value Counts of the "Selected" Column')
selected_counts = df['Selected'].value_counts()
st.bar_chart(selected_counts)

# Rename columns
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


#show how the admissions have changed over time
st.subheader('Admissions over time')
#use seaborn to plot the number of admissions by year
fig, ax = plt.subplots()
sns.countplot(x='Year', hue='Selected', data=df, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)

st.write("The number of admissions has not changed much over time.It is consistent with the number of cadets that are selected each year.")

#show pairplot
# Create columns
col1, col2 = st.columns(2)

# Height vs Weight
with col1:
    st.subheader('Height vs Weight')
    fig, ax = plt.subplots()
    sns.scatterplot(x='Height', y='Weight', hue='Selected', data=df, ax=ax)
    st.pyplot(fig)

# Toxicology vs Weight
with col1:
    st.subheader('Toxicology vs Weight')
    fig, ax = plt.subplots()
    sns.scatterplot(x='Toxicology', y='Weight', hue='Selected', data=df, ax=ax)
    st.pyplot(fig)

# Blood Pressure vs Weight
with col2:
    st.subheader('Blood Pressure vs Weight')
    fig, ax = plt.subplots()
    sns.scatterplot(x='Blood_pressure', y='Weight', hue='Selected', data=df, ax=ax)
    st.pyplot(fig)

# Underlying Condition vs Weight
with col2:
    st.subheader('Underlying Condition vs Weight')
    fig, ax = plt.subplots()
    sns.scatterplot(x='Underlying_condition', y='Weight', hue='Selected', data=df, ax=ax)
    st.pyplot(fig)

# Sit ups vs Push ups
col3, col4 = st.columns(2)

with col3:
    st.subheader('Sit ups vs Push ups')
    fig, ax = plt.subplots()
    sns.scatterplot(x='Sit ups', y='Push_ups', hue='Selected', data=df, ax=ax)
    st.pyplot(fig)

# 3 mile run vs Push ups
with col4:
    st.subheader('3 mile run vs Push ups')
    fig, ax = plt.subplots()
    sns.scatterplot(x='3 mile run', y='Push_ups', hue='Selected', data=df, ax=ax)
    st.pyplot(fig)

# Encode categorical variables
le = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = le.fit_transform(df[column])

# Fill missing values with the median of the column
df.fillna(df.median(), inplace=True)

# Split the data into training and testing sets
X = df.drop(['Selected','Number','Year'], axis=1)
y = df['Selected']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Show a heatmap of correlations
st.subheader('Correlation Heatmap')
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
st.pyplot()

# Build the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Display feature importances as a bar chart
st.subheader('Feature Importances')
# Get feature importances
importances = model.feature_importances_

# Convert the importances into a DataFrame
feature_importances = pd.DataFrame({"feature": X.columns, "importance": importances})

# Sort feature importances in descending order
feature_importances = feature_importances.sort_values(by="importance", ascending=False)
code ="""
# Get feature importances
importances = model.feature_importances_

# Convert the importances into a DataFrame
feature_importances = pd.DataFrame({"feature": X.columns, "importance": importances})

# Sort feature importances in descending order
feature_importances = feature_importances.sort_values(by="importance", ascending=False)
"""
st.code(code, language='python')


st.bar_chart(feature_importances.set_index('feature'))
# Impact on admission decision
if feature_importances["importance"].iloc[0] > 0.1:
    st.write("The 'Underlying Condition' is highly impactful in admission decisions.")
else:
    st.write("No single feature dominates admission decisions, indicating a balanced approach.")

# Model Evaluation
st.header('3. Model Evaluation')
st.write('Let\'s evaluate the model and make predictions.')

# Cross-validation scores
st.subheader('Cross-validation Scores')
code ="""

scores = cross_val_score(model, X, y, cv=5)
st.write(f"Cross-validation scores: {scores}")
st.write(f"Average cross-validation score: {scores.mean()}")
Cross-validation scores: [0.98481562 0.97830803 0.98264642 0.97396963 0.95      ]
Average cross-validation score: 0.9739479392624728

"""
st.code(code, language='python')

scores = cross_val_score(model, X, y, cv=5)
st.write(f"Cross-validation scores: {scores}")
st.write(f"Average cross-validation score: {scores.mean()}")

# Confusion Matrix
st.subheader('Confusion Matrix')

code ="""
cm = confusion_matrix(y_test, model.predict(X_test))
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

"""
st.code(code, language='python')

cm = confusion_matrix(y_test, model.predict(X_test))
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot()



# Model Evaluation
st.header('4. Model Prediction')
st.write('You can use the following interface to try out the trained model.')


# Sidebar for user input
st.sidebar.header('User Input')
st.sidebar.write('Please enter the following information to predict whether a cadet will be selected or not.')
gd = st.sidebar.slider('Gender', 0, 1, 1)
age = st.sidebar.slider('Age', 0, 30, 20)
height = st.sidebar.slider('Height (ft)', 0, 10, 5)
weight = st.sidebar.slider('Weight (kg)', 0, 100, 55)
eyes = st.sidebar.slider('Eyes', 0, 1, 1)
nose = st.sidebar.slider('Nose', 0, 1, 1)
teeth = st.sidebar.slider('Teeth', 0, 1, 1)
scars = st.sidebar.slider('Scars', 0, 3, 1)
limbs = st.sidebar.slider('Limbs', 0, 1, 1)
three_mile_run = st.sidebar.slider('3 mile run', 20, 100, 65)
sit_ups = st.sidebar.slider('Sit ups', 20, 100, 75)
push_ups = st.sidebar.slider('Push ups', 20, 100, 65)
leadership = st.sidebar.slider('Leadership', 20, 100, 75)
obedience = st.sidebar.slider('Obedience', 20, 100, 70)
courage = st.sidebar.slider('Courage', 20, 100, 65)
aptitude_test = st.sidebar.slider('Aptitude Test', 20, 100, 85)
oral_interview = st.sidebar.slider('Oral Interview', 20, 100, 70)
bp = st.sidebar.slider('Blood Pressure', 0, 1, 1)
toxicology = st.sidebar.slider('Toxicology', 0, 1, 0)
uc = st.sidebar.slider('Underlying Condition', 0, 1, 1)


input_data = {
    'Gender': [gd],
    'Age': [age],
    'Height': [height],
    'Weight': [weight],
    'Eyes': [eyes],
    'Nose': [nose],
    'Teeth': [teeth],
    'Scars': [scars],
    'Limbs': [limbs],
    '3 mile run': [three_mile_run],
    'Sit ups': [sit_ups],
    'Push_ups': [push_ups],
    'Leadership': [leadership],
    ' Obedience': [obedience],
    'Courage': [courage],
    'Aptitude_Test': [aptitude_test],
    'Oral_Interview': [oral_interview],
    'Blood_pressure': [bp],
    'Toxicology': [toxicology],
    'Underlying_condition': [uc]

}

input_df = pd.DataFrame(input_data)

# Make predictions
if st.sidebar.button('Predict'):
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.sidebar.write('<p style="color: green;">Selected: Yes</p>', unsafe_allow_html=True)
    else:
        st.sidebar.write('<p style="color: red;">Selected: No</p>', unsafe_allow_html=True)
