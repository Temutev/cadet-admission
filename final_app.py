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

st.subheader('Height vs Weight')
fig, ax = plt.subplots()
sns.scatterplot(x='Height', y='Weight', hue='Selected', data=df, ax=ax)
st.pyplot(fig)


st.subheader('Toxicology vs Weight')
fig, ax = plt.subplots()
sns.scatterplot(x='Toxicology', y='Weight', hue='Selected', data=df, ax=ax)
st.pyplot(fig)



# Encode categorical variables
le = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = le.fit_transform(df[column])

# Fill missing values with the median of the column
df.fillna(df.median(), inplace=True)

# Split the data into training and testing sets
X = df.drop('Selected', axis=1)
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

# Get feature importances
importances = model.feature_importances_

# Convert the importances into a DataFrame
feature_importances = pd.DataFrame({"feature": X.columns, "importance": importances})

# Display feature importances as a bar chart
st.subheader('Feature Importances')
st.bar_chart(feature_importances.set_index('feature'))

# Model Evaluation
st.header('3. Model Evaluation')
st.write('Let\'s evaluate the model and make predictions.')

# Cross-validation scores
st.subheader('Cross-validation Scores')
scores = cross_val_score(model, X, y, cv=5)
st.write(f"Cross-validation scores: {scores}")
st.write(f"Average cross-validation score: {scores.mean()}")

# Confusion Matrix
st.subheader('Confusion Matrix')
cm = confusion_matrix(y_test, model.predict(X_test))
st.write("Confusion Matrix:", cm)

# Model Evaluation
st.header('4. Model Prediction')
st.write('You can use the following interface to try out the trained model.')
new_df = df [['Blood_pressure','Scars','Underlying_condition','Toxicology','Teeth','Height','Weight']]

from imblearn.over_sampling import SMOTE
# transform the dataset
#oversample = SMOTE()
#new_df, y = oversample.fit_resample(new_df, y)
new = RandomForestClassifier(random_state=23)
X_train, X_test, y_train, y_test = train_test_split(new_df, y, test_size=0.2, random_state=42)

new.fit(X_train, y_train)
y_pred = new.predict(X_test)


# Sidebar for user input
st.sidebar.header('User Input')
bp = st.sidebar.slider('Blood Pressure', 0, 1, 1)
uc = st.sidebar.slider('Underlying Condition', 0, 1, 1)
height = st.sidebar.slider('Height (ft)', 0, 10, 5)
weight = st.sidebar.slider('Weight (kg)', 0, 300, 70)
toxicology = st.sidebar.slider('Toxicology', 0, 1, 0)
scars = st.sidebar.slider('Scars', 0, 3, 1)
teeth = st.sidebar.slider('Teeth', 0, 4, 1)

input_data = {
    'Blood_pressure': [bp], 'Scars': [scars],
    'Underlying_condition': [uc],
    'Toxicology': [toxicology],
    'Teeth': [teeth],
    'Height': [height],
    'Weight': [weight],
}

input_df = pd.DataFrame(input_data)

# Make predictions
if st.sidebar.button('Predict'):
    prediction = new.predict(input_df)
    if prediction[0] == 1:
        st.sidebar.write('Selected: Yes')
    else:
        st.sidebar.write('Selected: No')
