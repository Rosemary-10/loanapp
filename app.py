## Importing the libraries
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import pickle

# Loading the model
loaded_model = pickle.load(open('loan_classifier', 'rb'))

# Importing the dataset
load = pd.read_csv('bankloan.csv')

# Functions for the loan prediction
def loan_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    return 'Your loan has not been approved' if prediction[0] == 0 else 'Congratulations, your loan has been approved.'

#Creating a count plot for gender
def chart_page():
    st.title('Loan Application by Gender')
    count_gender=px.histogram(
        load,
        x = 'Gender',
        color='Loan_Status',
        title='Gender Status of Loan Applicants',
        labels={'Loan_Status': 'Loan_Status'})
    st.plotly_chart(count_gender)    ## To show the plot

    # Adding some insights
    st.subheader('Insights')
    st.markdown('Men apply for ;oan more than women')

def dashboard_page():
    st.title('Dashboard Page')
    st.markdown('Input your values')

    # Collecting the user inputs
    col1, col2, col3 = st.columns(3)  ## To specify the number of columns
    
    with col1:
        Gender= st.selectbox('Gender(0=Female, 1=Male)', options = [0,1])
        Married= st.selectbox('Married(0=No, 1=Yes)', options = [0,1])
        Dependents= st.selectbox('Dependents( 0 or 1)', options = [0,1])
        Education= st.selectbox('Education(0=Not Grads or 1= Grads)', options = [0,1])

    with col2:
        Self_Employed= st.selectbox('Self_Employed(Select 0 or 1)', options = [0,1])
        ApplicantIncome= st.number_input('ApplicantIncome', value = 0)
        CoapplicantIncome= st.number_input('CoapplicantIncome', value = 0)
        LoanAmount= st.number_input('LoanAmount', value = 0)

    with col3:
        Loan_Amount_Term= st.number_input('Loan_Amount_Term', value = 0)
        Credit_History= st.selectbox('Credit_History(Select 0 or 1)', options = [0,1])
        Property_Area_Rural= st.selectbox('Property Area Rural(Select 0 or 1)', options = [0,1])
        Property_Area_Urban= st.selectbox('Property Area Urban(Select 0 or 1)', options = [0,1])


    if st.button('Bank Loan Application System'):
        try:
            input_data = [
                int(Gender),
                int(Married),
                int(Dependents),
                int(Education),
                int(Self_Employed),
                int(ApplicantIncome),
                float(CoapplicantIncome),
                float(LoanAmount),
                int(Loan_Amount_Term),
                float(Credit_History),
                int(Property_Area_Rural),
                int(Property_Area_Urban)
        ]
            result = loan_prediction(input_data)
            st.success(result)
        except ValueError:
            st.error('Enter a valid input')


# Function to switch tabs
def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.selectbox('Select Page', ['Chart', 'Form Inputs'])

    if page == 'Chart':
        chart_page()
    elif page=='Form Inputs':
        dashboard_page()

# Run app
if __name__ == '__main__':
    main()
                
