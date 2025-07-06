import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title("💳 Credit Card Fraud Detection System")

@st.cache_data
def load_data():
    df = pd.read_csv("//Users/poorvajha/College/SEM4/CreditCard_Fraud_Detection/fraudTrain.csv")
    t_df = pd.read_csv("/Users/poorvajha/College/SEM4/CreditCard_Fraud_Detection/fraudTest.csv")
    return df, t_df

def preprocess_data(df):
    columns_to_drop = [col for col in ['cc_num', 'trans_date_trans_time', 'first', 'last', 'dob', 
                                       'street', 'trans_num', 'unix_time', 'merchant'] 
                       if col in df.columns]
    
    if columns_to_drop:
        df = df.drop(columns_to_drop, axis=1)
    
    for col in ['amt', 'city_pop']:
        if col in df.columns:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        if col == 'gender':
            df[col] = df[col].apply(lambda x: 1 if x == 'M' else 0)
        else:
            df[col] = LabelEncoder().fit_transform(df[col])

    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    return df

def prepare_prediction_input(amount, city_pop, lat, long):
    input_data = pd.DataFrame({
        'Unnamed: 0': [0], 
        'amt': [amount],
        'city_pop': [city_pop],
        'lat': [lat],
        'long': [long],
        'merch_lat': [lat + 0.01],
        'merch_long': [long + 0.01],
        'gender': ['M'],
        'category': ['shopping'],
        'city': ['default_city'],  
        'job': ['default_job'],
        'state': ['default_state'],
        'zip': ['00000']
    })
    
    return input_data

def main():
    if st.sidebar.button('Load Data'):
        with st.spinner('Loading and preprocessing data...'):
            df, t_df = load_data()
            df_preprocessed = preprocess_data(df)
            t_df_preprocessed = preprocess_data(t_df)
            
            st.session_state.df = df
            st.session_state.df_preprocessed = df_preprocessed
            st.session_state.t_df = t_df  
            st.session_state.t_df_preprocessed = t_df_preprocessed
            st.session_state.X_train = df_preprocessed.drop('is_fraud', axis=1)
            st.session_state.y_train = df_preprocessed['is_fraud']
            st.session_state.X_test = t_df_preprocessed.drop('is_fraud', axis=1)
            st.session_state.y_test = t_df_preprocessed['is_fraud']
            
            st.session_state.train_columns = st.session_state.X_train.columns.tolist()
            
            st.success('Data loaded and processed!')

    if 'df' in st.session_state:
        st.subheader('Data Overview')
        st.write("Training Data Shape:", st.session_state.df_preprocessed.shape)
        st.write("Test Data Shape:", st.session_state.t_df_preprocessed.shape)
        
        if st.checkbox('Show sample data'):
            st.dataframe(st.session_state.df_preprocessed.head())

    if 'df' in st.session_state and st.checkbox('Show EDA Visualizations'):
        st.header("Exploratory Data Analysis")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Basic Stats", "Geospatial", "Over Time", "3D View", "Correlations"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Fraud Proportion")
                fraud_proportion = st.session_state.df['is_fraud'].value_counts(normalize=True)
                fig, ax = plt.subplots()
                ax.pie(fraud_proportion, labels=['Non-Fraud', 'Fraud'], 
                      autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
                st.pyplot(fig)
            
            with col2:
                st.write("### Transaction Amount by Fraud")
                fig, ax = plt.subplots()
                sns.violinplot(x='is_fraud', y='amt', 
                              data=st.session_state.df_preprocessed, ax=ax)
                st.pyplot(fig)
                
            st.write("### Distributions")
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                sns.histplot(st.session_state.df_preprocessed['amt'], 
                            bins=50, kde=True, color='blue', ax=ax)
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots()
                sns.histplot(st.session_state.df_preprocessed['city_pop'], 
                            bins=50, kde=True, color='green', ax=ax)
                st.pyplot(fig)
        
        with tab2:
            st.write("### Geospatial Fraud Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='long', y='lat', hue='is_fraud',
                           data=st.session_state.df, palette='coolwarm', alpha=0.6, ax=ax)
            st.pyplot(fig)
        
        with tab3:
            st.write("### Fraud Over Time")
            df_time = st.session_state.df.copy()
            df_time['trans_date_trans_time'] = pd.to_datetime(df_time['trans_date_trans_time'])
            fraud_over_time = df_time.groupby(df_time['trans_date_trans_time'].dt.date)['is_fraud'].sum()
            fig, ax = plt.subplots(figsize=(10, 4))
            fraud_over_time.plot(ax=ax)
            st.pyplot(fig)
        
        with tab4:
            st.write("### 3D Visualization")
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(st.session_state.df_preprocessed['amt'], 
                      st.session_state.df_preprocessed['city_pop'], 
                      st.session_state.df_preprocessed['is_fraud'], 
                      c=st.session_state.df_preprocessed['is_fraud'], 
                      cmap='coolwarm', s=20)
            ax.set_xlabel('Amount')
            ax.set_ylabel('City Population')
            ax.set_zlabel('Is Fraud')
            st.pyplot(fig)
        
        with tab5:
            st.write("### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(12, 8))
            corr = st.session_state.df_preprocessed.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig)

    if 'X_train' in st.session_state and st.checkbox('Train Model'):
        st.header("Model Training")
        
        if st.checkbox('Apply SMOTE (Class Balancing)'):
            with st.spinner('Applying SMOTE...'):
                smote = SMOTE(random_state=42)
                X_train_balanced, y_train_balanced = smote.fit_resample(
                    st.session_state.X_train, st.session_state.y_train)
                
                st.write("Class distribution after SMOTE:", Counter(y_train_balanced))
                fig, ax = plt.subplots()
                sns.countplot(x=y_train_balanced, ax=ax)
                st.pyplot(fig)
        else:
            X_train_balanced, y_train_balanced = st.session_state.X_train, st.session_state.y_train
        
        if st.button('Train Isolation Forest'):
            with st.spinner('Training model...'):
                iso_forest = IsolationForest(contamination=0.02, random_state=42)
                iso_forest.fit(X_train_balanced)
                
                test_pred = np.where(iso_forest.predict(st.session_state.X_test) == -1, 1, 0)
                
                st.session_state.model = iso_forest
                st.session_state.test_pred = test_pred
                
                st.success('Model trained successfully!')
                
                st.write("### Model Performance")
                st.write(f"Accuracy: {accuracy_score(st.session_state.y_test, test_pred):.4f}")
                
                st.write("#### Confusion Matrix")
                cm = confusion_matrix(st.session_state.y_test, test_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['Non-Fraud', 'Fraud'],
                           yticklabels=['Non-Fraud', 'Fraud'], ax=ax)
                st.pyplot(fig)
                
                st.write("#### ROC Curve")
                test_scores = -iso_forest.decision_function(st.session_state.X_test)
                fpr, tpr, _ = roc_curve(st.session_state.y_test, test_scores)
                auc_score = roc_auc_score(st.session_state.y_test, test_scores)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.legend()
                st.pyplot(fig)

    if 'model' in st.session_state and st.checkbox('Make Predictions'):
        st.header("Fraud Prediction")
        
        with st.form("prediction_form"):
            st.write("### Enter Transaction Details")
            
            col1, col2 = st.columns(2)
            with col1:
                amount = st.number_input("Amount", min_value=0.0, value=100.0)
                city_pop = st.number_input("City Population", min_value=0, value=100000)
            with col2:
                lat = st.number_input("Latitude", value=40.0)
                long = st.number_input("Longitude", value=-75.0)
            
            submitted = st.form_submit_button("Predict")
        
        if submitted:
            input_data = prepare_prediction_input(amount, city_pop, lat, long)
            
            input_processed = preprocess_data(input_data)

            input_processed = input_processed[st.session_state.train_columns]
            
            prediction = st.session_state.model.predict(input_processed)
            result = "Fraud" if prediction[0] == -1 else "Not Fraud"
            
            st.success(f"Prediction: {result}")
            
            st.write("### Sample Predictions from Test Set")
            sample_data = st.session_state.t_df.sample(5)
            sample_processed = preprocess_data(sample_data)
            
            sample_processed = sample_processed[st.session_state.train_columns]
            
            sample_pred = np.where(
                st.session_state.model.predict(sample_processed) == -1, 
                'Fraud', 'Non-Fraud')
            sample_data['Prediction'] = sample_pred
            st.dataframe(sample_data[['amt', 'city_pop', 'lat', 'long', 'Prediction']])
main()