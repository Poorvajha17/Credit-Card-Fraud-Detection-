import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# ---------------------------
# File paths for artifacts
# ---------------------------
ISO_MODEL_PATH = "iso_forest_model.pkl"
RF_MODEL_PATH = "rf_model.pkl"
SCALER_PATH = "minmax_scaler.pkl"
CAT_MAPS_PATH = "categorical_maps.pkl"
TRAIN_COLUMNS_PATH = "train_columns.pkl"

# ---------------------------
# App config & CSS
# ---------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .fraud-alert {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #c62828;
        font-weight: bold;
    }
    .safe-alert {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2e7d32;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Utility functions
# ---------------------------
@st.cache_data
def load_data():
    """Load training and test datasets"""
    df = pd.read_csv("/Users/poorvajha/Desktop/College/SEM4/credit_card_fraud/Credit-Card-Fraud-Detection-/fraudTrain.csv")
    t_df = pd.read_csv("/Users/poorvajha/Desktop/College/SEM4/credit_card_fraud/Credit-Card-Fraud-Detection-/fraudTest.csv")
    return df, t_df

def drop_unneeded_columns(df):
    to_drop = ['cc_num','trans_date_trans_time','first','last','dob','street','trans_num','unix_time','merchant']
    return df.drop([c for c in to_drop if c in df.columns], axis=1)

def build_categorical_maps(df, categorical_cols):
    """Create mapping dicts for categorical columns to integer codes"""
    maps = {}
    for col in categorical_cols:
        if col == 'gender':
            # gender: M -> 1 else 0 (explicit)
            maps[col] = {'M': 1, 'F': 0}
        else:
            unique_vals = df[col].astype(str).fillna("___NA___").unique().tolist()
            maps[col] = {val: idx for idx, val in enumerate(unique_vals)}
    return maps

def transform_categoricals(df, cat_maps):
    """Map categories using saved maps. Unknowns get assigned new index = len(map)."""
    df = df.copy()
    for col, mp in cat_maps.items():
        if col in df.columns:
            if col == 'gender':
                # support numeric genders if present
                df[col] = df[col].apply(lambda x: mp.get(x, mp.get(str(x), 0)))
            else:
                df[col] = df[col].astype(str).fillna("___NA___").map(lambda x: mp.get(x, len(mp))).astype(int)
    return df

def fit_and_save_preprocessors(df):
    """Fit scaler and categorical maps on training data and save them."""
    df = df.copy()
    # separate features (drop is_fraud if present)
    if 'is_fraud' in df.columns:
        features = df.drop('is_fraud', axis=1)
    else:
        features = df

    # Identify categorical and numeric columns
    categorical_cols = features.select_dtypes(include=['object', 'category']).columns.tolist()
    # keep gender in categorical even if numeric in some cases
    if 'gender' in features.columns and 'gender' not in categorical_cols:
        categorical_cols.append('gender')

    numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # exclude boolean-like or encoded columns if any
    numeric_cols = [c for c in numeric_cols if c not in categorical_cols]

    # Build and save maps
    cat_maps = build_categorical_maps(features, categorical_cols)
    with open(CAT_MAPS_PATH, "wb") as f:
        pickle.dump(cat_maps, f)

    # Fit scaler on numeric columns
    scaler = MinMaxScaler()
    if numeric_cols:
        scaler.fit(features[numeric_cols])
    with open(SCALER_PATH, "wb") as f:
        pickle.dump({'scaler': scaler, 'numeric_cols': numeric_cols, 'categorical_cols': categorical_cols}, f)

    # Save train columns order for later alignment
    train_cols = list(features.columns)
    with open(TRAIN_COLUMNS_PATH, "wb") as f:
        pickle.dump(train_cols, f)
    return cat_maps, scaler, numeric_cols, categorical_cols, train_cols

def load_preprocessors():
    """Load saved scaler, numeric/categorical columns, maps and train column order"""
    if not os.path.exists(CAT_MAPS_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(TRAIN_COLUMNS_PATH):
        return None
    with open(CAT_MAPS_PATH, "rb") as f:
        cat_maps = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        sc_dict = pickle.load(f)
        scaler = sc_dict['scaler']
        numeric_cols = sc_dict['numeric_cols']
        categorical_cols = sc_dict['categorical_cols']
    with open(TRAIN_COLUMNS_PATH, "rb") as f:
        train_cols = pickle.load(f)
    return cat_maps, scaler, numeric_cols, categorical_cols, train_cols

def preprocess_for_model(df, cat_maps=None, scaler=None, numeric_cols=None, categorical_cols=None, fit=False):
    """
    Preprocess a DataFrame to be fed into models.
    If fit=True: expects fresh df and will return fitted artifacts.
    If fit=False: uses provided cat_maps and scaler to transform.
    """
    df = df.copy()
    # drop unwanted columns
    df = drop_unneeded_columns(df)

    # handle missing categorical/numeric columns gracefully by adding defaults
    # We'll ensure columns exist as in train columns later by alignment function.
    # For now, convert object columns to string
    for c in df.select_dtypes(include=['object', 'category']).columns:
        df[c] = df[c].astype(str).fillna("___NA___")

    # If fitting: build maps & scaler
    if fit:
        cat_maps, scaler, numeric_cols, categorical_cols, train_cols = fit_and_save_preprocessors(df)
        # transform using those
        df = transform_categoricals(df, cat_maps)
        if numeric_cols:
            df[numeric_cols] = scaler.transform(df[numeric_cols])
        return df, (cat_maps, scaler, numeric_cols, categorical_cols, train_cols)
    else:
        # require maps and scaler
        if cat_maps is None or scaler is None or numeric_cols is None or categorical_cols is None:
            raise ValueError("Preprocessors not provided for transform stage.")
        df = transform_categoricals(df, cat_maps)
        # ensure numeric cols exist (fill missing numeric cols with 0)
        for nc in numeric_cols:
            if nc not in df.columns:
                df[nc] = 0.0
        if numeric_cols:
            df[numeric_cols] = scaler.transform(df[numeric_cols])
        return df

def align_columns_to_train(df, train_columns):
    """
    Ensure df contains exactly the train_columns in same order.
    If some train columns missing -> create and fill with 0.
    If df has extra cols -> drop them.
    """
    out = pd.DataFrame(index=df.index)
    for c in train_columns:
        if c in df.columns:
            out[c] = df[c]
        else:
            # create default column (0)
            out[c] = 0
    return out[train_columns]

# ---------------------------
# Model training / loading
# ---------------------------
def load_or_train_models(X_train, y_train):
    """Load existing models or train new ones (and save them)."""
    models = {}

    # Train or load IsolationForest
    if os.path.exists(ISO_MODEL_PATH):
        with open(ISO_MODEL_PATH, "rb") as f:
            models['iso_forest'] = pickle.load(f)
        st.sidebar.success("✓ Isolation Forest loaded")
    else:
        with st.spinner('Training Isolation Forest...'):
            fraud_rate = y_train.mean() if hasattr(y_train, "mean") else (y_train.sum() / len(y_train))
            iso = IsolationForest(
                contamination=min(max(fraud_rate * 1.2, 0.001), 0.1),
                random_state=42,
                n_estimators=200,
                max_samples='auto'
            )
            iso.fit(X_train)
            with open(ISO_MODEL_PATH, "wb") as f:
                pickle.dump(iso, f)
            models['iso_forest'] = iso
            st.sidebar.success("✓ Isolation Forest trained & saved")

    # Train or load RandomForest (with SMOTE applied only on training data)
    if os.path.exists(RF_MODEL_PATH):
        with open(RF_MODEL_PATH, "rb") as f:
            models['random_forest'] = pickle.load(f)
        st.sidebar.success("✓ Random Forest loaded")
    else:
        with st.spinner('Training Random Forest with SMOTE and class weights...'):
            # Apply SMOTE on the numeric/categorical combined training features
            smote = SMOTE(sampling_strategy=0.5, random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)

            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=25,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight={0: 1, 1: 10},
                n_jobs=-1,
                random_state=42
            )
            rf.fit(X_res, y_res)
            with open(RF_MODEL_PATH, "wb") as f:
                pickle.dump(rf, f)
            models['random_forest'] = rf
            st.sidebar.success("✓ Random Forest trained & saved")

    return models

# ---------------------------
# Prediction input builder
# ---------------------------
def build_input_df_from_user(inputs):
    """
    inputs: dict containing keys for all features user will input.
    We will create a single-row DataFrame with those keys.
    """
    # keys we expect (after dropping the big list)
    expected = [
        "category", "amt", "gender", "city", "state", "zip",
        "lat", "long", "city_pop", "job", "merch_lat", "merch_long"
    ]
    data = {}
    for k in expected:
        # if user didn't provide merch_lat/merch_long, we compute from lat/long later
        data[k] = inputs.get(k, None)
    # If merch_lat/merch_long missing, compute tiny offset
    if pd.isna(data['merch_lat']) or data['merch_lat'] is None:
        data['merch_lat'] = float(data['lat']) + 0.01
    if pd.isna(data['merch_long']) or data['merch_long'] is None:
        data['merch_long'] = float(data['long']) + 0.01

    df = pd.DataFrame([data])
    return df

# ---------------------------
# Main App
# ---------------------------
def main():
    st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)

    # Load data once
    if 'data_loaded' not in st.session_state:
        df, t_df = load_data()
        # Save originals
        st.session_state.df_orig = df.copy()
        st.session_state.t_df_orig = t_df.copy()

        # Preprocess training data (fit encoders & scaler)
        df_proc, preprocess_artifacts = preprocess_for_model(df.copy(), fit=True)
        cat_maps, scaler, numeric_cols, categorical_cols, train_cols = preprocess_artifacts

        # Align training features & split
        X_train = df_proc.drop('is_fraud', axis=1)
        y_train = df_proc['is_fraud']
        # Save artifacts loaded into session state
        st.session_state.cat_maps = cat_maps
        st.session_state.scaler = scaler
        st.session_state.numeric_cols = numeric_cols
        st.session_state.categorical_cols = categorical_cols
        st.session_state.train_columns = train_cols
        st.session_state.df_preprocessed = df_proc
        # Preprocess test using saved preprocessors
        t_proc = preprocess_for_model(t_df.copy(), cat_maps=cat_maps, scaler=scaler,
                                      numeric_cols=numeric_cols, categorical_cols=categorical_cols, fit=False)
        st.session_state.t_df_preprocessed = t_proc
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.session_state.X_test = t_proc.drop('is_fraud', axis=1)
        st.session_state.y_test = t_proc['is_fraud']
        st.session_state.data_loaded = True

    # Load or train models (these will also load saved artifacts if exist)
    if 'models' not in st.session_state:
        st.session_state.models = load_or_train_models(st.session_state.X_train, st.session_state.y_train)

    # Sidebar info
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("---")
    st.sidebar.metric("Training Samples", f"{st.session_state.df_preprocessed.shape[0]:,}")
    st.sidebar.metric("Test Samples", f"{st.session_state.t_df_preprocessed.shape[0]:,}")
    st.sidebar.metric("Features", len(st.session_state.train_columns))
    fraud_rate = st.session_state.df_orig['is_fraud'].mean() * 100
    st.sidebar.metric("Fraud Rate", f"{fraud_rate:.2f}%")

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Fraud Prediction", "📈 Model Performance", "🔍 Data Analysis", "📋 Sample Data"])

    # =========================
    # Tab 1: Fraud Prediction
    # =========================
    with tab1:
        st.header("Real-Time Fraud Detection")
        st.markdown("Enter transaction details to predict fraud risk in real-time")

        # We'll collect all remaining columns (after dropping)
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                category = st.text_input("Category", value="shopping")
                amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=1.0, format="%.2f")
                gender = st.selectbox("Gender", options=["M", "F"], index=0)
                city = st.text_input("City", value="default_city")
            with col2:
                state = st.text_input("State", value="default_state")
                zip_code = st.text_input("ZIP", value="00000")
                lat = st.number_input("Latitude", value=40.7128, format="%.6f")
                long = st.number_input("Longitude", value=-74.0060, format="%.6f")
            with col3:
                city_pop = st.number_input("City Population", min_value=0, value=100000, step=1000)
                job = st.text_input("Job", value="default_job")

            submitted = st.form_submit_button("🔍 Analyze Transaction")

        if submitted:
            user_inputs = {
                "category": category,
                "amt": float(amt),
                "gender": gender,
                "city": city,
                "state": state,
                "zip": zip_code,
                "lat": float(lat),
                "long": float(long),
                "city_pop": float(city_pop),
                "job": job,
            }

            input_df = build_input_df_from_user(user_inputs)

            # Preprocess using saved maps and scaler and align to train columns
            input_processed = preprocess_for_model(input_df,
                                                  cat_maps=st.session_state.cat_maps,
                                                  scaler=st.session_state.scaler,
                                                  numeric_cols=st.session_state.numeric_cols,
                                                  categorical_cols=st.session_state.categorical_cols,
                                                  fit=False)

            input_aligned = align_columns_to_train(input_processed, st.session_state.train_columns)

            # Make predictions
            iso_pred = st.session_state.models['iso_forest'].predict(input_aligned)[0]
            iso_result = "FRAUD" if iso_pred == -1 else "LEGITIMATE"

            rf_pred = st.session_state.models['random_forest'].predict(input_aligned)[0]
            rf_result = "FRAUD" if rf_pred == 1 else "LEGITIMATE"

            rf_proba = st.session_state.models['random_forest'].predict_proba(input_aligned)[0]
            fraud_probability = rf_proba[1] * 100

            # Display
            st.markdown("### 🎯 Prediction Results")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Isolation Forest", iso_result,
                          delta="High Risk" if iso_result == "FRAUD" else "Low Risk",
                          delta_color="inverse")
            with c2:
                st.metric("Random Forest", rf_result,
                          delta="High Risk" if rf_result == "FRAUD" else "Low Risk",
                          delta_color="inverse")
            with c3:
                st.metric("Fraud Probability", f"{fraud_probability:.1f}%",
                          delta="Alert" if fraud_probability > 50 else "Safe",
                          delta_color="inverse")

            st.markdown("---")
            if iso_result == "FRAUD" or rf_result == "FRAUD":
                st.markdown(f'<div class="fraud-alert">⚠️ FRAUD ALERT: This transaction has been flagged as potentially fraudulent!</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="safe-alert">✓ LEGITIMATE: This transaction appears to be safe.</div>',
                            unsafe_allow_html=True)

    # =========================
    # Tab 2: Model Performance
    # =========================
    with tab2:
        st.header("Model Performance Metrics")

        # Calculate predictions if not already done
        if 'iso_pred' not in st.session_state:
            iso_pred = np.where(st.session_state.models['iso_forest'].predict(st.session_state.X_test) == -1, 1, 0)
            st.session_state.iso_pred = iso_pred

        if 'rf_pred' not in st.session_state:
            rf_pred = st.session_state.models['random_forest'].predict(st.session_state.X_test)
            st.session_state.rf_pred = rf_pred

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Isolation Forest")
            iso_acc = accuracy_score(st.session_state.y_test, st.session_state.iso_pred)
            st.metric("Accuracy", f"{iso_acc:.2%}")

            cm_iso = confusion_matrix(st.session_state.y_test, st.session_state.iso_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_iso, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Legitimate', 'Fraud'],
                        yticklabels=['Legitimate', 'Fraud'])
            ax.set_title('Confusion Matrix - Isolation Forest')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            st.pyplot(fig)

        with col2:
            st.subheader("Random Forest")
            rf_acc = accuracy_score(st.session_state.y_test, st.session_state.rf_pred)
            st.metric("Accuracy", f"{rf_acc:.2%}")

            cm_rf = confusion_matrix(st.session_state.y_test, st.session_state.rf_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax,
                        xticklabels=['Legitimate', 'Fraud'],
                        yticklabels=['Legitimate', 'Fraud'])
            ax.set_title('Confusion Matrix - Random Forest')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            st.pyplot(fig)

        st.markdown("---")

        # Classification reports
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Isolation Forest - Detailed Metrics")
            report_iso = classification_report(st.session_state.y_test, st.session_state.iso_pred,
                                               target_names=['Legitimate', 'Fraud'], output_dict=True)
            st.dataframe(pd.DataFrame(report_iso).transpose())

        with col2:
            st.subheader("Random Forest - Detailed Metrics")
            report_rf = classification_report(st.session_state.y_test, st.session_state.rf_pred,
                                              target_names=['Legitimate', 'Fraud'], output_dict=True)
            st.dataframe(pd.DataFrame(report_rf).transpose())

    # =========================
    # Tab 3: Data Analysis
    # =========================
    with tab3:
        st.header("Exploratory Data Analysis")

        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["Distribution", "Geospatial", "Correlations"])

        with analysis_tab1:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.subheader("Fraud Distribution")
                fraud_counts = st.session_state.df_orig['is_fraud'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#2ecc71', '#e74c3c']
                ax.pie(fraud_counts, labels=['Legitimate', 'Fraud'], autopct='%1.2f%%',
                       colors=colors, startangle=90, textprops={'fontsize': 12})
                ax.set_title('Transaction Distribution', fontsize=14, fontweight='bold')
                st.pyplot(fig)

            with col2:
                st.subheader("Amount Distribution by Fraud Status")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.violinplot(x='is_fraud', y='amt', data=st.session_state.df_preprocessed,
                              palette=['#2ecc71', '#e74c3c'], ax=ax)
                ax.set_xticklabels(['Legitimate', 'Fraud'])
                ax.set_xlabel('Transaction Type', fontsize=12)
                ax.set_ylabel('Amount (Normalized)', fontsize=12)
                ax.set_title('Transaction Amount Distribution', fontsize=14, fontweight='bold')
                st.pyplot(fig)

            with col3:
                st.subheader("Overall Transaction Amount Distribution")
                fig, ax = plt.subplots(figsize=(8,6))
                sns.histplot(st.session_state.df_orig['amt'], bins=50, kde=True, ax=ax, color="#3498db")
                ax.set_xlabel("Transaction Amount")
                ax.set_ylabel("Count")
                ax.set_title("Distribution of Transaction Amounts", fontsize=14, fontweight="bold")
                st.pyplot(fig)
            
            with col4:
                st.subheader("Fraud Rate by Category")
                fraud_rate = st.session_state.df_orig.groupby('category')['is_fraud'].mean().sort_values()

                fig, ax = plt.subplots(figsize=(10,6))
                sns.barplot(x=fraud_rate.index, y=fraud_rate.values, ax=ax, palette="viridis")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_ylabel("Fraud Rate")
                ax.set_title("Fraud Rate per Transaction Category", fontsize=14, fontweight="bold")
                st.pyplot(fig)


        with analysis_tab2:
            st.subheader("Geographical Distribution of Transactions")
            fig, ax = plt.subplots(figsize=(12, 8))
            fraud_data = st.session_state.df_orig[st.session_state.df_orig['is_fraud'] == 1]
            legit_data = st.session_state.df_orig[st.session_state.df_orig['is_fraud'] == 0].sample(
                n=min(5000, len(fraud_data) * 5))
            ax.scatter(legit_data['long'], legit_data['lat'],
                       c='blue', s=1, alpha=0.3, label='Legitimate')
            ax.scatter(fraud_data['long'], fraud_data['lat'],
                       c='red', s=10, alpha=0.6, label='Fraud')
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            ax.set_title('Fraud vs Legitimate Transactions by Location', fontsize=14, fontweight='bold')
            ax.legend()
            st.pyplot(fig)

        with analysis_tab3:
            st.subheader("Feature Correlation Heatmap")
            corr = st.session_state.df_preprocessed.corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr, cmap='coolwarm', center=0,
                        linewidths=0.5, annot=False, ax=ax)
            ax.set_title('Feature Correlations', fontsize=14, fontweight='bold')
            st.pyplot(fig)

    # =========================
    # Tab 4: Sample Data
    # =========================
    with tab4:
        st.header("Dataset Sample")
        show_type = st.radio("Show:", ["Preprocessed Data", "Original Data"], horizontal=True)
        if show_type == "Preprocessed Data":
            st.dataframe(st.session_state.df_preprocessed.head(100), use_container_width=True)
        else:
            st.dataframe(st.session_state.df_orig.head(100), use_container_width=True)

        st.markdown("---")
        st.subheader("Dataset Statistics")
        st.dataframe(st.session_state.df_preprocessed.describe(), use_container_width=True)


if __name__ == "__main__":
    main()
