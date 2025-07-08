import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import RobustScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Set page config FIRST
st.set_page_config(page_title="Fraud Detection", layout="wide", page_icon="💳")

# ---------- Load Model & Metadata ----------
@st.cache_resource
def load_model():
    """Load the Random Forest model and metadata"""
    model_path = "models/random_forest_model.pkl"  # Updated path to match training script
    
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please run the training script first to create the model.")
        st.info("Expected path: models/random_forest_model.pkl")
        st.stop()
    
    try:
        model_data = joblib.load(model_path)
        return model_data
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Load model data
model_data = load_model()
model = model_data['model']  # Updated key to match training script
selected_features = model_data['selected_features']
model_params = model_data.get('model_params', {})
training_performance = model_data.get('training_performance', {})

# ---------- Title and Model Info ----------
st.title("Fraud Detection System")

# Show model information
with st.expander("📊 Model Information", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌲 Model Details")
        st.write(f"**Algorithm:** Random Forest")
        st.write(f"**Number of Trees:** {model_params.get('n_estimators', 'N/A')}")
        st.write(f"**Max Depth:** {model_params.get('max_depth', 'N/A')}")
        st.write(f"**Features Used:** {len(selected_features)}")
    
    with col2:
        st.subheader("📈 Training Performance")
        if training_performance:
            st.write(f"**Training ROC-AUC:** {training_performance.get('train_roc_auc', 'N/A'):.4f}")
            st.write(f"**Test ROC-AUC:** {training_performance.get('test_roc_auc', 'N/A'):.4f}")
            st.write(f"**CV ROC-AUC:** {training_performance.get('cv_mean_roc_auc', 'N/A'):.4f}")

# ---------- Feature Engineering Function ----------
def create_features(df):
    """Create the same features as in the training script"""
    df_processed = df.copy()
    
    # Amount features (if Amount column exists)
    if 'Amount' in df_processed.columns:
        df_processed['Amount_log'] = np.log1p(df_processed['Amount'])
        df_processed['Amount_squared'] = df_processed['Amount'] ** 2
        st.info("✅ Amount features created")
    else:
        st.warning("⚠️ Amount column not found - some features may be missing")
    
    # Time features (if Time column exists)
    if 'Time' in df_processed.columns:
        df_processed['Time_hour'] = (df_processed['Time'] // 3600) % 24
        df_processed['Time_diff'] = df_processed['Time'].diff().fillna(0)
        st.info("✅ Time features created")
    else:
        st.warning("⚠️ Time column not found - some features may be missing")
    
    return df_processed

# ---------- Auto Prediction Function ----------
def predict_fraud(df, threshold=50):
    """Predict fraud for given dataframe"""
    try:
        # Feature engineering
        df_processed = create_features(df)
        
        # Check for required features
        missing_features = [feat for feat in selected_features if feat not in df_processed.columns]
        
        if missing_features:
            st.error("❌ **Missing Required Features:**")
            for feat in missing_features:
                st.write(f"- {feat}")
            return None
        
        # Select only the features used in training
        X_input = df_processed[selected_features]
        
        # Make predictions
        predictions_proba = model.predict_proba(X_input)[:, 1]
        predictions_binary = (predictions_proba >= threshold/100).astype(int)
        
        # Add predictions to dataframe
        results_df = df_processed.copy()
        results_df['Fraud_Probability'] = np.round(predictions_proba * 100, 2)
        results_df['Fraud_Prediction'] = predictions_binary
        results_df['Risk_Level'] = pd.cut(
            results_df['Fraud_Probability'], 
            bins=[0, 25, 50, 75, 100], 
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        return results_df, predictions_proba, predictions_binary
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return None

# ---------- Visualization Functions ----------
def create_fraud_distribution_chart(results_df):
    """Create fraud probability distribution chart"""
    fig = px.histogram(
        results_df, 
        x='Fraud_Probability', 
        nbins=20,
        title='Distribution of Fraud Probability Scores',
        labels={'Fraud_Probability': 'Fraud Probability (%)', 'count': 'Number of Transactions'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(
        xaxis_title="Fraud Probability (%)",
        yaxis_title="Number of Transactions",
        showlegend=False
    )
    return fig

def create_risk_level_pie_chart(results_df):
    """Create risk level distribution pie chart"""
    risk_counts = results_df['Risk_Level'].value_counts()
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#8b0000']
    
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title='Risk Level Distribution',
        color_discrete_sequence=colors
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_fraud_vs_amount_scatter(results_df):
    """Create scatter plot of fraud probability vs amount"""
    if 'Amount' in results_df.columns:
        fig = px.scatter(
            results_df, 
            x='Amount', 
            y='Fraud_Probability',
            color='Risk_Level',
            title='Fraud Probability vs Transaction Amount',
            labels={'Amount': 'Transaction Amount ($)', 'Fraud_Probability': 'Fraud Probability (%)'},
            color_discrete_map={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c', 'Critical': '#8b0000'}
        )
        fig.update_layout(
            xaxis_title="Transaction Amount ($)",
            yaxis_title="Fraud Probability (%)"
        )
        return fig
    return None

def create_time_analysis_chart(results_df):
    """Create time-based analysis chart"""
    if 'Time_hour' in results_df.columns:
        hourly_fraud = results_df.groupby('Time_hour').agg({
            'Fraud_Probability': 'mean',
            'Fraud_Prediction': 'sum'
        }).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Average Fraud Probability by Hour', 'Number of Fraud Cases by Hour'),
            vertical_spacing=0.1
        )
        
        # Average fraud probability by hour
        fig.add_trace(
            go.Scatter(
                x=hourly_fraud['Time_hour'],
                y=hourly_fraud['Fraud_Probability'],
                mode='lines+markers',
                name='Avg Fraud Probability',
                line=dict(color='#e74c3c', width=3)
            ),
            row=1, col=1
        )
        
        # Number of fraud cases by hour
        fig.add_trace(
            go.Bar(
                x=hourly_fraud['Time_hour'],
                y=hourly_fraud['Fraud_Prediction'],
                name='Fraud Cases',
                marker_color='#e74c3c'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Fraud Analysis by Time of Day',
            height=600,
            showlegend=False
        )
        fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
        fig.update_yaxes(title_text="Fraud Probability (%)", row=1, col=1)
        fig.update_yaxes(title_text="Number of Fraud Cases", row=2, col=1)
        
        return fig
    return None

def create_feature_importance_chart():
    """Create feature importance chart"""
    try:
        # Get feature importance from the model
        feature_importance = model.feature_importances_
        feature_names = selected_features
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=True)
        
        # Take top 10 features
        top_features = importance_df.tail(10)
        
        fig = px.bar(
            top_features,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 10 Most Important Features',
            labels={'Importance': 'Feature Importance', 'Feature': 'Features'},
            color='Importance',
            color_continuous_scale='Reds'
        )
        fig.update_layout(
            xaxis_title="Feature Importance",
            yaxis_title="Features",
            showlegend=False
        )
        return fig
    except Exception as e:
        st.error(f"Error creating feature importance chart: {str(e)}")
        return None

def create_prediction_confidence_gauge(fraud_prob):
    """Create gauge chart for prediction confidence"""
    confidence = max(fraud_prob, 100 - fraud_prob)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Prediction Confidence (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

# ---------- Input Method Selection ----------
st.subheader("📊 Transaction Input")
input_method = st.radio(
    "Choose input method:",
    ["Upload CSV File", "Manual Entry"],
    horizontal=True
)

uploaded_file = None
manual_data = None

if input_method == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload CSV file containing transaction data", type=["csv"])
else:
    st.subheader("✍️ Manual Transaction Entry")
    
    # Prediction settings for manual entry
    threshold = st.slider("Fraud Probability Threshold (%)", 
                        min_value=0, max_value=100, value=50, step=1)
    
    # Create manual entry form
    with st.form("manual_entry_form"):
        st.write("**Enter transaction details:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            amount = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=0.01)
            time_val = st.number_input("Time (seconds)", min_value=0, value=3600, step=1)
            v1 = st.number_input("Transaction Amount Deviation (V1)", value=0.0, format="%.6f")
            v2 = st.number_input("Login Frequency Score (V2)", value=0.0, format="%.6f")
            v3 = st.number_input("Device Risk Score (V3)", value=0.0, format="%.6f")
            v4 = st.number_input("IP Trust Score (V4)", value=0.0, format="%.6f")
            v5 = st.number_input("Location Mismatch Level (V5)", value=0.0, format="%.6f")
            v6 = st.number_input("Merchant Trust Index (V6)", value=0.0, format="%.6f")
            v7 = st.number_input("Time Since Last Transaction (V7)", value=0.0, format="%.6f")
            v8 = st.number_input("Velocity Score (V8)", value=0.0, format="%.6f")
            v9 = st.number_input("Unusual Purchase Pattern (V9)", value=0.0, format="%.6f")
            v10 = st.number_input("Anomaly Score (V10)", value=0.0, format="%.6f")

        
        with col2:
            v11 = st.number_input("Geolocation Deviation (V11)", value=0.0, format="%.6f")
            v12 = st.number_input("Card Present Flag (V12)", value=0.0, format="%.6f")
            v13 = st.number_input("Transaction Time Consistency (V13)", value=0.0, format="%.6f")
            v14 = st.number_input("Historical Risk Trend (V14)", value=0.0, format="%.6f")
            v15 = st.number_input("Proxy IP Score (V15)", value=0.0, format="%.6f")
            v16 = st.number_input("Unusual Time of Day (V16)", value=0.0, format="%.6f")
            v17 = st.number_input("High-Risk Country Flag (V17)", value=0.0, format="%.6f")
            v18 = st.number_input("Velocity of Login Attempts (V18)", value=0.0, format="%.6f")
            v19 = st.number_input("Customer Behavior Shift (V19)", value=0.0, format="%.6f")
            v20 = st.number_input("First-Time Merchant (V20)", value=0.0, format="%.6f")
            v21 = st.number_input("Rapid Spending Spike (V21)", value=0.0, format="%.6f")

        
        with col3:
            v22 = st.number_input("Mismatch in Billing/Shipping (V22)", value=0.0, format="%.6f")
            v23 = st.number_input("VPN/Anonymizer Flag (V23)", value=0.0, format="%.6f")
            v24 = st.number_input("Repeated Declines History (V24)", value=0.0, format="%.6f")
            v25 = st.number_input("Multiple Login Locations (V25)", value=0.0, format="%.6f")
            v26 = st.number_input("Unusual Browser Fingerprint (V26)", value=0.0, format="%.6f")
            v27 = st.number_input("Device Change Score (V27)", value=0.0, format="%.6f")
            v28 = st.number_input("Unusual Transaction Volume (V28)", value=0.0, format="%.6f")

        
        # Submit button
        submitted = st.form_submit_button("🚀 Analyze Transaction", type="primary")
        
        if submitted:
            # Create manual transaction data
            manual_data = pd.DataFrame({
                'Time': [time_val],
                'Amount': [amount],
                **{f'V{i}': [locals()[f'v{i}']] for i in range(1, 29)}
            })
            
            st.success("✅ Transaction created successfully!")
            st.dataframe(manual_data, use_container_width=True)
            
            # AUTOMATICALLY PREDICT
            st.subheader("🔮 Fraud Analysis Results")
            with st.spinner("Analyzing transaction for fraud..."):
                prediction_results = predict_fraud(manual_data, threshold)
                
                if prediction_results is not None:
                    results_df, predictions_proba, predictions_binary = prediction_results
                    
                    # Display results
                    fraud_prob = predictions_proba[0] * 100
                    is_fraud = predictions_binary[0]
                    risk_level = results_df['Risk_Level'].iloc[0]
                    
                    # Create prominent display for single transaction
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        color = "red" if is_fraud else "green"
                        status = "FRAUD DETECTED" if is_fraud else "LEGITIMATE"
                        st.markdown(f"### 🚨 **<span style='color:{color}'>{status}</span>**", unsafe_allow_html=True)
                    with col2:
                        st.metric("Fraud Probability", f"{fraud_prob:.2f}%")
                    with col3:
                        risk_color = {"Low": "green", "Medium": "orange", "High": "red", "Critical": "darkred"}.get(str(risk_level), "gray")
                        st.markdown(f"### 🎯 **<span style='color:{risk_color}'>{risk_level} Risk</span>**", unsafe_allow_html=True)
                    
                    # ========== NEW: ADD VISUALIZATION FOR SINGLE TRANSACTION ==========
                    st.subheader("📊 Prediction Visualization")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Confidence gauge
                        confidence_fig = create_prediction_confidence_gauge(fraud_prob)
                        st.plotly_chart(confidence_fig, use_container_width=True)
                    
                    with col2:
                        # Feature importance chart
                        feature_fig = create_feature_importance_chart()
                        if feature_fig:
                            st.plotly_chart(feature_fig, use_container_width=True)
                    
                    # Additional insights
                    st.subheader("📊 Transaction Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Transaction Summary:**")
                        st.write(f"• Amount: ${amount:,.2f}")
                        st.write(f"• Time: {time_val:,} seconds ({(time_val // 3600) % 24:.0f}:00 hour)")
                        st.write(f"• Risk Assessment: {risk_level}")
                        
                        if fraud_prob > 75:
                            st.error("⚠️ HIGH RISK: Immediate review recommended")
                        elif fraud_prob > 50:
                            st.warning("⚠️ MEDIUM RISK: Manual review suggested")
                        else:
                            st.success("✅ LOW RISK: Transaction appears normal")
                    
                    with col2:
                        st.write("**Model Confidence:**")
                        confidence = max(fraud_prob, 100 - fraud_prob)
                        st.write(f"• Prediction Confidence: {confidence:.1f}%")
                        st.write(f"• Threshold Used: {threshold}%")
                        
                        # Show some key features that might influence decision
                        if amount > 1000:
                            st.write("• High transaction amount detected")
                        if (time_val // 3600) % 24 < 6 or (time_val // 3600) % 24 > 22:
                            st.write("• Unusual transaction time")

# Handle CSV upload (FIXED FILTERING LOGIC)
if uploaded_file is not None:
    try:
        # Load the data
        df = pd.read_csv(uploaded_file)
        
        st.subheader("📄 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Transactions", len(df))
        with col2:
            st.metric("Number of Features", len(df.columns))
        with col3:
            if 'Class' in df.columns:
                fraud_count = df['Class'].sum()
                st.metric("Known Fraud Cases", fraud_count)
        
        # Show data preview
        st.write("**Data Preview:**")
        st.dataframe(df.head(), use_container_width=True)
        
        # Prediction settings for CSV
        col1, col2 = st.columns(2)
        with col1:
            threshold = st.slider("Fraud Probability Threshold (%) ", 
                                min_value=0, max_value=100, value=50, step=1)
        with col2:
            show_all = st.checkbox("Show all transactions", value=False)
        
        if st.button("🚀 Run Fraud Detection", type="primary"):
            prediction_results = predict_fraud(df, threshold)
            
            if prediction_results is not None:
                results_df, predictions_proba, predictions_binary = prediction_results
                
                # Store results in session state to persist filtering
                st.session_state.results_df = results_df
                st.session_state.show_all = show_all
                
        # Display results if they exist in session state
        if 'results_df' in st.session_state:
            results_df = st.session_state.results_df
            
            # Show bulk results summary
            st.subheader("📊 Prediction Results")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Transactions", len(results_df))
            with col2:
                fraud_detected = (results_df['Fraud_Prediction'] == 1).sum()
                st.metric("Fraud Detected", fraud_detected)
            with col3:
                fraud_rate = (fraud_detected / len(results_df)) * 100
                st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
            with col4:
                avg_risk = results_df['Fraud_Probability'].mean()
                st.metric("Avg Risk Score", f"{avg_risk:.1f}%")
            
            # ========== COMPREHENSIVE VISUALIZATIONS ==========
            st.subheader("📊 Fraud Detection Visualizations")
            
            # Row 1: Distribution and Risk Level
            col1, col2 = st.columns(2)
            
            with col1:
                # Fraud probability distribution
                dist_fig = create_fraud_distribution_chart(results_df)
                st.plotly_chart(dist_fig, use_container_width=True)
            
            with col2:
                # Risk level pie chart
                pie_fig = create_risk_level_pie_chart(results_df)
                st.plotly_chart(pie_fig, use_container_width=True)
            
            # Row 2: Scatter plot and Time analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Amount vs Fraud probability scatter
                scatter_fig = create_fraud_vs_amount_scatter(results_df)
                if scatter_fig:
                    st.plotly_chart(scatter_fig, use_container_width=True)
                else:
                    st.info("Amount data not available for scatter plot")
            
            with col2:
                # Feature importance
                feature_fig = create_feature_importance_chart()
                if feature_fig:
                    st.plotly_chart(feature_fig, use_container_width=True)
            
            # Row 3: Time analysis (full width)
            time_fig = create_time_analysis_chart(results_df)
            if time_fig:
                st.plotly_chart(time_fig, use_container_width=True)
            
            # FIXED: Show detailed results table with proper filtering
            st.subheader("📋 Detailed Results")
            
            # Create filter containers
            col1, col2, col3 = st.columns(3)
            
            # Get unique values for filters - handle NaN values properly
            unique_risk_levels = results_df['Risk_Level'].dropna().unique().tolist()
            unique_risk_levels.sort()
            
            with col1:
                risk_filter = st.selectbox(
                    "Filter by Risk Level", 
                    ["All"] + unique_risk_levels,
                    key="risk_filter"
                )
            with col2:
                fraud_filter = st.selectbox(
                    "Filter by Prediction", 
                    ["All", "Fraud", "Legitimate"],
                    key="fraud_filter"
                )
            with col3:
                prob_filter = st.slider(
                    "Min Fraud Probability", 
                    0, 100, 0, 
                    key="prob_filter"
                )
            
            # Apply filters step by step with debugging
            filtered_df = results_df.copy()
            
            # Show filter stats
            st.write(f"**Starting with {len(filtered_df)} transactions**")
            
            # Risk level filter
            if risk_filter != "All":
                filtered_df = filtered_df[filtered_df['Risk_Level'] == risk_filter]
                st.write(f"After risk filter ({risk_filter}): {len(filtered_df)} transactions")
            
            # Fraud prediction filter
            if fraud_filter == "Fraud":
                filtered_df = filtered_df[filtered_df['Fraud_Prediction'] == 1]
                st.write(f"After fraud filter (Fraud): {len(filtered_df)} transactions")
            elif fraud_filter == "Legitimate":
                filtered_df = filtered_df[filtered_df['Fraud_Prediction'] == 0]
                st.write(f"After fraud filter (Legitimate): {len(filtered_df)} transactions")
            
            # Probability filter
            if prob_filter > 0:
                filtered_df = filtered_df[filtered_df['Fraud_Probability'] >= prob_filter]
                st.write(f"After probability filter (>={prob_filter}%): {len(filtered_df)} transactions")
            
            # Check if any data remains after filtering
            if len(filtered_df) == 0:
                st.warning("⚠️ No transactions match the selected filters. Please adjust your filter criteria.")
            else:
                # Display filtered results
                st.success(f"✅ Showing {len(filtered_df)} transactions after filtering")
                
                # Choose columns to display
                display_columns = ['Time', 'Amount', 'Fraud_Probability', 'Fraud_Prediction', 'Risk_Level']
                available_columns = [col for col in display_columns if col in filtered_df.columns]
                
                if st.session_state.get('show_all', False):
                    # Show all columns
                    st.dataframe(filtered_df, use_container_width=True, height=400)
                else:
                    # Show only key columns
                    st.dataframe(filtered_df[available_columns], use_container_width=True, height=400)
                
                # Download filtered results
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Results as CSV",
                    data=csv,
                    file_name=f"fraud_detection_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_filtered"
                )
        
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.info("Please ensure your CSV file is properly formatted")

# Show instructions when no input is provided
if uploaded_file is None and manual_data is None:
    st.subheader("🚀 Quick Start")
    st.write("""
    **Choose your input method:**
    - 📁 **Upload CSV**: For batch processing multiple transactions with comprehensive visualizations
    - ✍️ **Manual Entry**: For single transaction analysis with instant prediction and charts
    
    The system uses a Random Forest model trained on credit card transaction data and provides:
    - Real-time fraud detection
    - Interactive visualizations
    - Risk level analysis
    - Feature importance insights
    """)
    
    # Show key features only
    with st.expander("🎯 Key Model Features", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("- 🌲 Random Forest (50 trees)")
            st.write("- 📊 Uses 15 most important features")
            st.write("- 📈 Interactive charts and graphs")
        with col2:
            st.write("- ⚖️ Balanced class weights")
            st.write("- 🎯 Prevents overfitting (max depth: 8)")
            st.write("- 🔍 Comprehensive risk analysis")

# ---------- Footer ----------
st.markdown("---")
st.markdown("*Fraud Detection System powered by Random Forest Machine Learning with Advanced Visualizations*")