import sys
import os
import pandas as pd
import time
import warnings
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class EnhancedFraudDetectionSystem:
    """
    Enhanced Fraud Detection System with comprehensive analysis capabilities
    """
    
    def __init__(self, model_path: str = "C:/Users/siach/venv/Fraud_detection_project/models/random_forest_model.pkl"):
        """Initialize the enhanced fraud detection system"""
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.model_metadata = None
        self.load_model()
    
    def load_model(self):
        """Load the trained Random Forest model and metadata"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        try:
            model_data = joblib.load(self.model_path)
            
            if isinstance(model_data, dict):
                # New format with metadata
                self.model = model_data['model']
                self.feature_names = model_data['selected_features']
                self.model_metadata = model_data
                
                if 'training_performance' in model_data:
                    perf = model_data['training_performance']
                    print(f"✅ Model loaded successfully!")
                    print(f"📊 Test ROC-AUC: {perf.get('test_roc_auc', 'N/A'):.4f}")
                    print(f"📊 CV ROC-AUC: {perf.get('cv_mean_roc_auc', 'N/A'):.4f}")
            else:
                # Legacy format (tuple)
                self.model, self.feature_names = model_data
                print("✅ Legacy model format loaded")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def create_enhanced_features(self, df):
        """Create comprehensive feature engineering"""
        enhanced_df = df.copy()
        
        # Amount-based features
        if 'Amount' in df.columns:
            enhanced_df['Amount_log'] = np.log1p(df['Amount'])
            enhanced_df['Amount_squared'] = df['Amount'] ** 2
            enhanced_df['Amount_normalized'] = (df['Amount'] - df['Amount'].mean()) / (df['Amount'].std() + 1e-8)
        
        # Time-based features
        if 'Time' in df.columns:
            enhanced_df['Time_hour'] = (df['Time'] // 3600) % 24
            enhanced_df['Time_day'] = (df['Time'] // 86400) % 7
            enhanced_df['Time_diff'] = df['Time'].diff().fillna(0)
            
            # Business hours indicator
            enhanced_df['Business_hours'] = ((enhanced_df['Time_hour'] >= 9) & 
                                           (enhanced_df['Time_hour'] <= 17)).astype(int)
        
        # V-feature interactions (for top important features)
        v_cols = [col for col in df.columns if col.startswith('V')]
        if len(v_cols) >= 2:
            important_v = ['V10', 'V14', 'V17', 'V4', 'V11', 'V12']
            available_important = [v for v in important_v if v in v_cols]
            
            if len(available_important) >= 2:
                enhanced_df[f'{available_important[0]}_{available_important[1]}_interaction'] = (
                    df[available_important[0]] * df[available_important[1]]
                )
        
        return enhanced_df
    
    def predict_single(self, transaction_data, return_details=False):
        """Predict fraud for a single transaction with enhanced analysis"""
        if isinstance(transaction_data, dict):
            input_df = pd.DataFrame([transaction_data])
        else:
            input_df = transaction_data.copy()
        
        # Create enhanced features
        enhanced_df = self.create_enhanced_features(input_df)
        
        # Prepare model input
        model_features_df = pd.DataFrame()
        missing_features = []
        
        for feat in self.feature_names:
            if feat in enhanced_df.columns:
                model_features_df[feat] = enhanced_df[feat]
            else:
                model_features_df[feat] = 0
                missing_features.append(feat)
        
        # Get predictions
        model_input = model_features_df[self.feature_names]
        prediction = self.model.predict(model_input)[0]
        probabilities = self.model.predict_proba(model_input)[0]
        fraud_probability = probabilities[1]
        confidence = max(probabilities)
        
        # Risk assessment
        risk_level, risk_color = self._assess_risk(fraud_probability)
        
        result = {
            'prediction': prediction,
            'fraud_probability': fraud_probability,
            'confidence': confidence,
            'risk_level': risk_level,
            'risk_color': risk_color
        }
        
        if return_details:
            result.update({
                'missing_features': missing_features,
                'available_features': [f for f in self.feature_names if f in enhanced_df.columns],
                'feature_contribution': self._get_feature_contribution(model_input)
            })
        
        return result
    
    def predict_batch(self, transactions_df):
        """Predict fraud for multiple transactions"""
        enhanced_df = self.create_enhanced_features(transactions_df)
        
        model_features_df = pd.DataFrame(index=enhanced_df.index)
        missing_features = []
        
        for feat in self.feature_names:
            if feat in enhanced_df.columns:
                model_features_df[feat] = enhanced_df[feat]
            else:
                model_features_df[feat] = 0
                missing_features.append(feat)
        
        model_input = model_features_df[self.feature_names]
        predictions = self.model.predict(model_input)
        probabilities = self.model.predict_proba(model_input)
        
        results_df = pd.DataFrame({
            'prediction': predictions,
            'fraud_probability': probabilities[:, 1],
            'confidence': np.max(probabilities, axis=1)
        }, index=transactions_df.index)
        
        # Add risk levels
        results_df['risk_level'] = results_df['fraud_probability'].apply(
            lambda x: self._assess_risk(x)[0]
        )
        
        return results_df
    
    def _assess_risk(self, fraud_probability):
        """Assess risk level based on fraud probability"""
        if fraud_probability >= 0.8:
            return "HIGH RISK", "🔴"
        elif fraud_probability >= 0.5:
            return "MEDIUM RISK", "🟡"
        elif fraud_probability >= 0.2:
            return "LOW-MEDIUM RISK", "🟠"
        else:
            return "LOW RISK", "🟢"
    
    def _get_feature_contribution(self, model_input):
        """Get approximate feature contribution using feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            feature_values = model_input.iloc[0].values
            importances = self.model.feature_importances_
            
            contributions = []
            for i, (feat, val, imp) in enumerate(zip(self.feature_names, feature_values, importances)):
                contributions.append({
                    'feature': feat,
                    'value': val,
                    'importance': imp,
                    'contribution_score': abs(val) * imp
                })
            
            return sorted(contributions, key=lambda x: x['contribution_score'], reverse=True)[:5]
        return []
    
    def get_model_insights(self):
        """Display comprehensive model insights"""
        print(f"\n🌳 RANDOM FOREST MODEL INSIGHTS:")
        print("=" * 50)
        
        if self.model is not None:
            print(f"   Number of trees: {self.model.n_estimators}")
            print(f"   Max depth: {self.model.max_depth}")
            print(f"   Min samples split: {self.model.min_samples_split}")
            print(f"   Min samples leaf: {self.model.min_samples_leaf}")
            print(f"   Max features: {self.model.max_features}")
            print(f"   Class weight: {self.model.class_weight}")
            
            if hasattr(self.model, 'oob_score_'):
                print(f"   Out-of-bag score: {self.model.oob_score_:.4f}")
        
        print(f"   Total features used: {len(self.feature_names)}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 Top 5 Most Important Features:")
            for idx, row in importance_df.head(5).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")

class FraudDetectionApp:
    """
    Enhanced Fraud Detection Application with comprehensive analytics
    """
    
    def __init__(self, data_path: str, sample_size: int = 1000, random_state: int = 42):
        """Initialize the fraud detection app"""
        self.data_path = data_path
        self.sample_size = sample_size
        self.random_state = random_state
        self.df = None
        self.fraud_system = EnhancedFraudDetectionSystem()
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare the dataset"""
        print("✅ Loading dataset...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at: {self.data_path}")
        
        try:
            self.df = pd.read_csv(self.data_path).sample(
                self.sample_size, 
                random_state=self.random_state
            ).reset_index(drop=True)
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
        
        print(f"✅ Loaded {len(self.df)} transactions")
        print(f"✅ Fraud cases in sample: {self.df['Class'].sum() if 'Class' in self.df.columns else 'Unknown'}")
        
        return self.df
    
    def run_enhanced_fraud_detection(self, n_transactions: int = 30) -> Dict[str, Any]:
        """Run enhanced fraud detection with detailed analysis"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_prepare_data() first.")
        
        n_transactions = min(n_transactions, len(self.df))
        
        print(f"\n🔍 Running Enhanced Fraud Detection on {n_transactions} transactions...")
        print("=" * 70)
        
        # Get data for processing
        feature_data = self.df.iloc[:n_transactions].copy()
        
        # Batch prediction for efficiency
        start_time = time.time()
        batch_results = self.fraud_system.predict_batch(feature_data.drop('Class', axis=1, errors='ignore'))
        processing_time = time.time() - start_time
        
        # Display individual transaction results
        predictions = []
        probabilities = []
        confidences = []
        risk_levels = []
        
        for i in range(n_transactions):
            prediction = batch_results.iloc[i]['prediction']
            fraud_prob = batch_results.iloc[i]['fraud_probability']
            confidence = batch_results.iloc[i]['confidence']
            risk_level = batch_results.iloc[i]['risk_level']
            
            predictions.append(prediction)
            probabilities.append(fraud_prob)
            confidences.append(confidence)
            risk_levels.append(risk_level)
            
            # Get risk color
            _, risk_color = self.fraud_system._assess_risk(fraud_prob)
            
            # Display transaction
            status = "🚨 FRAUD DETECTED!" if prediction == 1 else "✅ Legitimate"
            amount = feature_data.iloc[i]['Amount']
            
            print(f"Transaction {i+1:2d}: {status}")
            print(f"   Amount: ${amount:.2f}")
            print(f"   Fraud Probability: {fraud_prob:.4f}")
            print(f"   Confidence: {confidence:.4f}")
            print(f"   Risk Level: {risk_color} {risk_level}")
            
            # Add recommendations for high-risk transactions
            if fraud_prob >= 0.5:
                self._provide_transaction_recommendations(fraud_prob, amount)
            
            print()
            time.sleep(0.03)  # Visual delay
        
        # Compile results
        results = {
            'total_transactions': n_transactions,
            'fraud_detected': sum(predictions),
            'legitimate_transactions': n_transactions - sum(predictions),
            'fraud_rate': (sum(predictions) / n_transactions) * 100,
            'processing_time': processing_time,
            'predictions': predictions,
            'fraud_probabilities': probabilities,
            'confidences': confidences,
            'risk_levels': risk_levels,
            'avg_fraud_probability': np.mean(probabilities),
            'avg_confidence': np.mean(confidences),
            'high_risk_count': sum(1 for p in probabilities if p >= 0.8),
            'medium_risk_count': sum(1 for p in probabilities if 0.5 <= p < 0.8),
            'low_medium_risk_count': sum(1 for p in probabilities if 0.2 <= p < 0.5),
            'low_risk_count': sum(1 for p in probabilities if p < 0.2)
        }
        
        return results
    
    def _provide_transaction_recommendations(self, fraud_prob: float, amount: float):
        """Provide specific recommendations for suspicious transactions"""
        if fraud_prob >= 0.8:
            print("   💡 RECOMMENDATIONS:")
            print("      🚨 IMMEDIATE ACTION: Block transaction")
            print("      📞 Contact customer immediately")
            print("      🔍 Review recent transaction history")
        elif fraud_prob >= 0.5:
            print("   💡 RECOMMENDATIONS:")
            print("      ⚠️ CAUTION: Additional verification required")
            print("      🔐 Consider step-up authentication")
            if amount > 1000:
                print("      💰 High amount - escalate to supervisor")
    
    def display_enhanced_summary(self, results: Dict[str, Any]):
        """Display comprehensive summary with risk analysis"""
        print(f"\n{'='*70}")
        print(f"📊 ENHANCED FRAUD DETECTION SUMMARY")
        print(f"{'='*70}")
        
        # Basic statistics
        print(f"📈 TRANSACTION STATISTICS:")
        print(f"   Total processed: {results['total_transactions']}")
        print(f"   🚨 Fraud detected: {results['fraud_detected']}")
        print(f"   ✅ Legitimate: {results['legitimate_transactions']}")
        print(f"   📊 Fraud rate: {results['fraud_rate']:.1f}%")
        
        # Risk breakdown
        print(f"\n🎯 RISK LEVEL BREAKDOWN:")
        print(f"   🔴 High Risk (≥80%): {results['high_risk_count']}")
        print(f"   🟡 Medium Risk (50-80%): {results['medium_risk_count']}")
        print(f"   🟠 Low-Medium Risk (20-50%): {results['low_medium_risk_count']}")
        print(f"   🟢 Low Risk (<20%): {results['low_risk_count']}")
        
        # Performance metrics
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   🎯 Average fraud probability: {results['avg_fraud_probability']:.4f}")
        print(f"   💪 Average confidence: {results['avg_confidence']:.4f}")
        print(f"   ⏱️ Processing time: {results['processing_time']:.3f} seconds")
        print(f"   🚀 Transactions/second: {results['total_transactions']/results['processing_time']:.1f}")
        
        # Model information
        print(f"\n🌳 MODEL INFORMATION:")
        print(f"   Model: Random Forest ({self.fraud_system.model.n_estimators} trees)")
        print(f"   Features used: {len(self.fraud_system.feature_names)}")
        print(f"   Max depth: {self.fraud_system.model.max_depth}")
    
    def compare_with_actual_enhanced(self, results: Dict[str, Any], n_transactions: int):
        """Enhanced comparison with actual labels"""
        if 'Class' not in self.df.columns:
            print(f"\n⚠️ No actual labels available for comparison")
            return
        
        try:
            actual_labels = self.df.iloc[:n_transactions]['Class'].values
            predictions = results['predictions']
            probabilities = results['fraud_probabilities']
            
            # Calculate comprehensive metrics
            accuracy = accuracy_score(actual_labels, predictions)
            precision = precision_score(actual_labels, predictions, zero_division=0)
            recall = recall_score(actual_labels, predictions, zero_division=0)
            f1 = f1_score(actual_labels, predictions, zero_division=0)
            roc_auc = roc_auc_score(actual_labels, probabilities)
            
            # Confusion matrix components
            tp = sum(1 for i in range(n_transactions) if predictions[i] == 1 and actual_labels[i] == 1)
            fp = sum(1 for i in range(n_transactions) if predictions[i] == 1 and actual_labels[i] == 0)
            tn = sum(1 for i in range(n_transactions) if predictions[i] == 0 and actual_labels[i] == 0)
            fn = sum(1 for i in range(n_transactions) if predictions[i] == 0 and actual_labels[i] == 1)
            
            print(f"\n🎯 ENHANCED PERFORMANCE ANALYSIS:")
            print("=" * 50)
            print(f"📊 CLASSIFICATION METRICS:")
            print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            print(f"   ROC-AUC: {roc_auc:.4f}")
            
            print(f"\n📈 CONFUSION MATRIX:")
            print(f"   True Positives: {tp}")
            print(f"   False Positives: {fp}")
            print(f"   True Negatives: {tn}")
            print(f"   False Negatives: {fn}")
            
            # Business impact analysis
            print(f"\n💼 BUSINESS IMPACT ANALYSIS:")
            if fp > 0:
                print(f"   ⚠️ False Positives: {fp} legitimate transactions flagged")
                print(f"   💰 Potential customer friction: {fp} cases")
            if fn > 0:
                print(f"   🚨 False Negatives: {fn} fraud cases missed")
                print(f"   💸 Potential fraud losses: {fn} cases")
            
            # Risk calibration analysis
            print(f"\n🎯 RISK CALIBRATION:")
            risk_buckets = [0.2, 0.5, 0.8, 1.0]
            for i, threshold in enumerate(risk_buckets):
                if i == 0:
                    mask = np.array(probabilities) < threshold
                    bucket_name = f"Low Risk (<{threshold*100:.0f}%)"
                else:
                    mask = (np.array(probabilities) >= risk_buckets[i-1]) & (np.array(probabilities) < threshold)
                    bucket_name = f"Risk {risk_buckets[i-1]*100:.0f}%-{threshold*100:.0f}%"
                
                if mask.sum() > 0:
                    actual_fraud_rate = np.array(actual_labels)[mask].mean()
                    print(f"   {bucket_name}: {mask.sum()} transactions, {actual_fraud_rate*100:.1f}% actual fraud")
            
        except Exception as e:
            print(f"❌ Error in enhanced comparison: {str(e)}")
    
    def analyze_suspicious_transactions(self, n_transactions: int = 30):
        """Analyze the most suspicious transactions in detail"""
        if self.df is None:
            return
        
        feature_data = self.df.iloc[:n_transactions].drop('Class', axis=1, errors='ignore')
        batch_results = self.fraud_system.predict_batch(feature_data)
        
        # Get top suspicious transactions
        suspicious_indices = batch_results.nlargest(5, 'fraud_probability').index
        
        print(f"\n🔍 TOP 5 MOST SUSPICIOUS TRANSACTIONS:")
        print("=" * 60)
        
        for idx in suspicious_indices:
            transaction = self.df.iloc[idx]
            result = self.fraud_system.predict_single(
                transaction.drop('Class', errors='ignore').to_dict(), 
                return_details=True
            )
            
            print(f"\n📋 Transaction {idx + 1}:")
            print(f"   Amount: ${transaction['Amount']:.2f}")
            print(f"   Fraud Probability: {result['fraud_probability']:.4f}")
            print(f"   Risk Level: {result['risk_color']} {result['risk_level']}")
            
            if 'Class' in transaction:
                actual = "FRAUD" if transaction['Class'] == 1 else "LEGITIMATE"
                print(f"   Actual Label: {actual}")
            
            # Show top contributing features
            if result['feature_contribution']:
                print(f"   Top Contributing Features:")
                for contrib in result['feature_contribution'][:3]:
                    print(f"     {contrib['feature']}: {contrib['value']:.4f}")

def main():
    """Enhanced main execution function"""
    # Configuration
    DATA_PATH = r"C:\Users\siach\venv\Fraud_detection_project\data\creditcard.csv"
    N_TRANSACTIONS = 30
    SAMPLE_SIZE = 1000
    
    try:
        print("🚀 Enhanced Fraud Detection System Starting...")
        print("=" * 60)
        
        # Initialize app
        app = FraudDetectionApp(
            data_path=DATA_PATH,
            sample_size=SAMPLE_SIZE,
            random_state=42
        )
        
        # Load and prepare data
        app.load_and_prepare_data()
        
        # Display model insights
        app.fraud_system.get_model_insights()
        
        # Run enhanced fraud detection
        results = app.run_enhanced_fraud_detection(n_transactions=N_TRANSACTIONS)
        
        # Display enhanced results
        app.display_enhanced_summary(results)
        app.compare_with_actual_enhanced(results, N_TRANSACTIONS)
        
        # Analyze suspicious transactions
        app.analyze_suspicious_transactions(N_TRANSACTIONS)
        
        print(f"\n✅ Enhanced fraud detection completed successfully!")
        print("🎯 System ready for production deployment!")
        
    except Exception as e:
        print(f"\n❌ Application Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)