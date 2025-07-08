import joblib
import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionSystem:
    def __init__(self, model_path="C:/Users/siach/venv/Fraud_detection_project/models/random_forest_model.pkl"):
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
                self.model = model_data['model']
                self.feature_names = model_data['selected_features']
                self.model_metadata = model_data
                
                if 'training_performance' in model_data:
                    perf = model_data['training_performance']
                    print(f"✅ Model loaded successfully!")
                    print(f"📊 Test ROC-AUC: {perf.get('test_roc_auc', 'N/A'):.4f}")
            else:
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
        
        # V-feature interactions (if multiple V features exist)
        v_cols = [col for col in df.columns if col.startswith('V')]
        if len(v_cols) >= 2:
            # Create some interaction features for top important V features
            important_v = ['V10', 'V14', 'V17', 'V4', 'V11', 'V12']
            available_important = [v for v in important_v if v in v_cols]
            
            if len(available_important) >= 2:
                enhanced_df[f'{available_important[0]}_{available_important[1]}_interaction'] = (
                    df[available_important[0]] * df[available_important[1]]
                )
        
        return enhanced_df
    
    def predict_single(self, transaction_data, return_details=False):
        """
        Predict fraud for a single transaction
        
        Args:
            transaction_data: dict or DataFrame with transaction features
            return_details: bool, whether to return detailed analysis
        """
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
    
    def analyze_transaction(self, transaction_data):
        """Comprehensive transaction analysis"""
        result = self.predict_single(transaction_data, return_details=True)
        
        print("🔍 FRAUD DETECTION ANALYSIS")
        print("=" * 50)
        
        # Main prediction
        status = "🚨 FRAUD DETECTED" if result['prediction'] == 1 else "✅ LEGITIMATE"
        print(f"Prediction: {status}")
        print(f"Fraud Probability: {result['fraud_probability']:.4f}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Risk Level: {result['risk_color']} {result['risk_level']}")
        
        # Feature analysis
        if result['missing_features']:
            print(f"\n⚠️ Missing Features ({len(result['missing_features'])}): {result['missing_features'][:5]}...")
        
        print(f"✅ Available Features: {len(result['available_features'])}/{len(self.feature_names)}")
        
        # Top contributing features
        if result['feature_contribution']:
            print(f"\n📊 Top Contributing Features:")
            for contrib in result['feature_contribution']:
                print(f"   {contrib['feature']}: {contrib['value']:.4f} (importance: {contrib['importance']:.4f})")
        
        # Recommendations
        self._provide_recommendations(result)
        
        return result
    
    def _provide_recommendations(self, result):
        """Provide actionable recommendations"""
        print(f"\n💡 RECOMMENDATIONS:")
        
        fraud_prob = result['fraud_probability']
        
        if fraud_prob >= 0.8:
            print("   🚨 IMMEDIATE ACTION: Block transaction and contact customer")
            print("   🔍 Investigate: Review recent transaction history")
            print("   📞 Verify: Direct customer contact recommended")
        elif fraud_prob >= 0.5:
            print("   ⚠️ CAUTION: Additional verification required")
            print("   🔐 Security: Consider step-up authentication")
            print("   📊 Monitor: Flag for enhanced monitoring")
        elif fraud_prob >= 0.2:
            print("   👀 WATCH: Monitor for patterns")
            print("   📈 Track: Add to risk monitoring queue")
        else:
            print("   ✅ PROCEED: Normal processing")
            print("   📝 Log: Standard transaction logging")
    
    def get_model_summary(self):
        """Get comprehensive model summary"""
        print("🌳 RANDOM FOREST MODEL SUMMARY")
        print("=" * 50)
        
        if self.model_metadata and isinstance(self.model_metadata, dict):
            model = self.model_metadata['model']
            print(f"🔧 Model Configuration:")
            print(f"   Trees: {model.n_estimators}")
            print(f"   Max Depth: {model.max_depth}")
            print(f"   Min Samples Split: {model.min_samples_split}")
            print(f"   Min Samples Leaf: {model.min_samples_leaf}")
            
            if 'training_performance' in self.model_metadata:
                perf = self.model_metadata['training_performance']
                print(f"\n📊 Performance Metrics:")
                print(f"   Training ROC-AUC: {perf.get('train_roc_auc', 'N/A'):.4f}")
                print(f"   Test ROC-AUC: {perf.get('test_roc_auc', 'N/A'):.4f}")
                print(f"   CV Mean ROC-AUC: {perf.get('cv_mean_roc_auc', 'N/A'):.4f}")
        
        print(f"\n🎯 Feature Information:")
        print(f"   Total Features: {len(self.feature_names)}")
        print(f"   Feature Names: {self.feature_names}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n📈 Top 5 Most Important Features:")
            for idx, row in importance_df.head(5).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
    
    def validate_on_sample(self, sample_size=1000):
        """Validate model on a random sample"""
        try:
            df = pd.read_csv("C:/Users/siach/venv/Fraud_detection_project/data/creditcard.csv")
            sample_df = df.sample(sample_size, random_state=42)
            
            features = sample_df.drop('Class', axis=1)
            actual = sample_df['Class']
            
            results = self.predict_batch(features)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            accuracy = accuracy_score(actual, results['prediction'])
            precision = precision_score(actual, results['prediction'], zero_division=0)
            recall = recall_score(actual, results['prediction'], zero_division=0)
            f1 = f1_score(actual, results['prediction'], zero_division=0)
            roc_auc = roc_auc_score(actual, results['fraud_probability'])
            
            print(f"🔍 VALIDATION RESULTS (n={sample_size})")
            print("=" * 40)
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"ROC-AUC:   {roc_auc:.4f}")
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc
            }
            
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return None

# Usage Examples
def main():
    # Initialize the fraud detection system
    fraud_detector = FraudDetectionSystem()
    
    # Get model summary
    fraud_detector.get_model_summary()
    
    print("\n" + "="*60 + "\n")
    
    # Example 1: Analyze a suspicious transaction
    suspicious_transaction = {
        'Time': 54000,  # 15 hours into dataset
        'Amount': 5000.00,  # High amount
        'V1': -2.5,
        'V2': 1.8,
        'V3': -3.2,
        'V4': 2.1,
        'V5': -1.9
    }
    
    print("🔍 ANALYZING SUSPICIOUS TRANSACTION:")
    fraud_detector.analyze_transaction(suspicious_transaction)
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Analyze a normal transaction
    normal_transaction = {
        'Time': 10000,
        'Amount': 25.50,
        'V1': 0.1,
        'V2': -0.2,
        'V3': 0.3,
        'V4': -0.1,
        'V5': 0.2
    }
    
    print("🔍 ANALYZING NORMAL TRANSACTION:")
    fraud_detector.analyze_transaction(normal_transaction)
    
    print("\n" + "="*60 + "\n")
    
    # Validate on sample
    fraud_detector.validate_on_sample(500)

if __name__ == "__main__":
    main()