import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import seaborn as sns
import traceback

try:
    # Load dataset with larger sample size
    print("✅ Loading data...")
    df = pd.read_csv(r"C:\Users\siach\venv\Fraud_detection_project\data\creditcard.csv")
    
    # Use stratified sampling to preserve fraud ratio
    if len(df) > 100000:
        df, _ = train_test_split(df, train_size=100000, stratify=df['Class'], random_state=42)

    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:")
    print(df['Class'].value_counts())
    print(f"Fraud percentage: {(df['Class'].sum() / len(df)) * 100:.2f}%")

    # Feature engineering (Random Forest can handle more features better)
    print("✅ Creating additional features for Random Forest...")
    
    # Check if Amount column exists and handle safely
    if 'Amount' in df.columns:
        df['Amount_log'] = np.log1p(df['Amount'])  # Log transformation for Amount
        df['Amount_squared'] = df['Amount'] ** 2
        print("✅ Amount features created")
    else:
        print("⚠️ Warning: Amount column not found")
    
    # Check if Time column exists and handle safely
    if 'Time' in df.columns:
        df['Time_hour'] = (df['Time'] // 3600) % 24  # Extract hour of day
        df['Time_diff'] = df['Time'].diff().fillna(0)
        print("✅ Time features created")
    else:
        print("⚠️ Warning: Time column not found")

    # Define target and features
    y = df["Class"]
    X_full = df.drop("Class", axis=1)
    
    print(f"✅ Features shape: {X_full.shape}")
    print(f"✅ Target shape: {y.shape}")

    # For Random Forest, we can use more features (trees handle high dimensionality well)
    print("✅ Selecting top 15 features for Random Forest...")
    
    # Scale features for feature selection only (though RF doesn't need scaling for training)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_full)

    # Select top 15 features using SelectKBest
    selector = SelectKBest(score_func=f_classif, k=15)
    X_selected = selector.fit_transform(X_scaled, y)

    # Get selected feature names
    selected_mask = selector.get_support()
    selected_columns = X_full.columns[selected_mask]
    print("✅ Top 15 features selected:", list(selected_columns))

    # For Random Forest, use original (unscaled) data
    X_selected = X_full[selected_columns]
    print(f"✅ Selected features shape: {X_selected.shape}")

    # MODIFIED: Train-test split (70% train, 30% test) - KEY CHANGE HERE
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.3, stratify=y, random_state=42  # Changed from 0.2 to 0.3
    )
    
    print(f"✅ Train set shape: {X_train.shape}")
    print(f"✅ Test set shape: {X_test.shape}")
    print(f"📊 Train set fraud percentage: {(y_train.sum() / len(y_train)) * 100:.2f}%")
    print(f"📊 Test set fraud percentage: {(y_test.sum() / len(y_test)) * 100:.2f}%")

    # Check fraud count in training data
    fraud_count = sum(y_train == 1)
    print(f"📊 Fraud samples in training data: {fraud_count}")

    # MODIFIED: Apply more conservative SMOTE to reduce overfitting
    print("✅ Applying conservative SMOTE...")
    if fraud_count >= 6:
        k_neighbors = min(5, fraud_count - 1)
        # CHANGED: Use even more conservative ratio (0.4 instead of 0.6) to reduce overfitting
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=0.4)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"✅ SMOTE applied with k_neighbors={k_neighbors}")
        print(f"📊 Training data after SMOTE: {len(X_train_resampled)} samples")
        print(f"📊 Class distribution after SMOTE: {np.bincount(y_train_resampled)}")
    else:
        print(f"⚠️  Too few fraud samples ({fraud_count}) for SMOTE. Using original training data.")
        X_train_resampled, y_train_resampled = X_train, y_train

    # MODIFIED: More conservative Random Forest parameters to prevent overfitting
    print("✅ Training Random Forest model with conservative parameters...")
    final_model = RandomForestClassifier(
        n_estimators=50,            # REDUCED from 100 to 50 trees to prevent overfitting
        max_depth=8,                # REDUCED from 10 to 8 to prevent overfitting
        min_samples_split=30,       # INCREASED from 20 to 30
        min_samples_leaf=15,        # INCREASED from 10 to 15
        max_features='sqrt',        # Keep sqrt
        max_samples=0.8,            # NEW: Bootstrap with 80% of samples
        class_weight='balanced',    # Handle class imbalance
        random_state=42,
        n_jobs=-1                   # Use all CPU cores
    )

    final_model.fit(X_train_resampled, y_train_resampled)
    print("✅ Random Forest training completed!")

    # NEW: Cross-validation on training data to check for overfitting
    print("🔍 Performing cross-validation to check for overfitting...")
    cv_scores = cross_val_score(final_model, X_train_resampled, y_train_resampled, 
                               cv=5, scoring='roc_auc', n_jobs=-1)
    print(f"📊 Cross-validation ROC-AUC scores: {cv_scores}")
    print(f"📊 Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Evaluate on the test set
    print("🔍 Model Evaluation on test set:")
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]

    report_text = classification_report(y_test, y_pred, digits=4)
    print(report_text)

    # NEW: Calculate and display ROC-AUC
    test_roc_auc = roc_auc_score(y_test, y_proba)
    print(f"📊 Test set ROC-AUC: {test_roc_auc:.4f}")

    # NEW: Compare training vs test performance to detect overfitting
    train_pred = final_model.predict(X_train_resampled)
    train_proba = final_model.predict_proba(X_train_resampled)[:, 1]
    train_roc_auc = roc_auc_score(y_train_resampled, train_proba)
    
    print(f"\n🔍 Overfitting Check:")
    print(f"Training ROC-AUC: {train_roc_auc:.4f}")
    print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"Difference: {abs(train_roc_auc - test_roc_auc):.4f}")
    
    if abs(train_roc_auc - test_roc_auc) > 0.05:
        print("⚠️  WARNING: Potential overfitting detected (difference > 0.05)")
    else:
        print("✅ Good! No significant overfitting detected")

    # Check prediction diversity
    print("🔍 Checking model predictions diversity...")
    unique_probabilities = len(np.unique(y_proba))
    print(f"Unique probability values: {unique_probabilities}")
    print(f"Probability range: {y_proba.min():.8f} to {y_proba.max():.8f}")
    
    # NEW: Show probability distribution
    print(f"Probability distribution (test set):")
    prob_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    prob_counts = np.histogram(y_proba, bins=prob_bins)[0]
    for i, (start, end, count) in enumerate(zip(prob_bins[:-1], prob_bins[1:], prob_counts)):
        if count > 0:
            print(f"  {start:.1f}-{end:.1f}: {count} samples")

    # Feature importance analysis
    print("🔍 Feature Importance Analysis:")
    feature_importance = pd.DataFrame({
        'feature': selected_columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(feature_importance)

    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"✅ Created models directory: {models_dir}")

    # Alternative: Use absolute path
    models_dir_abs = "C:/Users/siach/venv/Fraud_detection_project/models"
    if not os.path.exists(models_dir_abs):
        os.makedirs(models_dir_abs)
        print(f"✅ Created models directory: {models_dir_abs}")

    # Plot feature importance
    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title('Top 10 Feature Importances - Random Forest')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(models_dir_abs, "feature_importance.png"))
        print("📊 Feature importance plot saved")
    except Exception as e:
        print(f"⚠️ Could not save feature importance plot: {e}")

    # Save classification report with additional metrics
    try:
        with open(os.path.join(models_dir_abs, "eval_report_rf.txt"), "w") as f:
            f.write("RANDOM FOREST MODEL EVALUATION REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Training Configuration:\n")
            f.write(f"- Train/Test Split: 70%/30%\n")
            f.write(f"- SMOTE Sampling Strategy: 0.4\n")
            f.write(f"- Random Forest Parameters: n_estimators=50, max_depth=8\n\n")
            f.write(f"Performance Metrics:\n")
            f.write(f"- Training ROC-AUC: {train_roc_auc:.4f}\n")
            f.write(f"- Test ROC-AUC: {test_roc_auc:.4f}\n")
            f.write(f"- Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n\n")
            f.write("Classification Report:\n")
            f.write(report_text)
            f.write("\n\nFeature Importance:\n")
            f.write(feature_importance.to_string())
        print("📄 Classification report saved")
    except Exception as e:
        print(f"⚠️ Could not save classification report: {e}")

    # Save confusion matrix
    try:
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay.from_estimator(final_model, X_test, y_test, cmap="Blues")
        plt.title("Confusion Matrix - Random Forest (70-30 Split)")
        plt.tight_layout()
        plt.savefig(os.path.join(models_dir_abs, "confusion_matrix_rf.png"))
        print("📊 Confusion matrix saved")
    except Exception as e:
        print(f"⚠️ Could not save confusion matrix: {e}")

    # Save model artifacts - THIS IS THE CRITICAL PART
    print("✅ Attempting to save model...")
    
    try:
        model_path = os.path.join(models_dir_abs, "random_forest_model.pkl")
        print(f"Saving model to: {model_path}")
        
        # Save model and selected features with additional metadata
        model_data = {
            'model': final_model,
            'selected_features': list(selected_columns),
            'train_test_split': 0.3,
            'smote_strategy': 0.4,
            'model_params': final_model.get_params(),
            'training_performance': {
                'train_roc_auc': train_roc_auc,
                'test_roc_auc': test_roc_auc,
                'cv_mean_roc_auc': cv_scores.mean(),
                'cv_std_roc_auc': cv_scores.std()
            }
        }
        joblib.dump(model_data, model_path)
        
        # Verify the file was created
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            print(f"✅ Model successfully saved! File size: {file_size} bytes")
        else:
            print("❌ Model file was not created!")
            
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        print("Full error traceback:")
        traceback.print_exc()

    # Additional Random Forest specific insights
    print("\n🔍 Random Forest Model Insights:")
    print(f"Number of trees: {final_model.n_estimators}")
    print(f"Max depth used: {max([tree.get_depth() for tree in final_model.estimators_])}")
    print(f"Min depth used: {min([tree.get_depth() for tree in final_model.estimators_])}")
    print(f"Average depth: {np.mean([tree.get_depth() for tree in final_model.estimators_]):.2f}")

    print("✅ Random Forest training pipeline completed successfully!")
    
    # List files in models directory to confirm
    print(f"\n📁 Files in models directory:")
    try:
        for file in os.listdir(models_dir_abs):
            file_path = os.path.join(models_dir_abs, file)
            file_size = os.path.getsize(file_path)
            print(f"  - {file} ({file_size} bytes)")
    except Exception as e:
        print(f"Could not list files: {e}")

except Exception as e:
    print(f"❌ FATAL ERROR: {e}")
    print("Full error traceback:")
    traceback.print_exc()