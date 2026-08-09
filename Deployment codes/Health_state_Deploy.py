import pandas as pd
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, roc_curve,auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# Load Model 1
model = joblib.load(r"C:\MyWorkingDir\Workingpython\Msc Project\models\Health_state_classifier.pkl")

# Load encoder
encoder = joblib.load(r"C:\MyWorkingDir\Workingpython\Msc Project\models\Health_state_label_encoder.pkl")


# Load new sensor data
df = pd.read_csv(
    r"C:\MyWorkingDir\Workingpython\Msc Project\Data\mixed_data.csv"
)

# Save actual labels (if available)

actual_health = None

if "health_state" in df.columns:
    actual_health = df["health_state"].copy()

DROP_COLS = [
    "timestamp", "health_label", "health_state",
    "fault_type", "fault_code", 
    "load_factor",       # DATA LEAKAGE — encodes operating condition directly
    "motor_power_kw",    # HIGH CORRELATION with motor_current_a (r > 0.98)
    "temp_ambient_c",    # NO SIGNAL — pure Gaussian noise around 35°C
]

# Create input features
X = df.drop(columns=DROP_COLS)


print("Input Features:")
print(X.head())

#Predictions
predictions=model.predict(X)

predictions_names=encoder.inverse_transform(predictions)

#Predictions Confidence
probabilities= model.predict_proba(X)
confidence=probabilities.max(axis=1)*100

#Add Predictions
df['Predicted_Health_State']=predictions_names
df['Confidence (%)']=confidence.round(2)

if actual_health is not None:

    df["Actual_Health_State"] = actual_health

    actual_encoded = encoder.transform(actual_health)

    print("\nAccuracy")

    print(f"{accuracy_score(actual_encoded, predictions):.4f}")

    print("\nClassification Report\n")

    print(
        classification_report(
            actual_encoded,
            predictions,
            target_names=encoder.classes_
        ))

    cm = confusion_matrix(actual_encoded, predictions)

    plt.figure(figsize=(8,6))

    sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_
        )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Module 2 - Health State Classification")

    plt.tight_layout()

    plt.savefig(
            r"C:\MyWorkingDir\Workingpython\Msc Project\outputs\Module2_ConfusionMatrix.png",
            dpi=300
        )

    plt.show()




#Display Predictions

print("\nPrediction Results\n")

display_columns = [
    "Predicted_Health_State",
    "Confidence (%)"
]

if actual_health is not None:
    display_columns.insert(0, "Actual_Health_State")

print(df[display_columns].head(20))

# Health State Summary

print("\n========== Health State Summary ==========\n")

print(df["Predicted_Health_State"].value_counts().sort_index())

# ==========================================================
# Display sample predictions for each health state
# ==========================================================

states = [
    "healthy",
    "degraded",
    "warning",
    "critical",
    "fault"
]

for state in states:

    subset = df[df["Predicted_Health_State"] == state]

    print("\n" + "=" * 70)
    print(f"Health State: {state}")
    print(f"Number of Records: {len(subset)}")

    if subset.empty:
        print("No predictions for this health state.")
    else:
        print(
            subset[
                [
                    "Predicted_Health_State",
                    "Confidence (%)"
                ]
            ].head()
        )

# Save Predictions

output_file = (
    r"C:\MyWorkingDir\Workingpython\Msc Project\outputs\Module2_Predictions.csv"
)

df.to_csv(output_file, index=False)

print(f"\nPredictions saved successfully to:\n{output_file}")