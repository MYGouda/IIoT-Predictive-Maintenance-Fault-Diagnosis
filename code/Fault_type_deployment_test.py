import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load Model 1
xgb_model = joblib.load(r"C:\MyWorkingDir\Workingpython\Msc Project\models\xgboost_fault_classifier.pkl")

# Load scaler
label_encoder = joblib.load(r"C:\MyWorkingDir\Workingpython\Msc Project\models\fault_label_encoder.pkl")


# Load new sensor data
df = pd.read_csv(
    r"C:\MyWorkingDir\Workingpython\Msc Project\Data\mixed_data.csv"
)


# Create binary label (only for comparing with actual result)
actual_fault = None

if "fault_type" in df.columns:
    actual_fault = df["fault_type"].copy()


# Same columns removed during training
DROP_COLS = [
    "timestamp",
    "health_label",
    "health_state",
    "fault_type",
    "fault_code",
    "load_factor",
    "motor_power_kw",
    "temp_ambient_c"
]


# Create input features
X = df.drop(columns=DROP_COLS)


print("Input Features:")
print(X.head())



# Prediction
prediction = xgb_model.predict(X)

prediction_names = label_encoder.inverse_transform(prediction)

# ============================================================
# Prediction probabilities
# ============================================================

probabilities = xgb_model.predict_proba(X)

confidence = probabilities.max(axis=1) * 100

# Add predictions to dataframe
# ============================================================

df["Predicted_Fault"] = prediction_names
df["Confidence (%)"] = confidence.round(2)

# ============================================================
# If actual labels exist, evaluate
# ============================================================

if actual_fault is not None:

    df["Actual_Fault"] = actual_fault

    actual_encoded = label_encoder.transform(actual_fault)

    print("\nAccuracy")

    print(f"{accuracy_score(actual_encoded, prediction):.4f}")

    print("\nClassification Report\n")

    print(
        classification_report(
            actual_encoded,
            prediction,
            target_names=label_encoder.classes_
        )
    )

    cm = confusion_matrix(actual_encoded, prediction)

    plt.figure(figsize=(8,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Fault Type Prediction")

    plt.tight_layout()

    plt.savefig("Module3_Test_ConfusionMatrix.png", dpi=300)

    plt.show()

# ============================================================
# Display predictions
# ============================================================

print("\nPrediction Results\n")

columns = ["Predicted_Fault", "Confidence (%)"]

if actual_fault is not None:
    columns.insert(0, "Actual_Fault")

display_columns = [
    "Predicted_Fault",
    "Confidence (%)"
]

if actual_fault is not None:
    display_columns.insert(0, "Actual_Fault")

print(df[df["Actual_Fault"] == "bearing_wear"][columns].head())

print(df[df["Actual_Fault"] == "imbalance"][columns].head())

print(df[df["Actual_Fault"] == "phase_loss"][columns].head())

df.to_csv(
    "Module3_Predictions.csv",
    index=False
)

print("Predictions saved successfully.")
