import pandas as pd
import joblib


# Load Model 1
model1 = joblib.load("knn_model.pkl")

# Load scaler
scaler = joblib.load("scaler.pkl")


# Load new sensor data
df = pd.read_csv(
    r"C:\MyWorkingDir\Workingpython\Msc Project\Data\mixed_data.csv"
)


# Create binary label (only for comparing with actual result)
df["health_binary"] = df["health_state"].apply(
    lambda x: 0 if x == "healthy" else 1
)


# Same columns removed during training
DROP_COLS = [
    "timestamp",
    "health_label",
    "health_state",
    "fault_type",
    "fault_code",
    "health_binary",
    "load_factor",
    "motor_power_kw",
    "temp_ambient_c"
]


# Create input features
X = df.drop(columns=DROP_COLS)


print("Input Features:")
print(X.head())


# Apply same scaling used during training
X_scaled = scaler.transform(X)


# Prediction
prediction = model1.predict(X_scaled)


# Convert prediction
df["Prediction"] = prediction

df["Prediction"] = df["Prediction"].map({
    0: "Healthy",
    1: "Faulty"
})


# Compare with actual
df["Actual"] = df["health_binary"].map({
    0: "Healthy",
    1: "Faulty"
})


print(df[["Actual","Prediction"]].head(20))