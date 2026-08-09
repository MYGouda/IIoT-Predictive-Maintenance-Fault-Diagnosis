import pandas as pd
import joblib

#Load Prediction file
predictions= pd.read_csv(r'C:\MyWorkingDir\Workingpython\Msc Project\outputs\Module3_Predictions.csv')


# Maintenance Knowledge Base
maintenance_rules = pd.read_csv(r'C:\MyWorkingDir\Workingpython\Msc Project\Data\maintenance_rules.csv')
#Generate Recommendations


#Display Results
results = predictions.merge(
    maintenance_rules,
    left_on="Predicted_Fault",
    right_on="Fault_Type",
    how="left"
)

faults = [
    "bearing_wear",
    "imbalance",
    "overload",
    "phase_loss",
    "insulation_failure",
    "none"
]

for fault in faults:

    subset = results[results["Predicted_Fault"] == fault]

    print("\n" + "=" * 70)
    print(f"Fault Type: {fault}")
    print(f"Number of Records: {len(subset)}")

    if subset.empty:
        print("No predictions for this fault.")
    else:
        print(subset[[
            "Predicted_Fault",
            "Confidence (%)",
            "Priority",
            "Action",
            "Estimated_Cost_USD",
            "Estimated_Downtime_hr"
        ]].head())

print("\n========== Fault Summary ==========\n")

print(results["Predicted_Fault"].value_counts())

print("\n========== Priority Summary ==========\n")

print(results["Priority"].value_counts())

print("\n========== Total Estimated Maintenance Cost ==========\n")

print(f"${results['Estimated_Cost_USD'].sum():,.2f}")

print("\n========== Total Estimated Downtime ==========\n")

print(f"{results['Estimated_Downtime_hr'].sum():,.1f} hours")

#Saving to CSV
results.to_csv(
    r"C:\MyWorkingDir\Workingpython\Msc Project\outputs\Maintenance_Recommendations.csv",
    index=False
)

print("\nMaintenance recommendations saved successfully.")