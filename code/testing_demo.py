import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, roc_curve,auc, roc_auc_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

#Load Dataset
df=pd.read_csv('C:\MyWorkingDir\Workingpython\Msc Project\Data\mixed_data.csv')
print(df.head())
print(df.info())
print(df.describe())

#EDA
print(f'Checking for nulls: \n {df.isna().sum()}')

print(f'Checking for Duplicates: \n {df.duplicated().sum()}')

#Check for Class Distribution
print(f'Health Labels Class Distribution: \n{df['health_label'].value_counts()}')
print(f'health_state Class Distribution: \n{df['health_state'].value_counts()}')
print(f'fault_type Class Distribution: \n{df['fault_type'].value_counts()}')



print(df['health_label'].unique())
print(df['health_state'].unique())
print(df['fault_type'].unique())


df['health_binary']=df['health_state'].apply(
    lambda x:0 if x=='healthy' else 1
)

print(df['health_binary'].value_counts())

DROP_COLS = [
    "timestamp","health_label", "health_state",
    "fault_type", "fault_code", "health_binary",
    "load_factor",       # DATA LEAKAGE — encodes operating condition directly
    "motor_power_kw",    # HIGH CORRELATION with motor_current_a (r > 0.98)
    "temp_ambient_c",    # NO SIGNAL — pure Gaussian noise around 35°C
]
#Defining Features and target
X= df.drop(columns=DROP_COLS)

y= df['health_binary']

print(X.shape)
print(y.shape)

# #Label Encoding
# le= LabelEncoder()

# df['fault_type']= le.fit_transform(df['fault_type'])

# print(df[['fault_type']].head())

#Splitting data
X_train, X_test , y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=42, stratify=y)

#Scsling data
scaler=StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

k_values= range(1,26)
accuricies=[]
for k in k_values:
    knn= KNeighborsClassifier(k)
    knn.fit(X_train,y_train)
    y_pred_k= knn.predict(X_test)
    acc= accuracy_score(y_test,y_pred_k)
    accuricies.append(acc)
    print(f'k= {k:2f} | accuracy: {acc:.4f}')

#Plotting Accuracy vs K
plt.figure(figsize=(8,6))
plt.plot(k_values,accuricies, marker='o')
plt.xticks(k_values)
plt.title("KNN Accuracy vs Number of Neighbors")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.savefig("KNN_K_Selection.png", dpi=300)

plt.show()

#Selecting Best K value
best_kvalue=k_values[accuricies.index(max(accuricies))]
best_accuracy= max(accuricies)

print(f'best K value {best_kvalue:2d}')
print(f'best accuracy: {best_accuracy:.4f}')


#Train the model
knn_model=KNeighborsClassifier(n_neighbors=best_kvalue)
knn_model.fit(X_train,y_train)

#prediction
y_pred=knn_model.predict(X_test)

#Evaluating
print(f'KNN Model Accuracy Score: {accuracy_score(y_test,y_pred):.4f}')
print(f'Classification Report:\n {classification_report(y_test,y_pred)}')

#Visualizing Confusion Matrix
cm=confusion_matrix(y_test,y_pred)

sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',xticklabels=["Healthy","Faulty"],
    yticklabels=["Healthy","Faulty"])
plt.title('KNN Model Confusion Matrix')
plt.xlabel('predicted')
plt.ylabel('Actual')
plt.savefig('KNN Model Confusion Matrix.png')
plt.show()


# Probability predictions
y_prob = knn_model.predict_proba(X_test)[:, 1]

# ROC metrics
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(7,6))

plt.plot(fpr, tpr,
         linewidth=2,
         label=f"AUC = {roc_auc:.4f}")

plt.plot([0,1], [0,1], 'k--', label="KNN Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - KNN Binary Classification")

plt.legend(loc="lower right")

plt.grid(alpha=0.3)

plt.savefig("ROC_Curve.png", dpi=300)

plt.show()



prediction=knn_model.predict(X)
print(f'predicted output : {'Healthy'if prediction[0]==0 else 'Faulty' }')

import joblib
import pandas as pd


# Load trained model
knn_model = joblib.load("knn_model.pkl")
print(type(knn_model))
# Load scaler
scaler = joblib.load("scaler.pkl")

#Load Dataset
df=pd.read_csv('C:\MyWorkingDir\Workingpython\Msc Project\Data\mixed_data.csv')
print(df.head())
print(df.info())
print(df.describe())

df["health_binary"] = df["health_state"].apply(
    lambda x: 0 if x == "healthy" else 1
)

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

X = df.drop(columns=DROP_COLS)

X_scaled = scaler.transform(X)

prediction = knn_model.predict(X_scaled)

df["Prediction"] = prediction

df["Prediction"] = df["Prediction"].map({
    0: "Healthy",
    1: "Faulty"
})

df["Actual"] = df["health_binary"].map({
    0: "Healthy",
    1: "Faulty"
})

print(df[["Actual", "Prediction"]])

from sklearn.metrics import accuracy_score

print(
    "Demo Accuracy:",
    accuracy_score(df["health_binary"], prediction)
)

prediction=knn_model.predict(X_scaled)
print(f'predicted output : {'Healthy'if prediction[0]==1 else 'Faulty' }')