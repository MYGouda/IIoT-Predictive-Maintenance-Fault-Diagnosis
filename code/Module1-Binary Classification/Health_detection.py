import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, roc_curve,auc, roc_auc_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

#Load Dataset
df=pd.read_csv(r'C:\MyWorkingDir\Workingpython\Msc Project\Data\sensor_data.csv')
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
    "timestamp", "health_label", "health_state",
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


#Splitting data
X_train, X_test , y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=42, stratify=y)

#Scaling data
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

joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model saved successfully!")

#prediction
y_pred=knn_model.predict(X_test)

#Evaluating
print(f'KNN Model Accuracy Score: {accuracy_score(y_test,y_pred):.4f}')
print(f'Classification Report:\n {classification_report(y_test,y_pred)}')

#Visualizing Confusion Matrix
cm=confusion_matrix(y_test,y_pred)

sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.title('KNN Model Confusion Matrix')
plt.xlabel('predicted')
plt.ylabel('Actual')
plt.savefig('KNN Model Confusion Matrix.png')
plt.show()

#Calculate Feature importance
result=permutation_importance(
    knn_model,
    X_test,y_test,
    scoring='accuracy',
    n_repeats=20,
    random_state=42
)

importance_df=pd.DataFrame({'feature':X.columns, 'importance':result.importances_mean})
importance_df= importance_df.sort_values(by='importance', ascending=False)
print(importance_df)

#ROC-AUC Curve 
# # roc_auc=roc_auc_score(y_test,knn_model.predict_proba(X_test)[:,1])
# # print(f'Roc_Auc score: {roc_auc:.4f}')
# y_prob= knn_model.predict_proba(X_test)[:,1]

# fpr, tpr, threshold = roc_curve(y_test,y_prob)
# roc_auc=auc(fpr,tpr)

# plt.figure()
# plt.plot(fpr,tpr, label=f'AUC= {roc_auc:.4f}')
# plt.plot([0,1],[0,1])

# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Roc Curve')
# plt.legend()
# plt.savefig('Roc Curve.png')
# plt.show()

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

plt.plot([0,1], [0,1], 'k--', label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - KNN Binary Classification")

plt.legend(loc="lower right")

plt.grid(alpha=0.3)

plt.savefig("ROC_Curve.png", dpi=300)

plt.show()

