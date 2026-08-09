import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, roc_curve,auc, roc_auc_score
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

#Label Encoding for health states
le= LabelEncoder()

df["health_state"]=le.fit_transform(df["health_state"])

print('Class Mapping')
for i, c in enumerate(le.classes_):
    print(i,'-->',c)

DROP_COLS = [
    "timestamp", "health_label", "health_state",
    "fault_type", "fault_code", 
    "load_factor",       # DATA LEAKAGE — encodes operating condition directly
    "motor_power_kw",    # HIGH CORRELATION with motor_current_a (r > 0.98)
    "temp_ambient_c",    # NO SIGNAL — pure Gaussian noise around 35°C
]
#Defining Features and target
X= df.drop(columns=DROP_COLS)

y= df['health_state']

print(X.shape)
print(y.shape)

#Training and splitting data
X_train, X_test, y_train,y_test= train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)


#Train the model
rf_model= RandomForestClassifier(n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1)


#Fitting Data
rf_model.fit(X_train,y_train)

#Prediction
y_pred= rf_model.predict(X_test)

#Creating CVs to validate
k_folds= StratifiedKFold(n_splits=5, shuffle= True, random_state=42)

Scores= cross_val_score(estimator=rf_model, X=X, y=y, cv= k_folds, scoring='accuracy')

for i, score in enumerate(Scores,1):
    print(f"Fold {i}: {score:.4f}")

print(f"Mean Accuracy : {score.mean():.4f}")
print(f"Std Accuracy  : {score.std():.4f}")


#Evaluating
print(f'Accuracy Score: {accuracy_score(y_test, y_pred):.4f}')

print(f'Classification Report:\n {classification_report(y_test,y_pred, target_names=le.classes_)}')

#Plotting Confusion MAtrix
cm=confusion_matrix(y_test,y_pred)

sns.heatmap(cm, cmap='Blues', annot=True,fmt='d', xticklabels=le.classes_,yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Accuracy')
plt.title("Random Forest - Health State Classification")
plt.savefig('Random Forest - Health State Classification.png')
plt.tight_layout()
plt.show()

joblib.dump(
    rf_model,
    r"C:\MyWorkingDir\Workingpython\Msc Project\models\Health_state_classifier.pkl"
)

joblib.dump(
    le,
    r"C:\MyWorkingDir\Workingpython\Msc Project\models\Health_state_label_encoder.pkl"
)

print("Model saved successfully.")