import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


#Load Dataset
#C:\MyWorkingDir\Workingpython\Msc Project\Data\sensor_plant_simulator.py
df = pd.read_csv(r"c:\MyWorkingDir\Workingpython\Msc Project\Data\sensor_data.csv")

#EDA
print(df.info())

print(df.isnull().sum())

print(df.duplicated().sum())

print(df["fault_type"].value_counts())

#Label Encoding
le= LabelEncoder()
df['fault_type']=le.fit_transform(df['fault_type'])

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

#Defining Features and Target
X=df.drop(columns=DROP_COLS)
y= df["fault_type"]

print('X shape: ', X.shape)
print('y shape: ', y.shape)

#Splitting Train and Test data
X_train, X_test, y_train,y_test= train_test_split(X, y , test_size=0.2, random_state=42, stratify=y)

#Training Model
xgb_model= XGBClassifier(
    objective='multi:softprob',
    num_class=len(le.classes_),
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42
)

xgb_model.fit(X_train,y_train, verbose= False)

#Prediction
y_pred=xgb_model.predict(X_test)

k_folds= StratifiedKFold(n_splits=5, shuffle= True, random_state=42)

Scores= cross_val_score(estimator=xgb_model, X=X, y=y, cv= k_folds, scoring='accuracy')

for i, score in enumerate(Scores,1):
    print(f"Fold {i}: {score:.4f}")

print(f"Mean Accuracy : {score.mean():.4f}")
print(f"Std Accuracy  : {score.std():.4f}")

#Evaluation
print(f'Accuracy: {accuracy_score(y_test,y_pred)}')
print(f'Classification Report: \n {classification_report(y_test,y_pred, target_names=le.classes_)}')

#Plotting Confusion Matrix
cm=confusion_matrix(y_test,y_pred)

sns.heatmap(cm,cmap='Blues', annot=True,fmt='d', xticklabels=le.classes_,yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Accuracy')
plt.title("XGBoost Fault Classification")
plt.savefig('XGBoost Fault Classification.png')
plt.tight_layout()
plt.show()

joblib.dump(
    xgb_model,
    r"C:\MyWorkingDir\Workingpython\Msc Project\models\xgboost_fault_classifier.pkl"
)

joblib.dump(
    le,
    r"C:\MyWorkingDir\Workingpython\Msc Project\models\fault_label_encoder.pkl"
)

print("Model saved successfully.")