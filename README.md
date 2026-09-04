# Predictive Maintenance for Industrial Motors using ESP32, MPU-6500 and Machine Learning

## Overview

This project presents an Industrial IoT (IIoT) predictive maintenance system for monitoring the health of industrial motors using sensor data and machine learning.

The system is built around a four stage decision pipeline rather than a single classifier. It starts with a binary health check, moves on to a five level severity estimate, then identifies the most likely root cause fault, and finally converts that diagnosis into a costed, prioritised maintenance recommendation. This structure is meant to mirror how a maintenance engineer actually reasons about a machine, not just produce a single label.

The project covers the full workflow of a predictive maintenance solution: sensor data acquisition, data preprocessing and feature engineering, model training and evaluation, and a rule based maintenance recommendation engine. It also includes a physical hardware proof of concept, built on a real ESP32 board, that confirms the sensor readings the models depend on can actually be acquired from live hardware rather than only from simulated data.

---

## Project Objectives

- Design an Industrial IoT condition monitoring architecture.
- Collect motor operating data across temperature, vibration, electrical and rotational sensors.
- Perform data preprocessing, feature engineering and data leakage screening.
- Develop and justify machine learning models for each stage of the maintenance decision.
- Detect abnormal operating conditions and estimate health severity.
- Identify the most likely fault type behind an abnormal reading.
- Generate maintenance recommendations with estimated cost, downtime and remaining useful life.
- Validate the sensor acquisition assumptions with real, physically wired hardware.

---

## System Architecture

```text
Industrial Motor
      |
      v
ESP32 Data Acquisition
      |
      +-- MPU-6500 (acceleration and gyroscope)
      +-- Temperature and humidity sensor
      +-- Voltage sensor
      +-- Current sensor (simulated dataset only, see Hardware section)
      |
      v
Sensor Data Collection
      |
      v
Data Preprocessing and Feature Engineering
      |
      v
Module 1 - Binary Health Detection (KNN)
   Healthy vs Faulty
      |
      v
Module 2 - Health State Classification (Random Forest)
   Healthy -> Degraded -> Warning -> Critical -> Fault
      |
      v
Module 3 - Fault Type Classification (XGBoost)
   Bearing Wear, Imbalance, Overload, Phase Loss, Insulation Failure, None
      |
      v
Module 4 - Maintenance Recommendation Engine
   Priority, Recommended Action, Estimated Cost, Estimated Downtime, Remaining Useful Life
```

---

## Hardware

This project has two distinct hardware stories, and it is worth being precise about which is which.

### Physically built and tested

The following was wired, programmed and exercised on real hardware, and is documented with firmware listings and captured serial output in the project report:

- DOIT ESP32 DEVKIT V1 development board, programmed with the Arduino IDE.
- DHT11 temperature and humidity sensor, wired to GPIO 4, tested under a controlled rising temperature load.
- MPU-6500 six axis accelerometer and gyroscope, wired over I2C (SDA on GPIO 21, SCL on GPIO 22, address 0x68), tested with the board held stationary and then moved by hand to confirm both channels respond correctly.
- A resistive voltage divider (30 kilohm and 7.5 kilohm resistors, 5 to 1 ratio) into ADC1 on GPIO 34, tested through a stable reading and a deliberate source disconnection event.

These three sensor nodes were exercised independently on separate firmware uploads. They have not yet been combined into a single logging sketch running on one motor mounted device, and no dataset has yet been collected from an actual operating motor. Closing that gap, by integrating all three nodes and collecting a labelled dataset from a physical motor test rig, is the leading item of future work.

### Simulated for the training dataset

The machine learning dataset used to train and evaluate the four modules is a simulated sensor stream, generated to approximate the statistical behaviour of a real induction motor under a defined set of fault conditions and operating loads. It includes a motor current feature that has not yet been validated against a physically wired current sensor, which is flagged here rather than left implicit.

---

## Dataset

The dataset (`sensor_data.csv`) contains 4,556 observations across 36 columns, collected at five second intervals, with no missing values and no duplicate rows.

### Health state distribution

| Health state | Count | Share |
|---|---|---|
| Healthy | 2,575 | 56.5% |
| Degraded | 740 | 16.2% |
| Warning | 532 | 11.7% |
| Critical | 414 | 9.1% |
| Fault | 295 | 6.5% |

### Fault type distribution

| Fault type | Count | Share |
|---|---|---|
| None | 2,865 | 62.9% |
| Bearing wear | 540 | 11.9% |
| Imbalance | 450 | 9.9% |
| Phase loss | 295 | 6.5% |
| Overload | 232 | 5.1% |
| Insulation failure | 174 | 3.8% |

### Feature groups

- Operational: load factor, motor speed (RPM).
- Electrical: motor current, motor voltage, motor power.
- Temperature: ambient, surface, bearing.
- Vibration: motor vibration, vibration severity.
- MPU-6500 accelerometer: mean, RMS, peak and standard deviation on X, Y and Z axes.
- MPU-6500 gyroscope: mean, RMS and peak on X, Y and Z axes.

### Features excluded from model training

Three features were screened out of the model input space on quantitative grounds rather than by assumption:

- `load_factor`, because it is set by the same simulation logic that determines fault severity and would let a model partly infer the label from how the data was generated rather than from genuine sensor evidence.
- `motor_power_kw`, because it correlates with `motor_current_a` at r = 0.975 across the dataset, making it largely redundant once current is included.
- `temp_ambient_c`, because its mean and spread are essentially constant across all five health states (34.98 to 35.02 degrees Celsius), indicating it carries no useful signal.

---

## Machine Learning Pipeline

### Module 1: Binary Health Detection

- Algorithm: K-Nearest Neighbours, with k selected by sweeping k = 1 to 25 against held out accuracy, features standardised with `StandardScaler`.
- Reported evaluation snapshot: 88.5% accuracy, ROC-AUC of 0.994, recall of 76.9% on the faulty class.
- Known limitation: the captured confusion matrix reflects a 26 sample evaluation snapshot rather than the full expected 912 sample test partition from the 80/20 split. This is flagged as a data integrity item to resolve, not smoothed over.

### Module 2: Health State Classification

- Algorithm: Random Forest (`n_estimators=200, max_depth=12, min_samples_split=5, min_samples_leaf=2, random_state=42`).
- Result: 100% accuracy on both the internal 912 record test split and an independent 128 record deployment file, across all five severity classes.
- This near perfect result is treated as a finding to investigate rather than a simple success. The severity label is closely tied to the fault type label by how the simulator generates data, which may explain why a flexible ensemble can recover it almost exactly.

### Module 3: Fault Type Classification

- Algorithm: XGBoost (`objective=multi:softprob, n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42`).
- Result: 100% accuracy on the independent 128 record deployment file across all six fault categories, including the rarest class, insulation failure, with only six examples.
- The same caution applied to Module 2 applies here regarding near ceiling accuracy on a simulated dataset.

### Module 4: Maintenance Recommendation Engine

- Implementation: a deterministic rule based join between the Module 3 fault prediction and an external maintenance knowledge base (`maintenance_rules.csv`), rather than a further learned model, so that the mapping from diagnosis to recommendation stays transparent and auditable.
- Output per prediction: priority level, recommended corrective action, estimated repair cost, estimated downtime and an estimated remaining useful life in days.
- On the 128 record deployment window, this produced a projected total maintenance cost of $25,400 and 220 hours of total downtime.

---

## What Makes This Different From a Single Fault Classifier

Most published work in this space evaluates a single classification task in isolation, usually either binary detection or flat multi class fault typing. This project instead chains four stages together, binary detection, severity staging, root cause diagnosis and costed recommendation, and documents the reasoning behind each algorithm choice and each excluded feature rather than treating them as defaults. It also treats the near perfect accuracy achieved on the later stages as something to interrogate, not just report, given that the training data is simulated.

---

## Technologies Used

**Programming**
- Python

**Data analysis**
- Pandas, NumPy

**Machine learning**
- Scikit-learn, XGBoost

**Visualisation**
- Matplotlib, Seaborn

**IoT and firmware**
- ESP32 (DOIT ESP32 DEVKIT V1), Arduino IDE, DHT11, MPU-6500

**Model persistence**
- Joblib

---

## Project Structure

```text
Predictive-Maintenance/
|
|-- Data/
|   |-- sensor_data.csv
|   |-- maintenance_rules.csv
|   |-- mixed_data.csv
|
|-- Data Samples/
|
|-- code/
|   |-- EDA/
|   |-- Module1-Binary Classification/
|   |-- Module2-Health_state_classification/
|   |-- Module3-Fault_Type classification/
|   |-- Module4-Cost Analysis/
|
|-- Deployment codes/
|
|-- firmware/
|   |-- DHT11.ino
|   |-- MPU6500.ino
|   |-- Voltage.ino
|
|-- models/
|
|-- outputs/
|   |-- images/
|   |-- prediction_output files/
|
|-- README.md
|-- requirements.txt
```

---

## Outputs

The project produces:

- Trained machine learning models (`.pkl`), saved with joblib.
- Confusion matrices and a ROC curve for each classification stage.
- Feature and permutation importance analysis.
- Prediction files for the severity and fault type modules.
- A maintenance recommendation report, including projected cost and downtime.
- Firmware listings and captured serial output from the physical sensor testing.

---

## Documentation

A full MSc dissertation report accompanies this repository, covering the literature this project builds on, the reasoning behind each modelling decision, a full evaluation of all four modules, a critical discussion of the near ceiling accuracy observed on Modules 2 and 3, and the physical hardware proof of concept summarised above.

---

## Future Improvements

- Integrate the three individually tested sensor nodes into a single firmware running on one motor mounted device.
- Collect a genuinely physical, fault labelled dataset from a motor test rig, rather than relying solely on the simulated dataset.
- Add a physical current sensor to close the one sensing gap not yet covered by the hardware proof of concept.
- Re-evaluate Module 1 on its full 912 sample test partition and tune its decision threshold for higher recall on the faulty class.
- Explore continuous degradation scoring as an alternative to the five discrete severity bands.
- Cloud dashboard integration, for example Power BI or Grafana.
- Integration with CMMS or ERP systems for a live, editable maintenance knowledge base.
- Explainable AI methods such as SHAP or LIME for prediction interpretation.

---

## Author

**Marina Youssef Gouda**

MSc Artificial Intelligence and Computer Science

GitHub: https://github.com/MYGouda
