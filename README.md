# Predictive Maintenance for Industrial Motors using ESP32, MPU-6500 and Machine Learning

## Overview

This project presents a complete **Industrial IoT (IIoT) Predictive Maintenance System** for monitoring the health of industrial motors using real-time sensor data and Machine Learning.

The system collects operational data from an **ESP32 microcontroller** connected to an **MPU-6500 IMU**, temperature, voltage, and current sensors. The acquired data is processed through a multi-stage machine learning pipeline that detects machine faults, estimates machine health severity, identifies the root cause of failures, and generates maintenance recommendations with estimated repair cost and downtime.

The project demonstrates the complete workflow of an industrial predictive maintenance solution, from sensor data acquisition to AI-driven maintenance decision support.

---

# Project Objectives

* Design an Industrial IoT condition monitoring system.
* Collect real-time motor operating data using multiple sensors.
* Perform data preprocessing and feature engineering.
* Develop machine learning models for predictive maintenance.
* Detect abnormal operating conditions.
* Estimate machine health severity.
* Identify machine fault types.
* Generate maintenance recommendations.
* Estimate maintenance cost and equipment downtime.

---

# System Architecture

```text
                     Industrial Motor
                            │
                            ▼
                  ESP32 Data Acquisition
                            │
     ┌───────────────┬───────────────┬───────────────┐
     │               │               │
     ▼               ▼               ▼
 MPU-6500      Temperature      Voltage & Current
Acceleration      Sensors            Sensors
 & Gyroscope
     │               │               │
     └───────────────┴───────────────┘
                     │
                     ▼
             Sensor Data Collection
                     │
                     ▼
             Data Preprocessing
                     │
                     ▼
            Feature Engineering
                     │
                     ▼
       Module 1 — Binary Classification
          Healthy  ↔  Faulty
                     │
                     ▼
      Module 2 — Health State Classification
 Healthy → Degraded → Warning → Critical → Fault
                     │
                     ▼
      Module 3 — Fault Type Classification
 Bearing Wear
 Imbalance
 Phase Loss
 Overload
 Insulation Failure
                     │
                     ▼
 Module 4 — Maintenance Recommendation Engine
 Priority
 Maintenance Action
 Estimated Repair Cost
 Estimated Downtime
```

---

# Hardware Components

* ESP32 Development Board
* MPU-6500 6-Axis IMU
* Temperature Sensor
* Voltage Sensor
* Current Sensor
* Industrial Motor (simulated)
* Breadboard
* Jumper Wires
* USB Power Supply

---

# Dataset

The project uses a simulated industrial motor dataset containing **4,556 observations** collected at **5-second intervals**.

### Features

#### Operational Parameters

* Load Factor
* Motor Speed (RPM)

#### Electrical Measurements

* Motor Current
* Motor Voltage
* Motor Power

#### Temperature Measurements

* Ambient Temperature
* Surface Temperature
* Bearing Temperature

#### Vibration Measurements

* Motor Vibration
* Vibration Severity

#### MPU-6500 Accelerometer Features

* Mean
* RMS
* Peak
* Standard Deviation

Measured on:

* X-axis
* Y-axis
* Z-axis

#### MPU-6500 Gyroscope Features

* Mean
* RMS
* Peak

Measured on:

* X-axis
* Y-axis
* Z-axis

---

# Machine Learning Pipeline

## Module 1 — Binary Health Detection

Objective:

Determine whether the machine is:

* Healthy
* Faulty

Algorithm:

* K-Nearest Neighbors (KNN)

Output:

* Binary health prediction
* Prediction confidence
* Confusion matrix
* ROC curve

---

## Module 2 — Health State Classification

Objective:

Estimate the machine health severity.

Classes:

* Healthy
* Degraded
* Warning
* Critical
* Fault

Algorithm:

* Random Forest Classifier

Output:

* Health state prediction
* Prediction confidence
* Classification report
* Confusion matrix

---

## Module 3 — Fault Type Classification

Objective:

Identify the root cause of machine failures.

Classes:

* Bearing Wear
* Imbalance
* Phase Loss
* Overload
* Insulation Failure
* None

Algorithm:

* XGBoost Classifier

Output:

* Fault prediction
* Prediction confidence
* Classification report
* Confusion matrix

---

## Module 4 — Maintenance Recommendation Engine

Objective:

Generate actionable maintenance recommendations based on predicted fault type.

The recommendation engine provides:

* Maintenance priority
* Recommended corrective action
* Estimated repair cost (USD)
* Estimated downtime (hours)

Maintenance knowledge is stored externally in a CSV-based knowledge base for easy updates and scalability.

---

# Technologies Used

## Programming

* Python

## Data Analysis

* Pandas
* NumPy

## Machine Learning

* Scikit-learn
* XGBoost

## Visualization

* Matplotlib
* Seaborn

## IoT

* ESP32
* Arduino IDE
* MPU-6500

## Model Persistence

* Joblib

---

# Project Structure

```text
Predictive-Maintenance/
│
├── Data/
│
├── Data Samples/
│
├── code/
│   ├── EDA/
│   ├── Module1-Binary Classification/
│   ├── Module2-Health_state_classification/
│   ├── Module3-Fault_Type classification/
│   └── Module4-Cost Analysis/
│
├── Deployment codes/
│
├── models/
│
├── outputs/
│   ├── images/
│   └── prediction_output files/
│
├── README.md
└── requirements.txt
```

---

# Outputs

The project automatically generates:

* Trained Machine Learning models (.pkl)
* Confusion matrices
* ROC curves
* Feature importance analysis
* Prediction reports
* Health state predictions
* Fault type predictions
* Maintenance recommendation reports

---

# Future Improvements

* Real-time ESP32 sensor streaming
* Edge AI deployment
* Time-series forecasting using LSTM
* Cloud dashboard integration (Power BI/Grafana)
* Integration with CMMS/ERP systems
* Explainable AI (SHAP/LIME) for prediction interpretation

---

# Author

**Marina Youssef Gouda**

**MSc Artificial Intelligence & Computer Science**

GitHub: https://github.com/MYGouda

