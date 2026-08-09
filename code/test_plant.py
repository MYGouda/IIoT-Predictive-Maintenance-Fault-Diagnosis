#!/usr/bin/env python3
"""
plant_sensor_simulator_v3.py
==============================
Reproduces the exact structure and statistics of the scripted fault-injection
log you provided (sensor_data.csv, 4556 rows): NOT a smooth wear-ramp, but a
repeating cycle of a healthy baseline period followed by one of two fixed
"fault chains", each chain a sequence of discrete fault episodes with tightly
characterized value distributions -- exactly as measured from your file.

Schema (identical column order to your file):

    timestamp,load_factor,health_label,health_state,fault_type,fault_code,
    motor_speed_rpm,motor_current_a,motor_voltage_v,motor_power_kw,
    motor_vibration_mm_s,temp_ambient_c,temp_surface_c,temp_bearing_c,
    accel_x_mean_ms2,accel_y_mean_ms2,accel_z_mean_ms2,
    accel_x_rms_ms2,accel_y_rms_ms2,accel_z_rms_ms2,
    accel_x_peak_ms2,accel_y_peak_ms2,accel_z_peak_ms2,
    accel_x_std_ms2,accel_y_std_ms2,accel_z_std_ms2,
    vibration_severity_ms2,
    gyro_x_mean_dps,gyro_y_mean_dps,gyro_z_mean_dps,
    gyro_x_rms_dps,gyro_y_rms_dps,gyro_z_rms_dps,
    gyro_x_peak_dps,gyro_y_peak_dps,gyro_z_peak_dps

How this was built
-------------------
Every numeric field below is sampled from a per-scenario Gaussian
(mean, std) fitted DIRECTLY from your uploaded file, grouped by
(fault_type, health_state) -- 8 scenarios in total. This keeps every stage
statistically identical to your real log, including the calibrated MPU
convention validated earlier against your MPU-6500 capture (accel bias ~0
on X/Y, gravity isolated on Z ~9.81 m/s^2; gyro bias ~0 dps).

Scenario structure, reverse-engineered from your file:

    HEALTHY baseline (variable length, load_factor cycles through
    {0.65, 0.70, 0.72, 0.75, 0.90})
        |
        v (alternates)
    CHAIN 1 -- mechanical fault progression:
        imbalance/degraded      (15 rows, load=0.85)
        bearing_wear/warning    ( 5 rows, load=0.85)
        bearing_wear/critical   (13 rows, load=0.80)
        phase_loss/fault        ( 5 rows, load=0.10)
        |
        v
    HEALTHY baseline
        |
        v
    CHAIN 2 -- electrical/thermal fault progression:
        none/degraded           (10 rows, load=0.80)   <- elevated load, no named fault yet
        overload/warning        ( 8 rows, load=0.80)
        insulation_failure/critical (6 rows, load=0.75)
        phase_loss/fault        ( 5 rows, load=0.00)
        |
        v
    (repeats, alternating chain1 / chain2)

health_label mapping (measured from your file): healthy=0, degraded=1,
warning=2, critical=3, fault=4.
fault_code mapping (measured from your file): none=0, overload=1,
bearing_wear=3, imbalance=4, phase_loss=5, insulation_failure=6.
Code 2 is reserved in the taxonomy (present in numbering, absent from your
sample) -- available here as an optional extra "voltage_sag" episode via
--include-voltage-sag if you want to exercise that class too.

Logging gaps: your file has ~19 non-5-second timestamp jumps (a few minutes
to a ~13-hour overnight pause), always landing inside a healthy or steady
fault stretch (i.e. the logger paused, the machine kept running in the same
state). Reproduced here via --gap-probability / --long-gap-probability.

Usage
-----
    # Regenerate ~the same size/shape as your file (4556 rows -> ~6.3 hours)
    python plant_sensor_simulator_v3.py --duration 22800 --fast --seed 42

    # Longer run, reproducible
    python plant_sensor_simulator_v3.py --duration 86400 --fast --seed 7
"""

import argparse
import csv
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


# --------------------------------------------------------------------------- #
# Health / fault taxonomy (measured directly from your uploaded file)
# --------------------------------------------------------------------------- #

HEALTH_LABEL = {"healthy": 0, "degraded": 1, "warning": 2, "critical": 3, "fault": 4}
FAULT_CODE = {
    "none": 0,
    "overload": 1,
    "voltage_sag": 2,       # reserved in taxonomy; not present in your sample, optional here
    "bearing_wear": 3,
    "imbalance": 4,
    "phase_loss": 5,
    "insulation_failure": 6,
}

CSV_FIELDS = [
    "timestamp", "load_factor", "health_label", "health_state", "fault_type", "fault_code",
    "motor_speed_rpm", "motor_current_a", "motor_voltage_v", "motor_power_kw",
    "motor_vibration_mm_s",
    "temp_ambient_c", "temp_surface_c", "temp_bearing_c",
    "accel_x_mean_ms2", "accel_y_mean_ms2", "accel_z_mean_ms2",
    "accel_x_rms_ms2", "accel_y_rms_ms2", "accel_z_rms_ms2",
    "accel_x_peak_ms2", "accel_y_peak_ms2", "accel_z_peak_ms2",
    "accel_x_std_ms2", "accel_y_std_ms2", "accel_z_std_ms2",
    "vibration_severity_ms2",
    "gyro_x_mean_dps", "gyro_y_mean_dps", "gyro_z_mean_dps",
    "gyro_x_rms_dps", "gyro_y_rms_dps", "gyro_z_rms_dps",
    "gyro_x_peak_dps", "gyro_y_peak_dps", "gyro_z_peak_dps",
]

NUMERIC_FIELDS = [c for c in CSV_FIELDS if c not in
                  ("timestamp", "load_factor", "health_label", "health_state", "fault_type", "fault_code")]

# Physical lower bounds so Gaussian sampling never produces nonsense (e.g. negative current)
NONNEGATIVE = {
    "motor_current_a", "motor_power_kw", "motor_vibration_mm_s",
    "accel_x_rms_ms2", "accel_y_rms_ms2", "accel_z_rms_ms2",
    "accel_x_peak_ms2", "accel_y_peak_ms2", "accel_z_peak_ms2",
    "accel_x_std_ms2", "accel_y_std_ms2", "accel_z_std_ms2",
    "vibration_severity_ms2",
    "gyro_x_rms_dps", "gyro_y_rms_dps", "gyro_z_rms_dps",
    "gyro_x_peak_dps", "gyro_y_peak_dps", "gyro_z_peak_dps",
    "motor_speed_rpm", "motor_voltage_v", "temp_ambient_c", "temp_surface_c", "temp_bearing_c",
}

PROFILES = {
    ("none", "healthy"): {
        "motor_speed_rpm": (1467.43, 3.76734),
        "motor_current_a": (8.07029, 0.502471),
        "motor_voltage_v": (400.024, 2.06335),
        "motor_power_kw": (4.75284, 0.296443),
        "motor_vibration_mm_s": (1.77966, 0.0755844),
        "temp_ambient_c": (35.007, 0.298914),
        "temp_surface_c": (88.3597, 3.80393),
        "temp_bearing_c": (93.5217, 4.35107),
        "accel_x_mean_ms2": (1.00583e-05, 0.000695369),
        "accel_y_mean_ms2": (1.79417e-05, 0.000671289),
        "accel_z_mean_ms2": (9.80999, 0.000314955),
        "accel_x_rms_ms2": (0.113966, 0.00748491),
        "accel_y_rms_ms2": (0.113945, 0.00746889),
        "accel_z_rms_ms2": (9.81006, 0.000315481),
        "accel_x_peak_ms2": (0.212481, 0.013091),
        "accel_y_peak_ms2": (0.212075, 0.0130087),
        "accel_z_peak_ms2": (9.88276, 0.00510497),
        "accel_x_std_ms2": (0.113964, 0.00748491),
        "accel_y_std_ms2": (0.113943, 0.00746913),
        "accel_z_std_ms2": (0.0351062, 0.00221267),
        "vibration_severity_ms2": (9.81139, 0.000363539),
        "gyro_x_mean_dps": (4.36893e-05, 0.00156194),
        "gyro_y_mean_dps": (-5.43689e-06, 0.001593),
        "gyro_z_mean_dps": (1.22718e-05, 0.000635998),
        "gyro_x_rms_dps": (0.086642, 0.0014381),
        "gyro_y_rms_dps": (0.0866228, 0.00143327),
        "gyro_z_rms_dps": (0.0244748, 0.000514632),
        "gyro_x_peak_dps": (0.244735, 0.0184586),
        "gyro_y_peak_dps": (0.243684, 0.0178883),
        "gyro_z_peak_dps": (0.0799358, 0.00696619),
    },
    ("imbalance", "degraded"): {
        "motor_speed_rpm": (1461.76, 1.88506),
        "motor_current_a": (8.94914, 0.0495652),
        "motor_voltage_v": (400.042, 1.95836),
        "motor_power_kw": (5.27071, 0.0396229),
        "motor_vibration_mm_s": (1.87689, 0.0520808),
        "temp_ambient_c": (35.0202, 0.295507),
        "temp_surface_c": (90.8279, 3.9168),
        "temp_bearing_c": (95.2952, 4.71924),
        "accel_x_mean_ms2": (-0.000401778, 0.00697273),
        "accel_y_mean_ms2": (0.000229111, 0.00657989),
        "accel_z_mean_ms2": (9.81002, 0.000325522),
        "accel_x_rms_ms2": (0.844699, 0.00733927),
        "accel_y_rms_ms2": (0.844831, 0.00781735),
        "accel_z_rms_ms2": (9.8101, 0.000324377),
        "accel_x_peak_ms2": (1.6545, 0.0173208),
        "accel_y_peak_ms2": (1.65438, 0.0171402),
        "accel_z_peak_ms2": (9.88854, 0.00431092),
        "accel_x_std_ms2": (0.844669, 0.00734099),
        "accel_y_std_ms2": (0.844805, 0.00781837),
        "accel_z_std_ms2": (0.038952, 0.000304278),
        "vibration_severity_ms2": (9.88258, 0.00110864),
        "gyro_x_mean_dps": (3.97778e-05, 0.00162379),
        "gyro_y_mean_dps": (7.33333e-05, 0.00153516),
        "gyro_z_mean_dps": (-1.04444e-05, 0.000631297),
        "gyro_x_rms_dps": (0.0866753, 0.001497),
        "gyro_y_rms_dps": (0.0866247, 0.00143148),
        "gyro_z_rms_dps": (0.0244531, 0.000519103),
        "gyro_x_peak_dps": (0.245982, 0.0188174),
        "gyro_y_peak_dps": (0.246033, 0.0194054),
        "gyro_z_peak_dps": (0.080214, 0.00701952),
    },
    ("bearing_wear", "warning"): {
        "motor_speed_rpm": (1461.78, 2.03269),
        "motor_current_a": (8.95129, 0.0475141),
        "motor_voltage_v": (400.214, 2.11207),
        "motor_power_kw": (5.27418, 0.038784),
        "motor_vibration_mm_s": (1.88041, 0.0508668),
        "temp_ambient_c": (34.9624, 0.300004),
        "temp_surface_c": (91.0087, 3.50595),
        "temp_bearing_c": (112.28, 6.54736),
        "accel_x_mean_ms2": (0.000791667, 0.00681675),
        "accel_y_mean_ms2": (2.23333e-05, 0.00657041),
        "accel_z_mean_ms2": (9.81, 0.000306267),
        "accel_x_rms_ms2": (0.865386, 0.00794914),
        "accel_y_rms_ms2": (0.865875, 0.0079936),
        "accel_z_rms_ms2": (9.81008, 0.000305894),
        "accel_x_peak_ms2": (1.91768, 0.0496077),
        "accel_y_peak_ms2": (1.92426, 0.0551596),
        "accel_z_peak_ms2": (9.88836, 0.00423904),
        "accel_x_std_ms2": (0.865359, 0.00794714),
        "accel_y_std_ms2": (0.86585, 0.00799252),
        "accel_z_std_ms2": (0.0389647, 0.000287666),
        "vibration_severity_ms2": (9.88617, 0.00121689),
        "gyro_x_mean_dps": (-7.3e-05, 0.00167044),
        "gyro_y_mean_dps": (-4.43333e-05, 0.00160154),
        "gyro_z_mean_dps": (-2.8e-05, 0.000617211),
        "gyro_x_rms_dps": (0.086688, 0.00148207),
        "gyro_y_rms_dps": (0.0865153, 0.00146186),
        "gyro_z_rms_dps": (0.0244413, 0.000535871),
        "gyro_x_peak_dps": (0.242037, 0.0176701),
        "gyro_y_peak_dps": (0.244826, 0.0193661),
        "gyro_z_peak_dps": (0.0798957, 0.00704469),
    },
    ("bearing_wear", "critical"): {
        "motor_speed_rpm": (1463.99, 1.81595),
        "motor_current_a": (10.7386, 0.504216),
        "motor_voltage_v": (400.113, 2.01826),
        "motor_power_kw": (6.32568, 0.298214),
        "motor_vibration_mm_s": (2.86065, 0.292451),
        "temp_ambient_c": (34.9811, 0.289299),
        "temp_surface_c": (90.166, 3.19442),
        "temp_bearing_c": (111.007, 6.04913),
        "accel_x_mean_ms2": (-0.00033625, 0.00633607),
        "accel_y_mean_ms2": (0.00061625, 0.00703936),
        "accel_z_mean_ms2": (9.81, 0.000312915),
        "accel_x_rms_ms2": (0.860631, 0.00730164),
        "accel_y_rms_ms2": (0.861143, 0.00806083),
        "accel_z_rms_ms2": (9.81007, 0.000311973),
        "accel_x_peak_ms2": (1.9112, 0.0461549),
        "accel_y_peak_ms2": (1.9191, 0.054339),
        "accel_z_peak_ms2": (9.88617, 0.00407338),
        "accel_x_std_ms2": (0.860606, 0.00730553),
        "accel_y_std_ms2": (0.861115, 0.00806237),
        "accel_z_std_ms2": (0.0374354, 0.000303514),
        "vibration_severity_ms2": (9.88533, 0.00109662),
        "gyro_x_mean_dps": (3e-05, 0.0015665),
        "gyro_y_mean_dps": (-0.000250417, 0.00146742),
        "gyro_z_mean_dps": (-2.08333e-06, 0.000684559),
        "gyro_x_rms_dps": (0.0865875, 0.0015433),
        "gyro_y_rms_dps": (0.0863658, 0.00149967),
        "gyro_z_rms_dps": (0.0244796, 0.000484209),
        "gyro_x_peak_dps": (0.24442, 0.0191701),
        "gyro_y_peak_dps": (0.243102, 0.0173504),
        "gyro_z_peak_dps": (0.0802958, 0.00689642),
    },
    ("phase_loss", "fault"): {
        "motor_speed_rpm": (1497.68, 2.89026),
        "motor_current_a": (5.3741, 0.714683),
        "motor_voltage_v": (270.07, 17.5886),
        "motor_power_kw": (2.13692, 0.319975),
        "motor_vibration_mm_s": (1.24206, 0.0642104),
        "temp_ambient_c": (35.018, 0.290968),
        "temp_surface_c": (74.7541, 2.81756),
        "temp_bearing_c": (83.3091, 3.22962),
        "accel_x_mean_ms2": (-8.47458e-05, 0.000607416),
        "accel_y_mean_ms2": (2.13559e-05, 0.000587195),
        "accel_z_mean_ms2": (9.81004, 0.000294741),
        "accel_x_rms_ms2": (0.0454647, 0.00477457),
        "accel_y_rms_ms2": (0.0454403, 0.00477058),
        "accel_z_rms_ms2": (9.81005, 0.00029571),
        "accel_x_peak_ms2": (0.115289, 0.0104173),
        "accel_y_peak_ms2": (0.113881, 0.0101844),
        "accel_z_peak_ms2": (9.85489, 0.00418959),
        "accel_x_std_ms2": (0.0454586, 0.00477676),
        "accel_y_std_ms2": (0.0454373, 0.00477018),
        "accel_z_std_ms2": (0.0158098, 0.00124571),
        "vibration_severity_ms2": (9.81026, 0.000295156),
        "gyro_x_mean_dps": (-2.37288e-05, 0.00154248),
        "gyro_y_mean_dps": (-0.000213559, 0.00156758),
        "gyro_z_mean_dps": (-2.33898e-05, 0.00064453),
        "gyro_x_rms_dps": (0.0864451, 0.00133981),
        "gyro_y_rms_dps": (0.0865644, 0.0013292),
        "gyro_z_rms_dps": (0.0245058, 0.000553204),
        "gyro_x_peak_dps": (0.244383, 0.0176292),
        "gyro_y_peak_dps": (0.245061, 0.019082),
        "gyro_z_peak_dps": (0.0799742, 0.00689806),
    },
    ("none", "degraded"): {
        "motor_speed_rpm": (1464.13, 1.94044),
        "motor_current_a": (8.60004, 0.0477361),
        "motor_voltage_v": (399.993, 1.94432),
        "motor_power_kw": (5.06449, 0.0361295),
        "motor_vibration_mm_s": (1.83736, 0.0516832),
        "temp_ambient_c": (35.0142, 0.286717),
        "temp_surface_c": (105.149, 3.79308),
        "temp_bearing_c": (95.1613, 3.09134),
        "accel_x_mean_ms2": (-5.62069e-05, 0.000603306),
        "accel_y_mean_ms2": (4.82759e-06, 0.000615471),
        "accel_z_mean_ms2": (9.80997, 0.000302074),
        "accel_x_rms_ms2": (0.121822, 0.000618767),
        "accel_y_rms_ms2": (0.121807, 0.000654625),
        "accel_z_rms_ms2": (9.81004, 0.000300676),
        "accel_x_peak_ms2": (0.22334, 0.00847737),
        "accel_y_peak_ms2": (0.223694, 0.00789947),
        "accel_z_peak_ms2": (9.88648, 0.00406017),
        "accel_x_std_ms2": (0.12182, 0.000619315),
        "accel_y_std_ms2": (0.121806, 0.000653924),
        "accel_z_std_ms2": (0.0374003, 0.000288814),
        "vibration_severity_ms2": (9.81155, 0.000301746),
        "gyro_x_mean_dps": (0.00018, 0.00167964),
        "gyro_y_mean_dps": (-2.2069e-05, 0.00166022),
        "gyro_z_mean_dps": (8.62069e-05, 0.000605506),
        "gyro_x_rms_dps": (0.0865934, 0.00138043),
        "gyro_y_rms_dps": (0.086611, 0.00141554),
        "gyro_z_rms_dps": (0.0245214, 0.000509183),
        "gyro_x_peak_dps": (0.24553, 0.0181094),
        "gyro_y_peak_dps": (0.244182, 0.0188282),
        "gyro_z_peak_dps": (0.0802493, 0.00772685),
    },
    ("overload", "warning"): {
        "motor_speed_rpm": (1464.06, 1.95262),
        "motor_current_a": (10.7509, 0.494529),
        "motor_voltage_v": (400.083, 1.92125),
        "motor_power_kw": (6.33249, 0.292195),
        "motor_vibration_mm_s": (2.82634, 0.28831),
        "temp_ambient_c": (35.0267, 0.285497),
        "temp_surface_c": (105.376, 3.66377),
        "temp_bearing_c": (95.2416, 2.71401),
        "accel_x_mean_ms2": (-2.88793e-05, 0.000612782),
        "accel_y_mean_ms2": (2.4569e-05, 0.000660818),
        "accel_z_mean_ms2": (9.81002, 0.000309267),
        "accel_x_rms_ms2": (0.121863, 0.000637686),
        "accel_y_rms_ms2": (0.121878, 0.000669656),
        "accel_z_rms_ms2": (9.8101, 0.000308191),
        "accel_x_peak_ms2": (0.223391, 0.00818944),
        "accel_y_peak_ms2": (0.22298, 0.0077669),
        "accel_z_peak_ms2": (9.88648, 0.00424032),
        "accel_x_std_ms2": (0.121862, 0.000637762),
        "accel_y_std_ms2": (0.121877, 0.000669387),
        "accel_z_std_ms2": (0.0374246, 0.000299784),
        "vibration_severity_ms2": (9.81161, 0.000309745),
        "gyro_x_mean_dps": (-5.17241e-06, 0.00153414),
        "gyro_y_mean_dps": (1.07759e-05, 0.00167551),
        "gyro_z_mean_dps": (9.91379e-06, 0.000648364),
        "gyro_x_rms_dps": (0.0865491, 0.00138147),
        "gyro_y_rms_dps": (0.0864853, 0.00151898),
        "gyro_z_rms_dps": (0.0245457, 0.000518518),
        "gyro_x_peak_dps": (0.244884, 0.0183615),
        "gyro_y_peak_dps": (0.244319, 0.0214146),
        "gyro_z_peak_dps": (0.0806358, 0.00757248),
    },
    ("insulation_failure", "critical"): {
        "motor_speed_rpm": (1466.32, 1.98416),
        "motor_current_a": (10.3173, 0.473855),
        "motor_voltage_v": (399.972, 1.81991),
        "motor_power_kw": (6.07553, 0.282157),
        "motor_vibration_mm_s": (2.80912, 0.273292),
        "temp_ambient_c": (34.9714, 0.275361),
        "temp_surface_c": (104.252, 3.64935),
        "temp_bearing_c": (94.5729, 2.57098),
        "accel_x_mean_ms2": (1.14943e-06, 0.000691989),
        "accel_y_mean_ms2": (-7.75862e-05, 0.000585861),
        "accel_z_mean_ms2": (9.80999, 0.000319456),
        "accel_x_rms_ms2": (0.116616, 0.000652295),
        "accel_y_rms_ms2": (0.116611, 0.000646818),
        "accel_z_rms_ms2": (9.81006, 0.000322217),
        "accel_x_peak_ms2": (0.216007, 0.00783505),
        "accel_y_peak_ms2": (0.216387, 0.00773296),
        "accel_z_peak_ms2": (9.88442, 0.0042411),
        "accel_x_std_ms2": (0.116613, 0.000652106),
        "accel_y_std_ms2": (0.11661, 0.000645112),
        "accel_z_std_ms2": (0.0359149, 0.000327663),
        "vibration_severity_ms2": (9.81145, 0.000324049),
        "gyro_x_mean_dps": (0.000112069, 0.00168715),
        "gyro_y_mean_dps": (2.12644e-05, 0.00162331),
        "gyro_z_mean_dps": (-7.93103e-05, 0.000588811),
        "gyro_x_rms_dps": (0.086642, 0.00140946),
        "gyro_y_rms_dps": (0.0865092, 0.00135982),
        "gyro_z_rms_dps": (0.0245615, 0.00053986),
        "gyro_x_peak_dps": (0.245589, 0.0186233),
        "gyro_y_peak_dps": (0.242248, 0.0191258),
        "gyro_z_peak_dps": (0.0807282, 0.00713257),
    },
}
# Optional extra scenario (fault_code=2), NOT present in your uploaded sample --
# interpolated between phase_loss (voltage collapse) and healthy (mild version),
# only used if --include-voltage-sag is passed.
PROFILES[("voltage_sag", "warning")] = {
    "motor_speed_rpm": (1472.0, 2.5),
    "motor_current_a": (9.4, 0.3),
    "motor_voltage_v": (350.0, 6.0),
    "motor_power_kw": (5.3, 0.15),
    "motor_vibration_mm_s": (1.95, 0.08),
    "temp_ambient_c": (35.0, 0.3),
    "temp_surface_c": (78.0, 3.0),
    "temp_bearing_c": (86.0, 3.0),
    "accel_x_mean_ms2": (0.0, 0.0007),
    "accel_y_mean_ms2": (0.0, 0.0007),
    "accel_z_mean_ms2": (9.81, 0.0003),
    "accel_x_rms_ms2": (0.13, 0.001),
    "accel_y_rms_ms2": (0.13, 0.001),
    "accel_z_rms_ms2": (9.81, 0.0003),
    "accel_x_peak_ms2": (0.24, 0.01),
    "accel_y_peak_ms2": (0.24, 0.01),
    "accel_z_peak_ms2": (9.886, 0.004),
    "accel_x_std_ms2": (0.13, 0.001),
    "accel_y_std_ms2": (0.13, 0.001),
    "accel_z_std_ms2": (0.038, 0.0003),
    "vibration_severity_ms2": (9.812, 0.0003),
    "gyro_x_mean_dps": (0.0, 0.0016),
    "gyro_y_mean_dps": (0.0, 0.0016),
    "gyro_z_mean_dps": (0.0, 0.0006),
    "gyro_x_rms_dps": (0.087, 0.0014),
    "gyro_y_rms_dps": (0.087, 0.0014),
    "gyro_z_rms_dps": (0.0245, 0.0005),
    "gyro_x_peak_dps": (0.245, 0.018),
    "gyro_y_peak_dps": (0.245, 0.018),
    "gyro_z_peak_dps": (0.08, 0.007),
}


# --------------------------------------------------------------------------- #
# Scenario scheduler
# --------------------------------------------------------------------------- #

# (fault_type, health_state, duration_rows, load_factor) -- durations and loads
# taken directly from your file's episode lengths.
CHAIN_1 = [  # mechanical fault progression
    ("imbalance", "degraded", 15, 0.85),
    ("bearing_wear", "warning", 5, 0.85),
    ("bearing_wear", "critical", 13, 0.80),
    ("phase_loss", "fault", 5, 0.10),
]
CHAIN_2 = [  # electrical / thermal fault progression
    ("none", "degraded", 10, 0.80),
    ("overload", "warning", 8, 0.80),
    ("insulation_failure", "critical", 6, 0.75),
    ("phase_loss", "fault", 5, 0.00),
]
CHAIN_VOLTAGE_SAG = [  # optional extra, fault_code=2, not in your original sample
    ("voltage_sag", "warning", 10, 0.80),
    ("phase_loss", "fault", 5, 0.00),
]

HEALTHY_LOAD_LEVELS = [0.65, 0.70, 0.72, 0.75, 0.90]


@dataclass
class SimConfig:
    duration_seconds: int = 22800          # ~6.3h, roughly matching your file's row count at 5s
    interval_seconds: float = 5.0
    output_file: str = "/mnt/user-data/outputs/plant_sensor_data_v3.csv"
    fast: bool = True
    seed: int | None = None
    healthy_min_rows: int = 30
    healthy_max_rows: int = 60
    gap_probability: float = 0.01          # short pause (minutes) chance per healthy row
    long_gap_probability: float = 0.0006   # rare overnight-scale pause chance per healthy row
    include_voltage_sag: bool = False      # add the optional fault_code=2 chain to rotation


def sample_row_values(fault_type: str, health_state: str) -> dict:
    profile = PROFILES[(fault_type, health_state)]
    out = {}
    for field in NUMERIC_FIELDS:
        mean, std = profile[field]
        val = random.gauss(mean, std)
        if field in NONNEGATIVE:
            val = max(0.0, val)
        out[field] = val
    return out


def round_row(values: dict) -> dict:
    rounded = {}
    for k, v in values.items():
        if k.startswith("accel_") or k.startswith("gyro_") or k == "vibration_severity_ms2":
            rounded[k] = round(v, 4)
        elif k == "motor_vibration_mm_s":
            rounded[k] = round(v, 3)
        else:
            rounded[k] = round(v, 2)
    return rounded


class PlantSimulator:
    def __init__(self, config: SimConfig):
        self.config = config
        if config.seed is not None:
            random.seed(config.seed)
        self.chains = [CHAIN_1, CHAIN_2] + ([CHAIN_VOLTAGE_SAG] if config.include_voltage_sag else [])
        self._chain_idx = 0

    def _next_chain(self):
        chain = self.chains[self._chain_idx % len(self.chains)]
        self._chain_idx += 1
        return chain

    def _episode_rows(self, fault_type, health_state, n_rows, load_factor):
        for _ in range(n_rows):
            values = sample_row_values(fault_type, health_state)
            yield fault_type, health_state, load_factor, values

    def _healthy_rows(self):
        n_rows = random.randint(self.config.healthy_min_rows, self.config.healthy_max_rows)
        load_factor = random.choice(HEALTHY_LOAD_LEVELS)
        for _ in range(n_rows):
            values = sample_row_values("none", "healthy")
            yield "none", "healthy", load_factor, values
            # small chance of a mid-segment load-level change, matching the
            # partial-segment load shifts observed in the source file
            if random.random() < 0.03:
                load_factor = random.choice(HEALTHY_LOAD_LEVELS)

    def _all_episodes(self):
        """Infinite generator alternating healthy periods with fault chains."""
        while True:
            yield from self._healthy_rows()
            chain = self._next_chain()
            for fault_type, health_state, n_rows, load_factor in chain:
                yield from self._episode_rows(fault_type, health_state, n_rows, load_factor)

    def run(self):
        out_dir = os.path.dirname(self.config.output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        n_target_rows = int(self.config.duration_seconds / self.config.interval_seconds)
        start_time = datetime.now(timezone.utc)
        current_time = start_time
        row_count = 0

        with open(self.config.output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()

            for fault_type, health_state, load_factor, values in self._all_episodes():
                if row_count >= n_target_rows:
                    break

                # occasional logging-gap timestamp jump, only during healthy stretches
                # (matches the gap placement observed in your uploaded file)
                if health_state == "healthy":
                    r = random.random()
                    if r < self.config.long_gap_probability:
                        current_time += timedelta(seconds=random.uniform(3600, 46800))
                    elif r < self.config.long_gap_probability + self.config.gap_probability:
                        current_time += timedelta(seconds=random.uniform(60, 1100))

                rounded = round_row(values)
                row = {
                    "timestamp": current_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "load_factor": load_factor,
                    "health_label": HEALTH_LABEL[health_state],
                    "health_state": health_state,
                    "fault_type": fault_type,
                    "fault_code": FAULT_CODE[fault_type],
                    **rounded,
                }
                writer.writerow(row)
                row_count += 1

                if row_count % max(1, (n_target_rows // 25 or 1)) == 0 or row_count == n_target_rows:
                    print(
                        f"[{row['timestamp']}] row={row_count:>6d}  load={load_factor:.2f}  "
                        f"RPM={row['motor_speed_rpm']:>7.1f}  I={row['motor_current_a']:>5.2f}A  "
                        f"Bearing={row['temp_bearing_c']:>6.1f}C  Vib={row['motor_vibration_mm_s']:>5.2f}mm/s  "
                        f"Fault={fault_type:<18} Health={health_state}"
                    )

                current_time += timedelta(seconds=self.config.interval_seconds)
                if not self.config.fast:
                    time.sleep(self.config.interval_seconds)

        print(f"\nDone. {row_count} rows written to '{self.config.output_file}'.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> SimConfig:
    p = argparse.ArgumentParser(description="Reproduce the scripted fault-injection PdM log format.")
    p.add_argument("--duration", type=int, default=22800, help="Total simulated duration in seconds")
    p.add_argument("--interval", type=float, default=5.0, help="Logging interval in seconds (default: 5)")
    p.add_argument("--output", type=str, default="/mnt/user-data/outputs/plant_sensor_data_v3.csv")
    p.add_argument("--fast", action="store_true", help="Run as fast as possible instead of real-time sleeping")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--healthy-min-rows", type=int, default=30)
    p.add_argument("--healthy-max-rows", type=int, default=60)
    p.add_argument("--gap-probability", type=float, default=0.01,
                   help="Per-healthy-row chance of a short (1-18min) logging gap")
    p.add_argument("--long-gap-probability", type=float, default=0.0006,
                   help="Per-healthy-row chance of a long (1-13hr) logging gap")
    p.add_argument("--include-voltage-sag", action="store_true",
                   help="Add the optional fault_code=2 'voltage_sag' chain into rotation "
                        "(not present in your original sample, included for taxonomy completeness)")
    args = p.parse_args()

    return SimConfig(
        duration_seconds=args.duration,
        interval_seconds=args.interval,
        output_file=args.output,
        fast=args.fast,
        seed=args.seed,
        healthy_min_rows=args.healthy_min_rows,
        healthy_max_rows=args.healthy_max_rows,
        gap_probability=args.gap_probability,
        long_gap_probability=args.long_gap_probability,
        include_voltage_sag=args.include_voltage_sag,
    )


if __name__ == "__main__":
    config = parse_args()
    sim = PlantSimulator(config)
    sim.run()