# """
# =============================================================================
#   Predictive Maintenance - Sensor Data Plant Simulator
# =============================================================================
#   Simulates real-time sensor data from:
#     - Motor Sensors     : current, voltage, power, speed (RPM), vibration
#     - Temperature Sensor: ambient temp, motor surface temp, bearing temp
#     - MPU (IMU)         : accelerometer (X/Y/Z), gyroscope (X/Y/Z)

#   Logging Rate : Every 5 seconds
#   Output       : CSV file + optional live console print
#   AI-Ready     : Includes fault injection & health labels for model training
# =============================================================================
# """

# import time
# import math
# import random
# import csv
# import os
# import json
# import argparse
# from datetime import datetime
# from enum import Enum


# # ---------------------------------------------------------------------------
# # Configuration
# # ---------------------------------------------------------------------------

# LOGGING_RATE_SECONDS = 5          # Logging interval (seconds)
# OUTPUT_CSV = "sensor_data.csv"    # Output file
# MAX_LOG_LINES = None              # None = run forever; set integer to cap rows

# # Motor nameplate parameters
# MOTOR_RATED_SPEED_RPM   = 1500
# MOTOR_RATED_CURRENT_A   = 10.0
# MOTOR_RATED_VOLTAGE_V   = 400.0
# MOTOR_RATED_POWER_KW    = 5.5

# # Temperature baselines (°C)
# TEMP_AMBIENT_BASE   = 35.0   # Typical ambient in a plant room
# TEMP_SURFACE_BASE   = 55.0   # Motor surface at normal load
# TEMP_BEARING_BASE   = 60.0   # Bearing temperature at normal load

# # MPU / IMU baselines
# ACCEL_IDLE_MS2  = 0.05   # m/s² baseline vibration at healthy state
# GYRO_IDLE_DPS   = 0.10   # °/s baseline gyro drift


# # ---------------------------------------------------------------------------
# # Machine Health State
# # ---------------------------------------------------------------------------

# class HealthState(Enum):
#     HEALTHY         = "healthy"
#     DEGRADED        = "degraded"        # Early wear / slight imbalance
#     WARNING         = "warning"         # Bearing wear, rising temp
#     CRITICAL        = "critical"        # Imminent failure
#     FAULT           = "fault"           # Active fault / stopped


# # Numeric label for ML training
# HEALTH_LABEL = {
#     HealthState.HEALTHY  : 0,
#     HealthState.DEGRADED : 1,
#     HealthState.WARNING  : 2,
#     HealthState.CRITICAL : 3,
#     HealthState.FAULT    : 4,
# }

# # Fault type codes
# FAULT_TYPES = {
#     "none"              : 0,
#     "overload"          : 1,
#     "overheat"          : 2,
#     "bearing_wear"      : 3,
#     "imbalance"         : 4,
#     "phase_loss"        : 5,
#     "insulation_failure": 6,
# }


# # ---------------------------------------------------------------------------
# # Sensor Models
# # ---------------------------------------------------------------------------

# class MotorSensor:
#     """Simulates 3-phase induction motor electrical & mechanical sensors."""

#     def __init__(self):
#         self.load_factor   = 0.75     # 0–1 (fraction of rated load)
#         self.phase_loss    = False
#         self.overload      = False

#     def set_operating_condition(self, load_factor: float,
#                                 phase_loss: bool = False,
#                                 overload: bool = False):
#         self.load_factor = max(0.0, min(1.0, load_factor))
#         self.phase_loss  = phase_loss
#         self.overload    = overload

#     def read(self) -> dict:
#         lf = self.load_factor

#         # Speed (RPM) — slight slip under load
#         slip = 0.03 * lf
#         speed = MOTOR_RATED_SPEED_RPM * (1 - slip)
#         speed += random.gauss(0, 2)

#         # Current (A) — proportional to load + noise
#         current = MOTOR_RATED_CURRENT_A * (0.3 + 0.7 * lf)
#         if self.overload:
#             current *= random.uniform(1.15, 1.35)
#         if self.phase_loss:
#             current *= random.uniform(1.4, 1.8)   # unbalanced current surge
#         current += random.gauss(0, 0.05)

#         # Voltage (V)
#         voltage = MOTOR_RATED_VOLTAGE_V + random.gauss(0, 2)
#         if self.phase_loss:
#             voltage = MOTOR_RATED_VOLTAGE_V * random.uniform(0.6, 0.75)

#         # Power (kW)
#         power_factor = 0.85
#         power = (math.sqrt(3) * voltage * current * power_factor) / 1000.0
#         power = max(0, power)

#         # Vibration (mm/s RMS) — healthy: < 2.8 mm/s (ISO 10816 Class I)
#         vibration_rms = 1.2 + 0.8 * lf + random.gauss(0, 0.05)
#         if self.overload:
#             vibration_rms += random.uniform(0.5, 1.5)

#         return {
#             "motor_speed_rpm"   : round(speed, 2),
#             "motor_current_a"   : round(current, 3),
#             "motor_voltage_v"   : round(voltage, 2),
#             "motor_power_kw"    : round(power, 3),
#             "motor_vibration_mm_s": round(vibration_rms, 3),
#         }


# class TemperatureSensor:
#     """Simulates NTC/PT100 temperature probes at ambient, motor, and bearing."""

#     def __init__(self):
#         self.load_factor     = 0.75
#         self.bearing_wear    = False
#         self.insulation_fail = False
#         self._accumulated_heat = 0.0    # thermal mass model

#     def set_operating_condition(self, load_factor: float,
#                                 bearing_wear: bool = False,
#                                 insulation_fail: bool = False):
#         self.load_factor     = max(0.0, min(1.0, load_factor))
#         self.bearing_wear    = bearing_wear
#         self.insulation_fail = insulation_fail
#         # Gradually accumulate heat
#         delta = (load_factor - 0.5) * 0.8
#         self._accumulated_heat = max(0, self._accumulated_heat + delta)
#         self._accumulated_heat = min(self._accumulated_heat, 40)

#     def read(self) -> dict:
#         lf = self.load_factor

#         # Ambient — slow drift ± small noise
#         ambient = TEMP_AMBIENT_BASE + random.gauss(0, 0.3)

#         # Motor surface temperature
#         surface = (TEMP_SURFACE_BASE
#                    + 20 * lf
#                    + self._accumulated_heat * 0.5
#                    + random.gauss(0, 0.4))
#         if self.insulation_fail:
#             surface += random.uniform(10, 20)

#         # Bearing temperature
#         bearing = (TEMP_BEARING_BASE
#                    + 15 * lf
#                    + self._accumulated_heat * 0.6
#                    + random.gauss(0, 0.3))
#         if self.bearing_wear:
#             bearing += random.uniform(8, 25)   # friction-induced heating

#         return {
#             "temp_ambient_c"  : round(ambient, 2),
#             "temp_surface_c"  : round(surface, 2),
#             "temp_bearing_c"  : round(bearing, 2),
#         }


# class MPUSensor:
#     """
#     Simulates an MPU-6050 / MPU-9250 style IMU.
#     Provides 3-axis accelerometer (m/s²) and 3-axis gyroscope (°/s).
#     High-frequency vibration is down-sampled to 5 s statistical features.
#     """

#     def __init__(self):
#         self.load_factor    = 0.75
#         self.imbalance      = False
#         self.bearing_wear   = False
#         self._t             = 0.0      # internal time counter
#         # MPU internal sampling @ 200 Hz, we report 5 s aggregates
#         self.internal_fs    = 200      # Hz
#         self.window_samples = self.internal_fs * LOGGING_RATE_SECONDS

#     def set_operating_condition(self, load_factor: float,
#                                 imbalance: bool = False,
#                                 bearing_wear: bool = False):
#         self.load_factor  = max(0.0, min(1.0, load_factor))
#         self.imbalance    = imbalance
#         self.bearing_wear = bearing_wear

#     def _generate_window(self) -> tuple:
#         """Generate a 5-second window of raw IMU samples and return statistics."""
#         lf  = self.load_factor
#         n   = self.window_samples
#         dt  = 1.0 / self.internal_fs

#         ax_buf, ay_buf, az_buf = [], [], []
#         gx_buf, gy_buf, gz_buf = [], [], []

#         base_vib = ACCEL_IDLE_MS2 + 0.15 * lf   # fundamental vibration

#         for i in range(n):
#             t = self._t + i * dt

#             # Rotational frequency (Hz)
#             rot_freq = (MOTOR_RATED_SPEED_RPM * (1 - 0.03 * lf)) / 60.0

#             # Accelerometer — fundamental + harmonics
#             ax = base_vib * math.sin(2 * math.pi * rot_freq * t)
#             ay = base_vib * math.cos(2 * math.pi * rot_freq * t)
#             az = 9.81 + base_vib * 0.3 * math.sin(4 * math.pi * rot_freq * t)

#             if self.imbalance:
#                 imb = random.uniform(0.5, 1.5)
#                 ax += imb * math.sin(2 * math.pi * rot_freq * t + 0.5)
#                 ay += imb * math.cos(2 * math.pi * rot_freq * t + 0.5)

#             if self.bearing_wear:
#                 # Bearing defect frequency ≈ 3.5× shaft speed
#                 bdf = 3.5 * rot_freq
#                 b_amp = random.uniform(0.1, 0.4)
#                 ax += b_amp * math.sin(2 * math.pi * bdf * t)
#                 ay += b_amp * math.sin(2 * math.pi * bdf * t)

#             # Add Gaussian noise
#             ax += random.gauss(0, 0.02)
#             ay += random.gauss(0, 0.02)
#             az += random.gauss(0, 0.01)

#             # Gyroscope (°/s)
#             gx = GYRO_IDLE_DPS * math.sin(2 * math.pi * rot_freq * t) + random.gauss(0, 0.05)
#             gy = GYRO_IDLE_DPS * math.cos(2 * math.pi * rot_freq * t) + random.gauss(0, 0.05)
#             gz = GYRO_IDLE_DPS * 0.2 * math.sin(4 * math.pi * rot_freq * t) + random.gauss(0, 0.02)

#             ax_buf.append(ax); ay_buf.append(ay); az_buf.append(az)
#             gx_buf.append(gx); gy_buf.append(gy); gz_buf.append(gz)

#         self._t += n * dt

#         def stats(buf):
#             n    = len(buf)
#             mean = sum(buf) / n
#             rms  = math.sqrt(sum(x**2 for x in buf) / n)
#             peak = max(abs(x) for x in buf)
#             var  = sum((x - mean)**2 for x in buf) / n
#             std  = math.sqrt(var)
#             return mean, rms, peak, std

#         return ax_buf, ay_buf, az_buf, gx_buf, gy_buf, gz_buf

#     def read(self) -> dict:
#         ax_b, ay_b, az_b, gx_b, gy_b, gz_b = self._generate_window()

#         def rms(buf):
#             return math.sqrt(sum(x**2 for x in buf) / len(buf))

#         def peak(buf):
#             return max(abs(x) for x in buf)

#         def std(buf):
#             m = sum(buf) / len(buf)
#             return math.sqrt(sum((x - m)**2 for x in buf) / len(buf))

#         def mean(buf):
#             return sum(buf) / len(buf)

#         # RMS acceleration magnitude
#         accel_rms_x = rms(ax_b)
#         accel_rms_y = rms(ay_b)
#         accel_rms_z = rms(az_b)

#         # Overall vibration severity (vector magnitude RMS)
#         vib_severity = math.sqrt(accel_rms_x**2 + accel_rms_y**2 + accel_rms_z**2)

#         return {
#             # Accelerometer statistics (m/s²)
#             "accel_x_mean_ms2" : round(mean(ax_b), 4),
#             "accel_y_mean_ms2" : round(mean(ay_b), 4),
#             "accel_z_mean_ms2" : round(mean(az_b), 4),
#             "accel_x_rms_ms2"  : round(accel_rms_x, 4),
#             "accel_y_rms_ms2"  : round(accel_rms_y, 4),
#             "accel_z_rms_ms2"  : round(accel_rms_z, 4),
#             "accel_x_peak_ms2" : round(peak(ax_b), 4),
#             "accel_y_peak_ms2" : round(peak(ay_b), 4),
#             "accel_z_peak_ms2" : round(peak(az_b), 4),
#             "accel_x_std_ms2"  : round(std(ax_b), 4),
#             "accel_y_std_ms2"  : round(std(ay_b), 4),
#             "accel_z_std_ms2"  : round(std(az_b), 4),
#             "vibration_severity_ms2": round(vib_severity, 4),
#             # Gyroscope statistics (°/s)
#             "gyro_x_mean_dps"  : round(mean(gx_b), 4),
#             "gyro_y_mean_dps"  : round(mean(gy_b), 4),
#             "gyro_z_mean_dps"  : round(mean(gz_b), 4),
#             "gyro_x_rms_dps"   : round(rms(gx_b), 4),
#             "gyro_y_rms_dps"   : round(rms(gy_b), 4),
#             "gyro_z_rms_dps"   : round(rms(gz_b), 4),
#             "gyro_x_peak_dps"  : round(peak(gx_b), 4),
#             "gyro_y_peak_dps"  : round(peak(gy_b), 4),
#             "gyro_z_peak_dps"  : round(peak(gz_b), 4),
#         }


# # ---------------------------------------------------------------------------
# # Fault Injection Scheduler
# # ---------------------------------------------------------------------------

# class FaultScheduler:
#     """
#     Cycles through machine health states over time to create a
#     labelled dataset that mirrors realistic degradation patterns.
#     """

#     SCENARIO = [
#         # (duration_steps, health_state, load_factor, faults...)
#         (20,  HealthState.HEALTHY,  0.70, {}),
#         (10,  HealthState.HEALTHY,  0.90, {}),                        # high load
#         (15,  HealthState.DEGRADED, 0.85, {"imbalance": True}),       # imbalance starts
#         (10,  HealthState.WARNING,  0.85, {"imbalance": True,
#                                            "bearing_wear": True}),    # bearing wear
#         (8,   HealthState.CRITICAL, 0.80, {"imbalance": True,
#                                            "bearing_wear": True,
#                                            "overload": True}),
#         (5,   HealthState.FAULT,    0.10, {"phase_loss": True}),      # fault
#         (20,  HealthState.HEALTHY,  0.65, {}),                        # recovery
#         (12,  HealthState.HEALTHY,  0.75, {}),
#         (10,  HealthState.DEGRADED, 0.80, {"insulation_fail": True}), # insulation degrading
#         (8,   HealthState.WARNING,  0.80, {"insulation_fail": True,
#                                            "overload": True}),
#         (6,   HealthState.CRITICAL, 0.75, {"insulation_fail": True,
#                                            "overload": True,
#                                            "overheat": True}),
#         (5,   HealthState.FAULT,    0.0,  {"phase_loss": True}),
#         (25,  HealthState.HEALTHY,  0.72, {}),                        # restart healthy
#     ]

#     def __init__(self):
#         self._step  = 0
#         self._index = 0
#         self._count = 0
#         self._total_steps = sum(d for d, *_ in self.SCENARIO)

#     def get_state(self) -> tuple:
#         """Return (health_state, load_factor, fault_dict, fault_label)."""
#         if self._index >= len(self.SCENARIO):
#             self._index = 0   # cycle

#         duration, state, load, faults = self.SCENARIO[self._index]
#         self._count += 1
#         if self._count >= duration:
#             self._count  = 0
#             self._index += 1

#         # Build fault type string
#         if state == HealthState.FAULT:
#             if faults.get("phase_loss"):
#                 fault_label = "phase_loss"
#             else:
#                 fault_label = "fault"
#         elif faults.get("insulation_fail") and state == HealthState.CRITICAL:
#             fault_label = "insulation_failure"
#         elif faults.get("bearing_wear"):
#             fault_label = "bearing_wear"
#         elif faults.get("imbalance"):
#             fault_label = "imbalance"
#         elif faults.get("overload"):
#             fault_label = "overload"
#         elif faults.get("overheat"):
#             fault_label = "overheat"
#         else:
#             fault_label = "none"

#         return state, load, faults, fault_label


# # ---------------------------------------------------------------------------
# # Plant Simulator (main class)
# # ---------------------------------------------------------------------------

# class PlantSimulator:
#     """
#     Top-level simulator that orchestrates all sensors, applies fault
#     injections, and logs data at the configured rate.
#     """

#     def __init__(self, output_csv: str = OUTPUT_CSV,
#                  verbose: bool = True,
#                  inject_faults: bool = True,
#                  max_rows: int = None,
#                  json_output: bool = False):
#         self.output_csv    = output_csv
#         self.verbose       = verbose
#         self.inject_faults = inject_faults
#         self.max_rows      = max_rows
#         self.json_output   = json_output

#         self.motor   = MotorSensor()
#         self.temp    = TemperatureSensor()
#         self.mpu     = MPUSensor()
#         self.sched   = FaultScheduler()

#         self._row_count = 0

#     # ------------------------------------------------------------------
#     def _apply_faults(self, load: float, faults: dict):
#         self.motor.set_operating_condition(
#             load_factor = load,
#             phase_loss  = faults.get("phase_loss", False),
#             overload    = faults.get("overload",   False),
#         )
#         self.temp.set_operating_condition(
#             load_factor     = load,
#             bearing_wear    = faults.get("bearing_wear",    False),
#             insulation_fail = faults.get("insulation_fail", False),
#         )
#         self.mpu.set_operating_condition(
#             load_factor  = load,
#             imbalance    = faults.get("imbalance",    False),
#             bearing_wear = faults.get("bearing_wear", False),
#         )

#     # ------------------------------------------------------------------
#     def _collect_sample(self) -> dict:
#         """Read all sensors and return a merged flat dict."""
#         ts      = datetime.utcnow().isoformat(timespec="seconds") + "Z"
#         motor_d = self.motor.read()
#         temp_d  = self.temp.read()
#         mpu_d   = self.mpu.read()

#         sample = {"timestamp": ts}
#         sample.update(motor_d)
#         sample.update(temp_d)
#         sample.update(mpu_d)
#         return sample

#     # ------------------------------------------------------------------
#     def _add_labels(self, sample: dict,
#                     state: HealthState,
#                     fault_label: str,
#                     load: float) -> dict:
#         sample["load_factor"]   = round(load, 2)
#         sample["health_label"]  = HEALTH_LABEL[state]
#         sample["health_state"]  = state.value
#         sample["fault_type"]    = fault_label
#         sample["fault_code"]    = FAULT_TYPES.get(fault_label, -1)
#         return sample

#     # ------------------------------------------------------------------
#     def _get_csv_fieldnames(self) -> list:
#         dummy_m = self.motor.read()
#         dummy_t = self.temp.read()
#         dummy_i = self.mpu.read()
#         base = {"timestamp": "", "load_factor": 0,
#                 "health_label": 0, "health_state": "", "fault_type": "",
#                 "fault_code": 0}
#         base.update(dummy_m)
#         base.update(dummy_t)
#         base.update(dummy_i)
#         return list(base.keys())

#     # ------------------------------------------------------------------
#     def run(self):
#         print("=" * 65)
#         print("  Predictive Maintenance — Sensor Plant Simulator")
#         print(f"  Output CSV   : {self.output_csv}")
#         print(f"  Logging rate : {LOGGING_RATE_SECONDS} s")
#         print(f"  Fault inject : {self.inject_faults}")
#         print(f"  Max rows     : {self.max_rows or 'unlimited'}")
#         print("=" * 65)

#         fieldnames = self._get_csv_fieldnames()
#         file_exists = os.path.isfile(self.output_csv)

#         with open(self.output_csv, "a", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             if not file_exists:
#                 writer.writeheader()

#             try:
#                 while True:
#                     t_start = time.time()

#                     # Determine health state
#                     if self.inject_faults:
#                         state, load, faults, fault_label = self.sched.get_state()
#                     else:
#                         state       = HealthState.HEALTHY
#                         load        = 0.75
#                         faults      = {}
#                         fault_label = "none"

#                     self._apply_faults(load, faults)

#                     # Collect sensor readings
#                     sample = self._collect_sample()
#                     sample = self._add_labels(sample, state, fault_label, load)

#                     # Write to CSV
#                     writer.writerow(sample)
#                     f.flush()
#                     self._row_count += 1

#                     # Print to console
#                     if self.verbose:
#                         self._print_sample(sample)

#                     # JSON side-output (latest reading)
#                     if self.json_output:
#                         with open("latest_reading.json", "w") as jf:
#                             json.dump(sample, jf, indent=2)

#                     # Stop after max_rows
#                     if self.max_rows and self._row_count >= self.max_rows:
#                         print(f"\n[INFO] Reached max rows ({self.max_rows}). Stopping.")
#                         break

#                     # Precise sleep to maintain 5 s interval
#                     elapsed = time.time() - t_start
#                     sleep_t = max(0, LOGGING_RATE_SECONDS - elapsed)
#                     time.sleep(sleep_t)

#             except KeyboardInterrupt:
#                 print(f"\n[INFO] Stopped by user. Logged {self._row_count} rows → {self.output_csv}")

#     # ------------------------------------------------------------------
#     def _print_sample(self, s: dict):
#         sep = "-" * 65
#         print(f"\n{sep}")
#         print(f"  [{s['timestamp']}]  State: {s['health_state'].upper():<10}  "
#               f"Fault: {s['fault_type']}")
#         print(sep)
#         print(f"  MOTOR  │ Speed: {s['motor_speed_rpm']:>7.1f} RPM  "
#               f"│ Current: {s['motor_current_a']:>6.3f} A  "
#               f"│ Voltage: {s['motor_voltage_v']:>6.1f} V")
#         print(f"         │ Power: {s['motor_power_kw']:>7.3f} kW   "
#               f"│ Vibration: {s['motor_vibration_mm_s']:>5.3f} mm/s")
#         print(f"  TEMP   │ Ambient: {s['temp_ambient_c']:>5.1f} °C  "
#               f"│ Surface: {s['temp_surface_c']:>5.1f} °C  "
#               f"│ Bearing: {s['temp_bearing_c']:>5.1f} °C")
#         print(f"  MPU    │ Accel RMS  X:{s['accel_x_rms_ms2']:>6.4f}  "
#               f"Y:{s['accel_y_rms_ms2']:>6.4f}  Z:{s['accel_z_rms_ms2']:>6.4f} m/s²")
#         print(f"         │ Gyro  RMS  X:{s['gyro_x_rms_dps']:>6.4f}  "
#               f"Y:{s['gyro_y_rms_dps']:>6.4f}  Z:{s['gyro_z_rms_dps']:>6.4f} °/s")
#         print(f"         │ Vibration severity: {s['vibration_severity_ms2']:>7.4f} m/s²  "
#               f"│ Load: {s['load_factor']*100:.0f}%  "
#               f"│ Label: {s['health_label']}")


# # ---------------------------------------------------------------------------
# # CLI Entry Point
# # ---------------------------------------------------------------------------

# def parse_args():
#     p = argparse.ArgumentParser(
#         description="Sensor Plant Simulator for Predictive Maintenance"
#     )
#     p.add_argument("--output",   "-o", default=OUTPUT_CSV,
#                    help="Output CSV file path (default: sensor_data.csv)")
#     p.add_argument("--rows",     "-n", type=int, default=None,
#                    help="Number of rows to log (default: unlimited)")
#     p.add_argument("--no-faults", action="store_true",
#                    help="Disable fault injection (healthy data only)")
#     p.add_argument("--quiet",    "-q", action="store_true",
#                    help="Suppress console output")
#     p.add_argument("--json",     "-j", action="store_true",
#                    help="Also write latest reading to latest_reading.json")
#     return p.parse_args()


# if __name__ == "__main__":
#     args = parse_args()

#     simulator = PlantSimulator(
#         output_csv    = args.output,
#         verbose       = not args.quiet,
#         inject_faults = not args.no_faults,
#         max_rows      = args.rows,
#         json_output   = args.json,
#     )
#     simulator.run()

"""
=============================================================================
  Predictive Maintenance - Sensor Data Plant Simulator
=============================================================================
"""

import time
import math
import random
import csv
import os
import json
import argparse
from datetime import datetime
from enum import Enum

LOGGING_RATE_SECONDS = 5
OUTPUT_CSV = "sensor_data.csv"
MAX_LOG_LINES = None

MOTOR_RATED_SPEED_RPM   = 1500
MOTOR_RATED_CURRENT_A   = 10.0
MOTOR_RATED_VOLTAGE_V   = 400.0
MOTOR_RATED_POWER_KW    = 5.5

TEMP_AMBIENT_BASE   = 35.0
TEMP_SURFACE_BASE   = 55.0
TEMP_BEARING_BASE   = 60.0

ACCEL_IDLE_MS2  = 0.05
GYRO_IDLE_DPS   = 0.10

class HealthState(Enum):
    HEALTHY         = "healthy"
    DEGRADED        = "degraded"
    WARNING         = "warning"
    CRITICAL        = "critical"
    FAULT           = "fault"

HEALTH_LABEL = {
    HealthState.HEALTHY  : 0,
    HealthState.DEGRADED : 1,
    HealthState.WARNING  : 2,
    HealthState.CRITICAL : 3,
    HealthState.FAULT    : 4,
}

FAULT_TYPES = {
    "none"              : 0,
    "overload"          : 1,
    "overheat"          : 2,
    "bearing_wear"      : 3,
    "imbalance"         : 4,
    "phase_loss"        : 5,
    "insulation_failure": 6,
}

class MotorSensor:
    def __init__(self):
        self.load_factor   = 0.75
        self.phase_loss    = False
        self.overload      = False

    def set_operating_condition(self, load_factor, phase_loss=False, overload=False):
        self.load_factor = max(0.0, min(1.0, load_factor))
        self.phase_loss  = phase_loss
        self.overload    = overload

    def read(self):
        lf = self.load_factor
        slip = 0.03 * lf
        speed = MOTOR_RATED_SPEED_RPM * (1 - slip) + random.gauss(0, 2)
        current = MOTOR_RATED_CURRENT_A * (0.3 + 0.7 * lf)
        if self.overload:
            current *= random.uniform(1.15, 1.35)
        if self.phase_loss:
            current *= random.uniform(1.4, 1.8)
        current += random.gauss(0, 0.05)
        voltage = MOTOR_RATED_VOLTAGE_V + random.gauss(0, 2)
        if self.phase_loss:
            voltage = MOTOR_RATED_VOLTAGE_V * random.uniform(0.6, 0.75)
        power_factor = 0.85
        power = max(0, (math.sqrt(3) * voltage * current * power_factor) / 1000.0)
        vibration_rms = 1.2 + 0.8 * lf + random.gauss(0, 0.05)
        if self.overload:
            vibration_rms += random.uniform(0.5, 1.5)
        return {
            "motor_speed_rpm"     : round(speed, 2),
            "motor_current_a"     : round(current, 3),
            "motor_voltage_v"     : round(voltage, 2),
            "motor_power_kw"      : round(power, 3),
            "motor_vibration_mm_s": round(vibration_rms, 3),
        }

class TemperatureSensor:
    def __init__(self):
        self.load_factor       = 0.75
        self.bearing_wear      = False
        self.insulation_fail   = False
        self._accumulated_heat = 0.0

    def set_operating_condition(self, load_factor, bearing_wear=False, insulation_fail=False):
        self.load_factor     = max(0.0, min(1.0, load_factor))
        self.bearing_wear    = bearing_wear
        self.insulation_fail = insulation_fail
        delta = (load_factor - 0.5) * 0.8
        self._accumulated_heat = min(max(0, self._accumulated_heat + delta), 40)

    def read(self):
        lf = self.load_factor
        ambient = TEMP_AMBIENT_BASE + random.gauss(0, 0.3)
        surface = TEMP_SURFACE_BASE + 20 * lf + self._accumulated_heat * 0.5 + random.gauss(0, 0.4)
        if self.insulation_fail:
            surface += random.uniform(10, 20)
        bearing = TEMP_BEARING_BASE + 15 * lf + self._accumulated_heat * 0.6 + random.gauss(0, 0.3)
        if self.bearing_wear:
            bearing += random.uniform(8, 25)
        return {
            "temp_ambient_c" : round(ambient, 2),
            "temp_surface_c" : round(surface, 2),
            "temp_bearing_c" : round(bearing, 2),
        }

class MPUSensor:
    def __init__(self):
        self.load_factor    = 0.75
        self.imbalance      = False
        self.bearing_wear   = False
        self._t             = 0.0
        self.internal_fs    = 200
        self.window_samples = self.internal_fs * LOGGING_RATE_SECONDS

    def set_operating_condition(self, load_factor, imbalance=False, bearing_wear=False):
        self.load_factor  = max(0.0, min(1.0, load_factor))
        self.imbalance    = imbalance
        self.bearing_wear = bearing_wear

    def _generate_window(self):
        lf = self.load_factor
        n  = self.window_samples
        dt = 1.0 / self.internal_fs
        ax_buf, ay_buf, az_buf = [], [], []
        gx_buf, gy_buf, gz_buf = [], [], []
        base_vib = ACCEL_IDLE_MS2 + 0.15 * lf
        for i in range(n):
            t = self._t + i * dt
            rot_freq = (MOTOR_RATED_SPEED_RPM * (1 - 0.03 * lf)) / 60.0
            ax = base_vib * math.sin(2 * math.pi * rot_freq * t)
            ay = base_vib * math.cos(2 * math.pi * rot_freq * t)
            az = 9.81 + base_vib * 0.3 * math.sin(4 * math.pi * rot_freq * t)
            if self.imbalance:
                imb = random.uniform(0.5, 1.5)
                ax += imb * math.sin(2 * math.pi * rot_freq * t + 0.5)
                ay += imb * math.cos(2 * math.pi * rot_freq * t + 0.5)
            if self.bearing_wear:
                bdf   = 3.5 * rot_freq
                b_amp = random.uniform(0.1, 0.4)
                ax += b_amp * math.sin(2 * math.pi * bdf * t)
                ay += b_amp * math.sin(2 * math.pi * bdf * t)
            ax += random.gauss(0, 0.02); ay += random.gauss(0, 0.02); az += random.gauss(0, 0.01)
            gx = GYRO_IDLE_DPS * math.sin(2 * math.pi * rot_freq * t) + random.gauss(0, 0.05)
            gy = GYRO_IDLE_DPS * math.cos(2 * math.pi * rot_freq * t) + random.gauss(0, 0.05)
            gz = GYRO_IDLE_DPS * 0.2 * math.sin(4 * math.pi * rot_freq * t) + random.gauss(0, 0.02)
            ax_buf.append(ax); ay_buf.append(ay); az_buf.append(az)
            gx_buf.append(gx); gy_buf.append(gy); gz_buf.append(gz)
        self._t += n * dt
        return ax_buf, ay_buf, az_buf, gx_buf, gy_buf, gz_buf

    def read(self):
        ax_b, ay_b, az_b, gx_b, gy_b, gz_b = self._generate_window()
        rms  = lambda b: math.sqrt(sum(x**2 for x in b) / len(b))
        peak = lambda b: max(abs(x) for x in b)
        std  = lambda b: math.sqrt(sum((x - sum(b)/len(b))**2 for x in b) / len(b))
        mean = lambda b: sum(b) / len(b)
        ax_r, ay_r, az_r = rms(ax_b), rms(ay_b), rms(az_b)
        return {
            "accel_x_mean_ms2"      : round(mean(ax_b), 4),
            "accel_y_mean_ms2"      : round(mean(ay_b), 4),
            "accel_z_mean_ms2"      : round(mean(az_b), 4),
            "accel_x_rms_ms2"       : round(ax_r, 4),
            "accel_y_rms_ms2"       : round(ay_r, 4),
            "accel_z_rms_ms2"       : round(az_r, 4),
            "accel_x_peak_ms2"      : round(peak(ax_b), 4),
            "accel_y_peak_ms2"      : round(peak(ay_b), 4),
            "accel_z_peak_ms2"      : round(peak(az_b), 4),
            "accel_x_std_ms2"       : round(std(ax_b), 4),
            "accel_y_std_ms2"       : round(std(ay_b), 4),
            "accel_z_std_ms2"       : round(std(az_b), 4),
            "vibration_severity_ms2": round(math.sqrt(ax_r**2 + ay_r**2 + az_r**2), 4),
            "gyro_x_mean_dps"       : round(mean(gx_b), 4),
            "gyro_y_mean_dps"       : round(mean(gy_b), 4),
            "gyro_z_mean_dps"       : round(mean(gz_b), 4),
            "gyro_x_rms_dps"        : round(rms(gx_b), 4),
            "gyro_y_rms_dps"        : round(rms(gy_b), 4),
            "gyro_z_rms_dps"        : round(rms(gz_b), 4),
            "gyro_x_peak_dps"       : round(peak(gx_b), 4),
            "gyro_y_peak_dps"       : round(peak(gy_b), 4),
            "gyro_z_peak_dps"       : round(peak(gz_b), 4),
        }

class FaultScheduler:
    SCENARIO = [
        (20, HealthState.HEALTHY,  0.70, {}),
        (10, HealthState.HEALTHY,  0.90, {}),
        (15, HealthState.DEGRADED, 0.85, {"imbalance": True}),
        (10, HealthState.WARNING,  0.85, {"imbalance": True, "bearing_wear": True}),
        (8,  HealthState.CRITICAL, 0.80, {"imbalance": True, "bearing_wear": True, "overload": True}),
        (5,  HealthState.FAULT,    0.10, {"phase_loss": True}),
        (20, HealthState.HEALTHY,  0.65, {}),
        (12, HealthState.HEALTHY,  0.75, {}),
        (10, HealthState.DEGRADED, 0.80, {"insulation_fail": True}),
        (8,  HealthState.WARNING,  0.80, {"insulation_fail": True, "overload": True}),
        (6,  HealthState.CRITICAL, 0.75, {"insulation_fail": True, "overload": True, "overheat": True}),
        (5,  HealthState.FAULT,    0.0,  {"phase_loss": True}),
        (25, HealthState.HEALTHY,  0.72, {}),
    ]

    def __init__(self):
        self._index = 0
        self._count = 0

    def get_state(self):
        if self._index >= len(self.SCENARIO):
            self._index = 0
        duration, state, load, faults = self.SCENARIO[self._index]
        self._count += 1
        if self._count >= duration:
            self._count  = 0
            self._index += 1
        if state == HealthState.FAULT:
            fault_label = "phase_loss" if faults.get("phase_loss") else "fault"
        elif faults.get("insulation_fail") and state == HealthState.CRITICAL:
            fault_label = "insulation_failure"
        elif faults.get("bearing_wear"):  fault_label = "bearing_wear"
        elif faults.get("imbalance"):     fault_label = "imbalance"
        elif faults.get("overload"):      fault_label = "overload"
        elif faults.get("overheat"):      fault_label = "overheat"
        else:                             fault_label = "none"
        return state, load, faults, fault_label

class PlantSimulator:
    def __init__(self, output_csv=OUTPUT_CSV, verbose=True,
                 inject_faults=True, max_rows=None, json_output=False):
        self.output_csv    = output_csv
        self.verbose       = verbose
        self.inject_faults = inject_faults
        self.max_rows      = max_rows
        self.json_output   = json_output
        self.motor   = MotorSensor()
        self.temp    = TemperatureSensor()
        self.mpu     = MPUSensor()
        self.sched   = FaultScheduler()
        self._row_count = 0

    def _apply_faults(self, load, faults):
        self.motor.set_operating_condition(load, faults.get("phase_loss", False), faults.get("overload", False))
        self.temp.set_operating_condition(load, faults.get("bearing_wear", False), faults.get("insulation_fail", False))
        self.mpu.set_operating_condition(load, faults.get("imbalance", False), faults.get("bearing_wear", False))

    def _collect_sample(self):
        ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        sample = {"timestamp": ts}
        sample.update(self.motor.read())
        sample.update(self.temp.read())
        sample.update(self.mpu.read())
        return sample

    def _add_labels(self, sample, state, fault_label, load):
        sample["load_factor"]  = round(load, 2)
        sample["health_label"] = HEALTH_LABEL[state]
        sample["health_state"] = state.value
        sample["fault_type"]   = fault_label
        sample["fault_code"]   = FAULT_TYPES.get(fault_label, -1)
        return sample

    def _get_csv_fieldnames(self):
        base = {"timestamp": "", "load_factor": 0, "health_label": 0,
                "health_state": "", "fault_type": "", "fault_code": 0}
        base.update(self.motor.read())
        base.update(self.temp.read())
        base.update(self.mpu.read())
        return list(base.keys())

    def run(self):
        print("=" * 65)
        print("  Predictive Maintenance — Sensor Plant Simulator")
        print(f"  Output CSV   : {self.output_csv}")
        print(f"  Logging rate : {LOGGING_RATE_SECONDS} s")
        print(f"  Fault inject : {self.inject_faults}")
        print(f"  Max rows     : {self.max_rows or 'unlimited'}")
        print("=" * 65)

        fieldnames  = self._get_csv_fieldnames()
        file_exists = os.path.isfile(self.output_csv)

        with open(self.output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            try:
                while True:
                    t_start = time.time()

                    if self.inject_faults:
                        state, load, faults, fault_label = self.sched.get_state()
                    else:
                        state, load, faults, fault_label = HealthState.HEALTHY, 0.75, {}, "none"

                    self._apply_faults(load, faults)
                    sample = self._collect_sample()
                    sample = self._add_labels(sample, state, fault_label, load)

                    writer.writerow(sample)
                    f.flush()
                    self._row_count += 1

                    if self.verbose:
                        self._print_sample(sample)

                    if self.json_output:
                        with open("latest_reading.json", "w") as jf:
                            json.dump(sample, jf, indent=2)

                    # ── STOP AT MAX ROWS ──────────────────────────────────
                    if self.max_rows and self._row_count >= self.max_rows:
                        print(f"\n[INFO] Reached {self._row_count} rows. "
                              f"Saved to '{self.output_csv}'. Stopping.")
                        break
                    # ─────────────────────────────────────────────────────

                    elapsed = time.time() - t_start
                    time.sleep(max(0, LOGGING_RATE_SECONDS - elapsed))

            except KeyboardInterrupt:
                print(f"\n[INFO] Stopped by user. Logged {self._row_count} rows → {self.output_csv}")

    def _print_sample(self, s):
        sep = "-" * 65
        print(f"\n{sep}")
        print(f"  [{s['timestamp']}]  State: {s['health_state'].upper():<10}  Fault: {s['fault_type']}")
        print(sep)
        print(f"  MOTOR  │ Speed: {s['motor_speed_rpm']:>7.1f} RPM  │ Current: {s['motor_current_a']:>6.3f} A  │ Voltage: {s['motor_voltage_v']:>6.1f} V")
        print(f"         │ Power: {s['motor_power_kw']:>7.3f} kW   │ Vibration: {s['motor_vibration_mm_s']:>5.3f} mm/s")
        print(f"  TEMP   │ Ambient: {s['temp_ambient_c']:>5.1f} °C  │ Surface: {s['temp_surface_c']:>5.1f} °C  │ Bearing: {s['temp_bearing_c']:>5.1f} °C")
        print(f"  MPU    │ Accel RMS  X:{s['accel_x_rms_ms2']:>6.4f}  Y:{s['accel_y_rms_ms2']:>6.4f}  Z:{s['accel_z_rms_ms2']:>6.4f} m/s²")
        print(f"         │ Vibration severity: {s['vibration_severity_ms2']:>7.4f} m/s²  │ Load: {s['load_factor']*100:.0f}%  │ Label: {s['health_label']}")

def parse_args():
    p = argparse.ArgumentParser(description="Sensor Plant Simulator for Predictive Maintenance")
    p.add_argument("--output",    "-o", default=OUTPUT_CSV)
    p.add_argument("--rows",      "-n", type=int, default=5000)  # stop at 5000 rows
    p.add_argument("--no-faults", action="store_true")
    p.add_argument("--quiet",     "-q", action="store_true")
    p.add_argument("--json",      "-j", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    PlantSimulator(
        output_csv    = args.output,
        verbose       = not args.quiet,
        inject_faults = not args.no_faults,
        max_rows      = args.rows,
        json_output   = args.json,
    ).run()