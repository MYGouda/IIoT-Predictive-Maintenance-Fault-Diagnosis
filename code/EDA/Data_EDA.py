import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load Dataset
df=pd.read_csv('C:\MyWorkingDir\Workingpython\Msc Project\Data\sensor_data.csv')
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

#Visulaize Class distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='health_binary')
plt.title('Health Vs Faulty')
plt.xlabel('Health label')
plt.ylabel('Count')
plt.savefig('Health Vs Faulty')
plt.show()

#Visulaize Each State distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='health_state')
plt.title('Health State Distribution')
plt.xlabel('Health State')
plt.ylabel('Count')
plt.savefig('Health State Distribution.png')
plt.show()


#Visualizing data
Features=['motor_speed_rpm', 'motor_current_a', 'motor_voltage_v', 'motor_power_kw',
'motor_vibration_mm_s','temp_surface_c', 'temp_bearing_c','vibration_severity_ms2']

df[Features].hist(bins=30, figsize=(10,8))
plt.suptitle('Features Distribution')
plt.tight_layout()
plt.savefig('Features Distribution.png')
plt.show()



#Health State Comparison
imp_features=['temp_bearing_c','motor_current_a','motor_power_kw','motor_vibration_mm_s']

for feature in imp_features:
    plt.figure(figsize=(6,5))
    sns.boxenplot(data= df, 
                  x= 'health_state',
                  y=feature
                  )
    plt.title(f'{feature} by health_state')
    plt.tight_layout()
    plt.show()


#Exploring Distribution

# for feature in imp_features:
#     plt.figure(figsize=(6,5))

#     sns.histplot(data=df, x=feature, bins=30, kde=True )
#     plt.title(f'Distribution of {feature}')
#     plt.tight_layout()
#     plt.show()

STATE_ORDER   = ["healthy", "degraded", "warning", "critical", "fault"]
STATE_PALETTE = {"healthy":"#2ecc71", "degraded":"#f1c40f",
                 "warning":"#e67e22", "critical":"#e74c3c", "fault":"#8e44ad"}

## motor_current_a by health state
plt.figure(figsize=(10, 5))
for state in STATE_ORDER:
    subset = df[df["health_state"] == state]["motor_current_a"]
    sns.kdeplot(subset, label=state, color=STATE_PALETTE[state],
                fill=True, alpha=0.25, linewidth=2)

plt.title("motor_current_a distribution by health state",
          fontsize=13, fontweight="bold")
plt.xlabel("Motor Current (A)")
plt.ylabel("Density")
plt.legend(title="Health state")
plt.tight_layout()
plt.savefig("motor_current_by_state.png", dpi=150, bbox_inches="tight")
plt.show()

## temp_bearing_c by health state
plt.figure(figsize=(10, 5))
for state in STATE_ORDER:
    subset = df[df["health_state"] == state]["temp_bearing_c"]
    sns.kdeplot(subset, label=state, color=STATE_PALETTE[state],
                fill=True, alpha=0.25, linewidth=2)

plt.title("temp_bearing_c distribution by health state", fontsize=13, fontweight="bold")
plt.xlabel("Bearing Temperature (°C)")
plt.ylabel("Density")
plt.legend(title="Health state")
plt.tight_layout()
plt.savefig("temp_bearing_by_state.png", dpi=150, bbox_inches="tight")
plt.show()

## motor_power_kw by health state
plt.figure(figsize=(10, 5))
for state in STATE_ORDER:
    subset = df[df["health_state"] == state]["motor_power_kw"]
    sns.kdeplot(subset, label=state, color=STATE_PALETTE[state],
                fill=True, alpha=0.25, linewidth=2)

plt.title("motor_power_kw distribution by health state",
          fontsize=13, fontweight="bold")
plt.xlabel("Motor Power (kw)")
plt.ylabel("Density")
plt.legend(title="Health state")
plt.tight_layout()
plt.savefig("motor_power_by_state.png", dpi=150, bbox_inches="tight")
plt.show()

## motor_vibration_mm_s by health state
plt.figure(figsize=(10, 5))
for state in STATE_ORDER:
    subset = df[df["health_state"] == state]["motor_vibration_mm_s"]
    sns.kdeplot(subset, label=state, color=STATE_PALETTE[state],
                fill=True, alpha=0.25, linewidth=2)

plt.title("motor_vibration_mm_s distribution by health state",
          fontsize=13, fontweight="bold")
plt.xlabel("Motor Vibration (mm/s)")
plt.ylabel("Density")
plt.legend(title="Health state")
plt.tight_layout()
plt.savefig("motor_vibration_by_state.png", dpi=150, bbox_inches="tight")
plt.show()