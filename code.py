'''   Distance Weather        Day       Time Train_type  Delay_min Congestion
0          100   Clear     Monday    Morning    Express          5        Low
1          150   Rainy    Tuesday  Afternoon  Superfast         10     Medium
2          200   Foggy  Wednesday    Evening      Local         15       High
3           50   Clear   Thursday      Night    Express          2        Low
4           75   Rainy     Friday    Morning  Superfast          8     Medium
...        ...     ...        ...        ...        ...        ...        ...
2873       945   Clear    Tuesday      Night      Local       1210     Medium
2874       925   Rainy  Wednesday    Morning    Express       1215       High
2875       950   Foggy   Thursday  Afternoon  Superfast       1220        Low
2876       930   Clear     Friday    Evening      Local       1225     Medium
2877       955   Rainy   Saturday      Night    Express       1230       High

[2878 rows x 7 columns]
Accuracy: 0.7595486111111112 '''

import pandas as pd

# Importing dataset
dataset = pd.read_csv(r"C:\Users\hites\Downloads\train delay data.csv")

# Changing column's name
dataset.columns = ['distance','weather','day','time','train_type','delay_min','congestion']

print(dataset)


# Doing Fixed Mappings
dataset['weather'] = dataset['weather'].map({'Clear': 0,'Rainy': 1,'Foggy': 2})

# Convert days into weekday/weekend
dataset['day'] = dataset['day'].map({'Monday': 0, 'Tuesday': 0, 'Wednesday': 0,'Thursday': 0, 'Friday': 0,'Saturday': 1, 'Sunday': 1})

dataset['time'] = dataset['time'].map({'Morning': 0,'Afternoon': 1,'Evening': 2,'Night': 3})

dataset['train_type'] = dataset['train_type'].map({'Express': 0,'Superfast': 1,'Local': 2})

dataset['congestion'] = dataset['congestion'].map({'Low': 0,'Medium': 1,'High': 2})


dataset = dataset.dropna()


# Creating Target

dataset['delay'] = dataset['delay_min'].apply(lambda x: 1 if x > 60 else 0)


# Features
X = dataset[['distance', 'weather', 'day', 'time', 'train_type', 'congestion']]
y = dataset['delay']



# Train Model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=50
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))


print("\n ===== Train Delay Predictor ===== \n")

# Distance
distance = int(input(" Enter Distance (km): "))

# Weather
print("\n Weather Options:")
print("0 → Clear")
print("1 → Rainy")
print("2 → Foggy")
weather = int(input("Choose weather (0/1/2): "))

# Day
print("\n Day Options:")
print("0 → Weekday")
print("1 → Weekend")
day = int(input("Choose day (0/1): "))

# Time
print("\n Time Options:")
print("0 → Morning")
print("1 → Afternoon")
print("2 → Evening")
print("3 → Night")
time = int(input("Choose time (0/1/2/3): "))

# Train Type
print("\n Train Type:")
print("0 → Express")
print("1 → Superfast")
print("2 → Local")
train_type = int(input("Choose train type (0/1/2): "))

# Congestion
print("\n Route Congestion:")
print("0 → Low")
print("1 → Medium")
print("2 → High")
congestion = int(input("Choose congestion (0/1/2): "))


# Creating DataFrame
new_data = pd.DataFrame({
    'distance': [distance],
    'weather': [weather],
    'day': [day],
    'time': [time],
    'train_type': [train_type],
    'congestion': [congestion]})

# Scale
new_data_scaled = scaler.transform(new_data)

# Prediction
prediction = model.predict(new_data_scaled)

print("\n RESULT:")
if prediction[0] == 1:
    print(" Train will be DELAYED(more than 1hr delay)")
else:
    print(" Train will be ON TIME(less than 1 hr delay)")
