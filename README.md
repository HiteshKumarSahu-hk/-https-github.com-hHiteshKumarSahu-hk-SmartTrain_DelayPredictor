Train Delay Predictor (Using Python & scikit-learn)

This is a Python project where I predict whether a train will be delayed by more than 60 minutes using Machine Learning. The script trains a Logistic Regression model on 2878 real-world journey records and achieves ~75.95% accuracy. After training, an interactive CLI lets the user enter journey details and get an instant prediction. I made this project to understand how machine learning works in Python.


What This Project Does
- Predicts if a train will be DELAYED (>60 min) or ON TIME
- Trained on 2878 rows of real journey data
- Encodes weather, day, time, train type, and congestion automatically
- Scales input features using StandardScaler
- Accepts live user input and gives instant prediction via CLI
- Achieves ~75.95% accuracy on test data


Requirements
Install the required libraries before running the script:
pip install pandas scikit-learn


Dataset Setup
Download the dataset and update the path in main.py:
https://www.kaggle.com/datasets/ravisingh0399/train-delay-dataset
dataset = pd.read_csv(r"C:\Users\hites\Downloads\train delay data.csv")
The dataset has 2878 rows and 7 columns: distance, weather, day, time, train_type, delay_min, congestion.


How to Use
1. Update the dataset path in main.py as shown above.
2. Run the file:
python main.py
3. Enter values when prompted:
Enter Distance (km): 200
Choose weather (0=Clear / 1=Rainy / 2=Foggy): 1
Choose day (0=Weekday / 1=Weekend): 0
Choose time (0=Morning / 1=Afternoon / 2=Evening / 3=Night): 2
Choose train type (0=Express / 1=Superfast / 2=Local): 1
Choose congestion (0=Low / 1=Medium / 2=High): 2
4. Output you will get:
Train will be DELAYED (more than 1hr delay)
   — or —
Train will be ON TIME (less than 1 hr delay)


How the Script Works (Simple Explanation)
- Loads the CSV and renames columns to lowercase
- Encodes categorical columns into numbers using fixed mappings:
     weather:     Clear=0, Rainy=1, Foggy=2
     day:         Mon-Fri=0 (Weekday), Sat-Sun=1 (Weekend)
     time:        Morning=0, Afternoon=1, Evening=2, Night=3
     train_type:  Express=0, Superfast=1, Local=2
     congestion:  Low=0, Medium=1, High=2
- Removes NaN rows using dropna()
- Creates target: delay_min > 60 = 1 (Delayed), else 0 (On Time)
- Splits data: 60% train, 40% test (random_state=50)
- Scales features using StandardScaler (fit on train only)
- Trains Logistic Regression model
- Accepts user input, scales it, and predicts delay


Model Info
- Algorithm: Logistic Regression
- Dataset: 2878 rows x 7 columns
- Train/Test Split: 60% training, 40% testing (random_state=50)
- Feature Scaling: StandardScaler
- Accuracy: 0.7595486111111112 (~75.95%)
- Features used: distance, weather, day, time, train_type, congestion
- Target: delay_min > 60 = 1 (Delayed), else 0 (On Time)


Why I Made This Project
I wanted to practice:
- Python data handling with pandas
- Using scikit-learn for machine learning
- Understanding how encoding and scaling work
- Building a real-world prediction tool from scratch
This project can be extended to add a web interface or use real-time train data.


Conclusion
This Train Delay Predictor project helped me understand how machine learning can be applied to real-world problems. Using Logistic Regression on 2878 journey records, the model achieves ~75.95% accuracy. It covers the full pipeline from data preprocessing to live prediction via CLI, and can be extended with more advanced models or a web app in the future.

