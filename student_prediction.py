import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("data/student_data.csv")

# Input features
X = data[[
    "study_hours",
    "attendance",
    "previous_marks",
    "assignment_score"
]]

# Target value
y = data["final_marks"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Student Performance Prediction")
print("--------------------------------")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# Predict marks for a new student
study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))
previous_marks = float(input("Enter previous marks: "))
assignment_score = float(input("Enter assignment score: "))

new_student = pd.DataFrame([[
    study_hours,
    attendance,
    previous_marks,
    assignment_score
]], columns=X.columns)

predicted_marks = model.predict(new_student)

print("Predicted Final Marks:", round(predicted_marks[0], 2))
