import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
df = pd.read_csv("data/MFG10YearTerminationData.csv")

# Target variable
df['STATUS'] = df['STATUS'].map({
    'ACTIVE': 0,
    'TERMINATED': 1
})

# Select only useful features
selected_features = [
    'age',
    'length_of_service',
    'city_name',
    'department_name',
    'gender_full',
    'BUSINESS_UNIT'
]

# Encode categorical columns
encoders = {}

for col in ['city_name', 'department_name', 'gender_full', 'BUSINESS_UNIT']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Features and Target
X = df[selected_features]
y = df['STATUS']

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Build ANN
model = Sequential()

model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

# Save model
model.save("model.h5")

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save encoders
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoders, f)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)

print(f"Accuracy: {acc * 100:.2f}%")