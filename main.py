from model import train_model
import pandas as pd

model, scaler, columns = train_model()

# Create EMPTY row with all columns
sample = pd.DataFrame(columns=columns)

# Fill only known values
sample.loc[0, 'tenure'] = 12
sample.loc[0, 'MonthlyCharges'] = 80
sample.loc[0, 'TotalCharges'] = 960

# Fill rest with 0
sample = sample.fillna(0).infer_objects(copy=False)

# SCALE input (same as training)
sample = scaler.transform(sample)

prediction = model.predict(sample)

if prediction[0] == 1:
    print("Customer will churn")
else:
    print("Customer will stay")