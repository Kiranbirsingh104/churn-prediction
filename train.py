from model import train_model
import joblib

# Get all outputs
model, sc, ct = train_model()

# Save model
model.save("model.h5")

# Save preprocessing objects
joblib.dump(sc, "scaler.pkl")
joblib.dump(ct, "ct.pkl")
