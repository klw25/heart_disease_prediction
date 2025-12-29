import joblib
import os
from compare_model import compare_model

def export_model():
    # This calls the function and gets the returned model
    model = compare_model()

    base_dir = os.path.dirname(__file__)
    save_path = os.path.join(base_dir, 'medical_model.pkl')

    joblib.dump(model, save_path)
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    export_model()