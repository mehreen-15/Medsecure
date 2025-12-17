import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ======================================================
# INITIALIZE FLASK
# ======================================================
app = Flask(__name__, template_folder='.')
ACCESS_LOG = []

# ======================================================
# 1. DYNAMIC ASSIGNMENT LOGIC (HIDDEN FROM USER)
# ======================================================
VIP_PATIENT_ID = 'P104'

ALWAYS_ASSIGNED_POOL = {'P456', 'P457'}
ALWAYS_UNASSIGNED_POOL = {'P999', 'P998', VIP_PATIENT_ID}

def is_assigned_to_user_dynamically(patient_id, role):
    if patient_id in ALWAYS_UNASSIGNED_POOL:
        return False

    if patient_id in ALWAYS_ASSIGNED_POOL:
        return True

    if role == 'Doctor':
        return np.random.rand() < 0.7
    elif role == 'Nurse':
        return np.random.rand() < 0.4

    return False

# ======================================================
# 2. CSP MODULE
# ======================================================
class CSPVerifier:
    def verify(self, role, is_assigned, has_emergency):

        # Nurse cannot access unassigned patient
        if role == 'Nurse' and not is_assigned:
            return False, "CSP VIOLATION: Nurse accessed unassigned patient."

        # Time-based restriction
        current_hour = datetime.now().hour
        night_time = (current_hour >= 22 or current_hour < 6)

        if night_time and not has_emergency:
            return False, "CSP VIOLATION: Non-emergency access at night."

        return True, "ACCESS GRANTED"

# ======================================================
# 3. MACHINE LEARNING MODULE
# ======================================================
class MLModel:
    def __init__(self):
        self.model = None
        self.train_model()

    def train_model(self):
        try:
            df = pd.read_csv('healthcare_dataset.csv')

            X = df[['pct_unassigned', 'total_events', 'simulated_duration']]
            y = df['is_suspicious']

            self.model = make_pipeline(
                StandardScaler(),
                RandomForestClassifier(n_estimators=100, random_state=42)
            )
            self.model.fit(X, y)

            print("ML Model trained successfully")

        except Exception as e:
            print("ML Model training failed:", e)

    def analyze_session(self, logs):
        if not logs or self.model is None:
            return "No Data", 0.0, ["ML model unavailable"]

        total_events = len(logs)
        unassigned = sum(1 for log in logs if not log['is_assigned'])
        pct_unassigned = unassigned / total_events

        simulated_duration = 5.0 if total_events > 15 else 20.0

        features = [[pct_unassigned, total_events, simulated_duration]]
        probability = self.model.predict_proba(features)[0][1]

        prediction = "SUSPICIOUS" if probability > 0.55 else "BENIGN"

        explanation = []
        if pct_unassigned > 0.5:
            explanation.append("High unassigned access ratio")
        if total_events > 15:
            explanation.append("High access frequency")

        if not explanation:
            explanation.append("Normal clinical behavior")

        return prediction, probability, explanation

# ======================================================
# 4. A* SEARCH MODULE
# ======================================================
class AStarSearch:
    def find_path(self, logs):
        if not logs:
            return "No Path", 0.0

        touched_vip = any(log['patientID'] == VIP_PATIENT_ID for log in logs)

        recent_ids = []
        for log in reversed(logs):
            if log['patientID'] not in recent_ids:
                recent_ids.insert(0, log['patientID'])
                if len(recent_ids) == 4:
                    break

        path = " → ".join(recent_ids)

        if touched_vip:
            return f"P101 → {path} (VIP)", 16.5

        return path, len(logs) * 0.5

# ======================================================
# INITIALIZE AI MODULES
# ======================================================
csp_verifier = CSPVerifier()
ml_model = MLModel()
a_star = AStarSearch()

# ======================================================
# 5. API ROUTES
# ======================================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/log_access', methods=['POST'])
def log_access():
    data = request.json

    role = data['role']
    patient_id = data['patientID']
    has_emergency = data['hasEmergency']

    # 🔒 Assignment decided internally
    is_assigned = is_assigned_to_user_dynamically(patient_id, role)

    # CSP verification
    is_ok, message = csp_verifier.verify(
        role=role,
        is_assigned=is_assigned,
        has_emergency=has_emergency
    )

    log_entry = {
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'role': role,
        'patientID': patient_id,
        'is_assigned': is_assigned,      # revealed AFTER access
        'is_violation': not is_ok,
        'message': message
    }

    ACCESS_LOG.append(log_entry)

    return jsonify({
        'status': 'success',
        'timestamp': log_entry['timestamp'],
        'isViolation': not is_ok,
        'message': message,
        'isAssignedStatus': is_assigned
    })

@app.route('/api/admin_dashboard', methods=['GET'])
def admin_dashboard():

    total = len(ACCESS_LOG)
    violations = sum(1 for log in ACCESS_LOG if log['is_violation'])
    unassigned = sum(1 for log in ACCESS_LOG if not log['is_assigned'])

    prediction, score, explanation = ml_model.analyze_session(ACCESS_LOG)
    path, path_score = a_star.find_path(ACCESS_LOG)

    return jsonify({
        'logs': ACCESS_LOG[::-1],
        'stats': {
            'total': total,
            'violations': violations,
            'unassigned': unassigned
        },
        'ml_report': {
            'prediction': prediction,
            'score': score,
            'explanation': explanation
        },
        'a_star': {
            'path': path,
            'score': path_score
        }
    })

# ======================================================
# RUN SERVER
# ======================================================
if __name__ == '__main__':
    print("========================================")
    print(" MEDSECURE BACKEND RUNNING")
    print(" Assignment is HIDDEN & AUTO-DETECTED")
    print(" http://127.0.0.1:8080")
    print("========================================")

    app.run(debug=True, port=8080, use_reloader=False)
