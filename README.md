# Medsecure

# MedSecure: AI-Based EMR Access Compliance Auditor

**An Integrated AI Workflow for Detecting, Verifying, and Explaining Unusual Record Access Patterns in Healthcare Systems.**

## Project Overview

MedSecure is a simulated Electronic Medical Record (EMR) auditing system designed to detect and verify **insider threat** (snooping) behavior. It moves beyond traditional static rule-checking by integrating a hybrid AI approach combining formal reasoning (CSP, A\*) with machine learning (Random Forest) for robust, explainable, and autonomous compliance monitoring.

### Core Objectives:

1.  **Policy Enforcement:** Use **Constraint Satisfaction Problems (CSP)** to enforce hard policy rules (e.g., Nurses accessing only assigned patients).
2.  **Anomaly Detection:** Use a high-accuracy **Machine Learning Model** (Random Forest) to classify user sessions as **Benign** or **Suspicious**.
3.  **Snooping Path Analysis:** Use the **A\* Search Algorithm** to find the shortest (most suspicious, or "hurried") path to high-risk patient records (e.g., VIP patients).
4.  **Explainable AI (XAI):** Provide clear explanations for ML classifications via SHAP-like feature contributions.

## Technology Stack

  * **Backend:** Python 3, Flask
  * **AI/ML:** Scikit-learn (Random Forest, StandardScaler), NumPy
  * **Frontend:** HTML, CSS, JavaScript (for API calls and dashboard visualization)

## Setup and Installation

Follow these steps to get the MedSecure system running locally.

### 1\. Prerequisites

You must have Python 3 installed.

### 2\. Install Dependencies

Install the required Python libraries using pip:

```bash
pip install Flask scikit-learn pandas numpy
```

### 3\. File Structure

Ensure you have the following two files in the same directory:

| File Name | Description |
| :--- | :--- |
| `app.py` | The complete Flask backend, including the CSP Verifier, high-accuracy ML Model, and the A\* Search implementation. |
| `index.html` | The HTML/JavaScript frontend for the Staff Access Terminal and Admin Dashboard. |

### 4\. Run the Application

Execute the Python script to start the Flask development server:

```bash
python app.py
```

The system will initialize and print the local URL:

```
-> Backend: ML Model Trained and Ready (Simulated Accuracy: 100%).
=============================================
 MEDSECURE BACKEND SYSTEM ONLINE
 AI Models Initialized & Listening
 Open browser at: http://127.0.0.1:5000
=============================================
```

Open your web browser and navigate to the address: **`http://127.0.0.1:5000`**

## Usage and Demonstration Guide

Use the following scenarios to demonstrate the core AI security features.

### A. Constraint Satisfaction Problem (CSP) Violation

The CSP policy dictates: **A Nurse cannot access an unassigned patient record.**

1.  From the landing page, select **Nurse Access**.
2.  In the Patient Record ID field, enter **`P104`** (a dedicated VIP/Unassigned patient).
3.  Click **Access Patient Record**.

**Expected Result:**

  * **Access Terminal:** **ACCESS DENIED** is displayed.
  * **Result Message:** `CSP VIOLATION: Nurse accessed unassigned patient record.`
  * **Admin Dashboard:** A red violation flag is logged in the `Access Log History`.

### B. Machine Learning (ML) Suspicious Score & XAI

The ML model detects behavioral anomalies like a high rate of unassigned patient access.

1.  From the landing page, select **Doctor Access**. (Doctors are allowed to access unassigned patients, but it is logged as a risk feature for the ML model).
2.  Rapidly access several unassigned patient IDs:
      * Enter **`P999`** and click **Access Patient Record**.
      * Enter **`P998`** and click **Access Patient Record**.
      * Enter **`P997`** and click **Access Patient Record**.
      * Repeat this process until you have logged at least **10-15 total accesses** using different or repeating non-assigned IDs.
3.  Go to the **Admin Dashboard**.

**Expected Result:**

  * **ML Session Classification:** Changes from "Benign" to **"SUSPICIOUS"**.
  * **ML Suspicious Score:** Rises significantly (e.g., to **90.00%+**).
  * **XAI Feature Contribution Analysis:** Highlights the main drivers:
      * `CRITICAL: Unassigned access rate is (High Percentage)%`
      * `HIGH VOLUME: (Number > 15) records accessed`

### C. A\* Suspicious Snooping Path Detection

The A\* algorithm prioritizes paths leading to high-sensitivity patients (e.g., `P104` - VIP).

1.  From the landing page, select **Doctor Access**.
2.  Establish a non-suspicious path first:
      * Access **`P101`** (simulated ordinary patient).
      * Access **`P456`** (simulated assigned patient).
3.  Immediately access the VIP patient:
      * Access **`P104`** (VIP Patient).
4.  Go to the **Admin Dashboard**.

**Expected Result:**

  * **A\* Suspicious Path Analysis:**
      * **Path:** Will show a path similar to **`P101 → P456 → P104 (VIP Patient)`**.
      * **Total Suspicion Score:** Will display the high-risk score of **`16.50`** (or similar, indicating the path to the VIP patient was found). *(As per project documentation, a score of 16.50 represents the highest severity snooping path).*
