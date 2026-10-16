# FemAI — Full-Stack ML-Powered PCOS Risk Awareness & Wellness Platform

FemAI is a clean, modern, responsive web application and educational machine learning prototype designed for **Polycystic Ovary Syndrome (PCOS)** risk awareness and evidence-informed lifestyle guidance.

> **Responsible Health Disclaimer:**  
> FemAI is an educational and risk-awareness prototype. **It is NOT a medical device, diagnostic test, or clinical probability calculator.** It evaluates user-reported symptom patterns against historical research data to foster informed discussions with licensed healthcare professionals. Never use FemAI to replace professional clinical diagnosis, pelvic ultrasound, or laboratory hormonal assays.

---

## 🌸 Visual & Design Philosophy

The application is styled with a **soft neuromorphic aesthetic**:
- **Background:** `#FFF8F5` (Warm off-white)
- **Primary Pink:** `#E9A8B8`
- **Light Pink:** `#F8DDE4`
- **Warm Cream:** `#FFF1E6`
- **Text:** `#493941` (Deep graphite)
- **Aesthetic Elements:** Rounded cards, dual soft drop-shadows, recessed inputs, pill-shaped buttons, accessible typography (*Plus Jakarta Sans*), and generous whitespace.

---

## 🏛 System Architecture & Technology Stack

```
FemAI/
├── backend/
│   ├── main.py                     # FastAPI entry point & CORS configuration
│   ├── requirements.txt            # Python dependencies (scikit-learn==1.6.1)
│   ├── models/
│   │   └── femai_svm_model.pkl     # Preserved trained Linear SVM pipeline
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── prediction.py           # Pydantic V2 validation schemas
│   ├── services/
│   │   ├── __init__.py
│   │   └── model_service.py        # Model loading & inference service
│   └── tests/
│       ├── __init__.py
│       └── test_api.py             # Pytest test suite (8 test cases)
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx          # Dynamic nav with live backend status indicator
│   │   │   ├── Footer.jsx          # Educational links and medical disclaimer
│   │   │   └── DisclaimerBanner.jsx# Reusable health notice component
│   │   ├── pages/
│   │   │   ├── HomePage.jsx        # Hero, awareness intro, workflow, overview
│   │   │   ├── AssessmentPage.jsx  # 10-feature questionnaire with validation
│   │   │   ├── ResultsPage.jsx     # Plain-language classification breakdown
│   │   │   ├── WellnessDashboard.jsx# Habit checklist & non-cure wellness pillars
│   │   │   └── AboutPage.jsx       # Transparent ML architecture & metrics
│   │   ├── services/
│   │   │   └── api.js              # Dedicated API client with graceful error handling
│   │   ├── App.jsx                 # React Router (JavaScript)
│   │   ├── App.css
│   │   ├── index.css               # Neuromorphic design system & Tailwind directives
│   │   └── main.jsx
│   ├── .env.example
│   ├── .env
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── package.json
├── femai.ipynb                     # Original research & training notebook
├── femai_svm_model.pkl             # Original root model artifact (preserved)
├── .gitignore
└── README.md
```

### Technology Highlights
- **Frontend:** React 18, Vite 5, JavaScript (no TypeScript), Tailwind CSS, Lucide React, React Router 6.
- **Backend:** Python 3.12, FastAPI, Pydantic V2, Uvicorn, Scikit-learn 1.6.1, Pandas, Joblib.
- **Testing:** Pytest & FastAPI TestClient.

---

## 🔬 Machine Learning Pipeline & Verified Metrics

### 1. Dataset Origin
- **Source:** Kaggle Polycystic Ovary Syndrome dataset (124,958 records).
- **Target:** `Diagnosis` (87.9% No, 12.1% Yes — heavily imbalanced).
- **Data Integrity:** `Undiagnosed PCOS Likelihood` and `Diagnosis` were strictly excluded from model features to prevent target leakage.

### 2. The 10 Model Features & Exact Category Values
| # | Feature Name | Type | Valid Values / Ranges | Preprocessing |
|---|--------------|------|-----------------------|---------------|
| 1 | `Age` | Numerical | 15.0 – 50.0 (Median: 32.0) | `SimpleImputer(median)` + `StandardScaler` |
| 2 | `BMI` | Categorical | `Normal`, `Overweight`, `Obese`, `Underweight` | `OneHotEncoder(handle_unknown='ignore')` |
| 3 | `Menstrual Regularity` | Categorical | `Regular`, `Irregular` | `OneHotEncoder` |
| 4 | `Hirsutism` | Categorical | `Yes`, `No` | `OneHotEncoder` |
| 5 | `Acne Severity` | Categorical | `Mild`, `Moderate`, `Severe`, `Not recorded` | `SimpleImputer("Not recorded")` + `OneHotEncoder` |
| 6 | `Family History of PCOS`| Categorical | `Yes`, `No` | `OneHotEncoder` |
| 7 | `Insulin Resistance` | Categorical | `Yes`, `No` | `OneHotEncoder` |
| 8 | `Lifestyle Score` | Numerical | 1.0 – 10.0 (Median: 6.0) | `SimpleImputer(median)` + `StandardScaler` |
| 9 | `Stress Levels` | Categorical | `Low`, `Medium`, `High` | `OneHotEncoder` |
| 10| `Fertility Concerns` | Categorical | `Yes`, `No` | `OneHotEncoder` |

### 3. Model Algorithm
- **Classifier:** `LinearSVC(class_weight='balanced', max_iter=3000, random_state=42)`
- **Sample Prediction Verified:**
  - Sample Input: 25yo, Overweight, Irregular cycles, Hirsutism, Moderate acne, No family history, No insulin resistance, Lifestyle Score 5, High stress, Fertility concerns.
  - Prediction: **Positive** (SVM decision score: `+0.2662`).

### 4. Verified Evaluation Metrics (24,992-Record Test Split)
- **Accuracy:** `58.1%`
- **Positive-Class Precision:** `14.1%`
- **Positive-Class Recall:** `48.3%`
- **Positive-Class F1-Score:** `0.218`

> **Key Model Limitation:**  
> In a dataset where 88% of patients are negative, a precision of 14.1% means that **approximately 6 out of 7 positive classifications by the linear model are false positives**. The SVM raw decision score is an uncalibrated geometric distance, **not a probability**, and is never represented as such to the user.

---

## 🚀 Setup & Run Instructions (Windows)

### Prerequisites
1. **Python 3.12** (Scikit-learn 1.6.1 requires Python ≤ 3.12).
2. **Node.js** (v18 or higher) & **npm**.

---

### Step 1: Set Up and Start the Backend

1. Open PowerShell or Command Prompt in the project root:
   ```powershell
   cd C:\Users\Shaily\OneDrive\Desktop\FemAI
   ```

2. Create and activate a Python 3.12 virtual environment (if not already created):
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install backend dependencies:
   ```powershell
   pip install -r backend\requirements.txt
   ```

4. Run the automated backend test suite:
   ```powershell
   python -m pytest backend\tests\test_api.py -v
   ```
   *(Expected output: 8 passed)*

5. Start the FastAPI development server:
   ```powershell
   python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
   ```
   - API Root: [http://127.0.0.1:8000](http://127.0.0.1:8000)
   - Interactive Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - Health Endpoint: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

### Step 2: Set Up and Start the Frontend

1. Open a second terminal window and navigate to `frontend`:
   ```powershell
   cd C:\Users\Shaily\OneDrive\Desktop\FemAI\frontend
   ```

2. Install npm dependencies (if not already installed):
   ```powershell
   npm install
   ```

3. Start the Vite React development server:
   ```powershell
   npm run dev
   ```

4. Open your browser at:
   👉 **`http://localhost:5173`**

---

## 📡 API Endpoints Reference

### `GET /health`
Returns system status and model readiness.
```json
{
  "status": "healthy",
  "model_loaded": true,
  "backend": "FastAPI",
  "message": "Model ready for inference"
}
```

### `GET /model-info`
Provides model metadata, input feature requirements, and disclaimers without exposing internal sensitive paths.

### `POST /predict`
Evaluates the 10 symptom features.
- **Request Body:**
```json
{
  "Age": 25,
  "BMI": "Overweight",
  "Menstrual Regularity": "Irregular",
  "Hirsutism": "Yes",
  "Acne Severity": "Moderate",
  "Family History of PCOS": "No",
  "Insulin Resistance": "No",
  "Lifestyle Score": 5,
  "Stress Levels": "High",
  "Fertility Concerns": "Yes"
}
```
- **Response Body:**
```json
{
  "prediction": "Positive",
  "result_type": "experimental_model_classification",
  "raw_decision_score": 0.2662,
  "disclaimer": "This experimental output is not a medical diagnosis. FemAI does not provide medical probabilities or replace clinical evaluation by a physician."
}
```
- **Error Handling:** Returns `422 Unprocessable Entity` for invalid inputs (e.g. invalid category names, out-of-range ages, missing fields) with clear validation messages.

---

## 🛡️ Privacy & Responsible AI Safeguards
- **Client-Side Privacy:** Assessment responses are evaluated transiently and are never persisted to a database or shared with third parties.
- **Sanitized Logging:** The backend logs technical execution and prediction labels without logging individual user health questionnaire answers.
- **Honest Communication:** No misleading percentages or risk probabilities are displayed to the user.
- **Graceful Failure:** If the backend is unreachable, the frontend displays clear diagnostic guidance rather than fabricating a fake prediction.
