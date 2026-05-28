"""
Sample Data Generator
Creates a realistic MTSamples-style CSV for local development.
Real MTSamples dataset: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
"""

import csv
import random
from pathlib import Path

OUTPUT_PATH = Path("data/sample/mtsamples.csv")

SAMPLE_NOTES = [
    {
        "description": "Chest pain evaluation",
        "medical_specialty": "Cardiology",
        "sample_name": "Chest Pain - Acute Coronary Syndrome",
        "transcription": (
            "CHIEF COMPLAINT: Chest pain. "
            "HISTORY: 58-year-old male presenting with substernal chest pain radiating to the left arm. "
            "Pain onset 2 hours ago, 8/10 severity. Associated diaphoresis and mild dyspnea. "
            "History of hypertension, hyperlipidemia. Current medications include aspirin 81mg daily, "
            "atorvastatin 40mg, lisinopril 10mg. "
            "EXAMINATION: BP 158/92, HR 96, O2 sat 97% on room air. EKG shows ST elevation in leads II, III, aVF. "
            "Troponin I pending. "
            "ASSESSMENT: ST-elevation myocardial infarction (STEMI), inferior wall. "
            "PLAN: Activate cath lab, aspirin 325mg loading dose, heparin drip, emergent cardiology consult."
        ),
        "keywords": "chest pain, STEMI, myocardial infarction, troponin, cath lab, aspirin",
    },
    {
        "description": "Type 2 Diabetes Management",
        "medical_specialty": "Endocrinology",
        "sample_name": "Diabetes Follow-up",
        "transcription": (
            "CHIEF COMPLAINT: Diabetes follow-up. "
            "HISTORY: 52-year-old female with Type 2 diabetes mellitus, diagnosed 6 years ago. "
            "Current medications: metformin 1000mg twice daily, glipizide 5mg daily. "
            "Last HbA1c was 8.2%, up from 7.8% three months ago. Patient reports poor dietary adherence. "
            "Denies polyuria, polydipsia, or hypoglycemic episodes. "
            "BMI 31.2. Blood pressure 138/84. Peripheral pulses intact bilaterally. "
            "Feet examined, no ulcers or neuropathic changes noted. "
            "ASSESSMENT: Type 2 diabetes, poorly controlled. Hypertension. Obesity. "
            "PLAN: Increase metformin to 2000mg daily. Add lisinopril for renal protection. "
            "Referral for diabetes education and nutrition counseling. Repeat HbA1c in 3 months. "
            "Annual ophthalmology and podiatry referrals placed."
        ),
        "keywords": "diabetes, metformin, HbA1c, insulin, glucose, hypertension, nephropathy",
    },
    {
        "description": "Community-acquired pneumonia",
        "medical_specialty": "Pulmonology",
        "sample_name": "Pneumonia - Inpatient",
        "transcription": (
            "CHIEF COMPLAINT: Productive cough and fever. "
            "HISTORY: 67-year-old male with 4-day history of productive cough with yellow sputum, "
            "fever up to 38.9°C, and shortness of breath on exertion. History of COPD and smoking. "
            "EXAMINATION: Temperature 38.7, HR 102, RR 22, O2 sat 91% on room air. "
            "Breath sounds decreased at right base with dullness to percussion. "
            "INVESTIGATIONS: CBC shows WBC 14,200. CXR demonstrates right lower lobe consolidation. "
            "Blood cultures drawn. Sputum culture sent. "
            "ASSESSMENT: Community-acquired pneumonia, right lower lobe. COPD. PORT score III. "
            "PLAN: Admit for IV antibiotics. Azithromycin 500mg IV daily plus ceftriaxone 1g IV daily. "
            "O2 supplementation to maintain saturation >94%. Albuterol nebulizers PRN. "
            "DVT prophylaxis with enoxaparin."
        ),
        "keywords": "pneumonia, cough, fever, azithromycin, ceftriaxone, COPD, consolidation",
    },
    {
        "description": "Left hip fracture",
        "medical_specialty": "Orthopedic",
        "sample_name": "Hip Fracture - Elderly",
        "transcription": (
            "CHIEF COMPLAINT: Left hip pain after fall. "
            "HISTORY: 78-year-old female presenting after mechanical fall at home. "
            "Complains of severe left hip pain, unable to bear weight. "
            "History of osteoporosis, on alendronate. Taking lisinopril and furosemide for CHF. "
            "EXAMINATION: Left lower extremity shortened and externally rotated. "
            "Severe pain with passive range of motion of left hip. Neurovascular exam intact. "
            "IMAGING: X-ray left hip shows displaced intracapsular femoral neck fracture. "
            "ASSESSMENT: Left femoral neck fracture, displaced. Osteoporosis. "
            "PLAN: NPO for surgery. Orthopaedics consult for hemiarthroplasty. "
            "DVT prophylaxis. Pain management with IV morphine. Calcium and Vitamin D supplementation."
        ),
        "keywords": "hip fracture, fall, osteoporosis, orthopedic, hemiarthroplasty, morphine, DVT",
    },
    {
        "description": "Ischemic stroke",
        "medical_specialty": "Neurology",
        "sample_name": "Acute Ischemic Stroke",
        "transcription": (
            "CHIEF COMPLAINT: Sudden onset left-sided weakness. "
            "HISTORY: 65-year-old male presenting with sudden onset left arm and leg weakness and facial droop. "
            "Last known well 90 minutes ago. History of atrial fibrillation, on warfarin (INR 1.6). "
            "History of hypertension. "
            "EXAMINATION: BP 182/96. NIHSS score 14. Left hemiplegia, left facial droop, dysarthria. "
            "CT head: no hemorrhage. CT angiography: right MCA occlusion. "
            "ASSESSMENT: Acute ischemic stroke, right MCA territory. Atrial fibrillation. "
            "PLAN: Activate stroke code. tPA eligibility assessment — within window, no contraindications. "
            "IV tPA 0.9mg/kg initiated. Interventional neurology for thrombectomy evaluation. "
            "ICU admission. Hold warfarin. Aspirin after 24 hours."
        ),
        "keywords": "stroke, CVA, tPA, hemiplegia, MCA, atrial fibrillation, warfarin, dysarthria",
    },
    {
        "description": "Major depressive disorder",
        "medical_specialty": "Psychiatry",
        "sample_name": "Depression - New Diagnosis",
        "transcription": (
            "CHIEF COMPLAINT: Low mood and inability to function. "
            "HISTORY: 34-year-old female presenting with 6-week history of persistent low mood, "
            "anhedonia, insomnia, fatigue, poor concentration, and weight loss of 8 lbs. "
            "Denies suicidal ideation. No prior psychiatric history. No current medications. "
            "EXAMINATION: Alert and oriented x3. Affect flat. Speech slow but coherent. "
            "PHQ-9 score: 18 (moderately severe). GAD-7 score: 11. "
            "ASSESSMENT: Major depressive disorder, moderate severity. Rule out hypothyroidism. "
            "PLAN: Initiate sertraline 50mg daily, titrate to 100mg in 2 weeks. "
            "Refer to outpatient cognitive behavioural therapy. "
            "TSH and CBC ordered. Follow-up in 2 weeks. Safety plan provided."
        ),
        "keywords": "depression, PHQ-9, sertraline, insomnia, anhedonia, CBT, psychiatric",
    },
    {
        "description": "Colorectal cancer staging",
        "medical_specialty": "Oncology",
        "sample_name": "Colon Cancer - New Diagnosis",
        "transcription": (
            "CHIEF COMPLAINT: Blood in stool and weight loss. "
            "HISTORY: 61-year-old male with 3-month history of bright red blood per rectum, "
            "unintentional weight loss of 15 lbs, and change in bowel habits. "
            "Colonoscopy revealed a 4cm mass in the sigmoid colon. Biopsy confirmed adenocarcinoma. "
            "EXAMINATION: Abdomen soft, mild left lower quadrant tenderness. "
            "INVESTIGATIONS: CT chest/abdomen/pelvis: sigmoid mass with 2 enlarged mesenteric nodes. No distant metastases. "
            "CEA 8.4 ng/mL (elevated). "
            "ASSESSMENT: Colorectal adenocarcinoma, stage IIIB (T3N1M0). "
            "PLAN: Multidisciplinary tumor board presentation. "
            "General surgery consult for laparoscopic sigmoid resection. "
            "Medical oncology consult for adjuvant FOLFOX chemotherapy post-operatively. "
            "Genetic counseling referral. Palliative care introduction."
        ),
        "keywords": "colon cancer, adenocarcinoma, chemotherapy, resection, oncology, CEA, FOLFOX",
    },
    {
        "description": "Acute appendicitis",
        "medical_specialty": "Gastroenterology",
        "sample_name": "Appendicitis - Surgical",
        "transcription": (
            "CHIEF COMPLAINT: Abdominal pain, worse over 18 hours. "
            "HISTORY: 24-year-old male with periumbilical pain migrating to right lower quadrant over 18 hours. "
            "Associated nausea and vomiting. No diarrhea. Last ate 12 hours ago. "
            "No prior abdominal surgeries. No current medications. "
            "EXAMINATION: Temperature 38.1, HR 94. Guarding and rebound tenderness at McBurney's point. "
            "Rovsing's sign positive. "
            "INVESTIGATIONS: WBC 14,800 with left shift. CT abdomen: dilated appendix 9mm with periappendiceal fat stranding. "
            "Alvarado score 8. "
            "ASSESSMENT: Acute appendicitis, uncomplicated. "
            "PLAN: NPO, IV fluids, IV cefazolin + metronidazole. "
            "General surgery consult — emergent laparoscopic appendectomy. "
            "Pain management with IV morphine and ondansetron."
        ),
        "keywords": "appendicitis, appendectomy, abdominal pain, nausea, WBC, metronidazole, surgery",
    },
    {
        "description": "Annual wellness exam",
        "medical_specialty": "General Medicine",
        "sample_name": "Wellness Visit - Adult",
        "transcription": (
            "CHIEF COMPLAINT: Annual wellness exam. "
            "HISTORY: 45-year-old female presenting for routine annual physical. "
            "No acute complaints. History of mild hypercholesterolemia managed with diet. "
            "Current medications: multivitamin, occasional ibuprofen for headaches. "
            "Non-smoker, social alcohol use only. Regular exercise 3x/week. "
            "Family history: father with MI at 62, mother with type 2 diabetes. "
            "EXAMINATION: BP 122/78, HR 68, BMI 24.1. All vitals normal. "
            "Physical exam unremarkable. Pap smear performed. "
            "INVESTIGATIONS: Lipid panel, CBC, CMP, TSH ordered. Fasting glucose ordered given family history. "
            "ASSESSMENT: Healthy adult female. Elevated cardiovascular risk family history. "
            "PLAN: Continue current lifestyle. Recommend mammogram. "
            "Colonoscopy at age 45 or sooner if family history warrants. "
            "Flu vaccine administered. Follow-up after lab results."
        ),
        "keywords": "wellness, preventive care, cholesterol, hypertension, screening, mammogram",
    },
    {
        "description": "Congestive heart failure exacerbation",
        "medical_specialty": "Cardiology",
        "sample_name": "CHF - Acute Decompensation",
        "transcription": (
            "CHIEF COMPLAINT: Worsening shortness of breath and leg swelling. "
            "HISTORY: 72-year-old male with known systolic CHF (EF 30%) presenting with 3-day history "
            "of progressive dyspnea, orthopnea, and bilateral leg edema. Gained 8 lbs over 1 week. "
            "Current medications: furosemide 40mg, carvedilol 12.5mg, lisinopril 5mg, spironolactone 25mg. "
            "Reports dietary noncompliance with high sodium intake. "
            "EXAMINATION: BP 148/90, HR 110, RR 24, O2 sat 88% on room air. "
            "JVD present. Bilateral crackles to mid-lung fields. 3+ pitting edema bilateral lower extremities. "
            "INVESTIGATIONS: BNP 1840 pg/mL. BMP: creatinine 1.4, eGFR 48. Echo: EF unchanged at 30%. "
            "CXR: cardiomegaly, bilateral pulmonary vascular congestion. "
            "ASSESSMENT: Acute decompensated congestive heart failure. CKD stage 3. "
            "PLAN: Admit to telemetry. IV furosemide 80mg bolus then infusion. "
            "O2 to maintain sat >94%. Fluid restriction 1.5L/day. "
            "Cardiology consult. Daily weights. Monitor renal function closely."
        ),
        "keywords": "heart failure, CHF, furosemide, BNP, edema, dyspnea, cardiomegaly",
    },
]


def generate_sample_data(n: int = 200, output_path: str = str(OUTPUT_PATH)) -> None:
    """
    Generate a synthetic MTSamples-style CSV by augmenting the base notes.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n):
        base = random.choice(SAMPLE_NOTES).copy()
        # Minor augmentation to create variety
        base["description"] = f"{base['description']} (case {i+1})"
        rows.append(base)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["description", "medical_specialty", "sample_name", "transcription", "keywords"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {n} sample records → {output_path}")


if __name__ == "__main__":
    generate_sample_data(n=200)
