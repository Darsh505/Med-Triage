from dataclasses import dataclass
from typing import Dict


@dataclass
class PatientScenario:
    id: str
    initial_symptoms: str
    vitals: Dict[str, str]
    exam_findings: Dict[str, str]
    test_results: Dict[str, str]
    correct_diagnosis: str
    correct_treatment: str


PATIENT_POOL = [
    PatientScenario(
        id="P001",
        initial_symptoms="Patient presents with right lower quadrant (RLQ) pain, nausea, and fever.",
        vitals={"HR": "105", "BP": "120/80", "Temp": "101.2F", "RR": "18"},
        exam_findings={"abdomen": "RLQ rebound tenderness and guarding."},
        test_results={
            "blood_cbc": "WBC 16,000 (Elevated)",
            "ultrasound_abdomen": "Enlarged appendix with periappendiceal fluid.",
        },
        correct_diagnosis="appendicitis",
        correct_treatment="surgery",
    ),
    PatientScenario(
        id="P002",
        initial_symptoms="Patient presents with sudden onset of left-sided weakness and slurred speech.",
        vitals={"HR": "88", "BP": "180/100", "Temp": "98.6F", "RR": "16"},
        exam_findings={"neurological": "Left facial droop, left arm drift, expressive aphasia."},
        test_results={
            "ct_head": "No acute hemorrhage. Early signs of ischemic changes in right MCA territory.",
            "blood_glucose": "110 mg/dL",
        },
        correct_diagnosis="ischemic_stroke",
        correct_treatment="thrombolytics",
    ),
    PatientScenario(
        id="P003",
        initial_symptoms="Patient complaining of severe crushing chest pain radiating to the left arm, and diaphoresis.",
        vitals={"HR": "110", "BP": "150/90", "Temp": "98.8F", "RR": "22"},
        exam_findings={"chest": "Clear to auscultation. Heart regular rate, clammy skin."},
        test_results={
            "ecg": "ST elevation in leads II, III, aVF.",
            "blood_troponin": "Elevated at 0.5 ng/mL",
        },
        correct_diagnosis="myocardial_infarction",
        correct_treatment="pci_and_aspirin",
    ),
    PatientScenario(
        id="P004",
        initial_symptoms="Patient reports high fever, productive cough with rust-colored sputum, and chills for 3 days.",
        vitals={"HR": "115", "BP": "110/70", "Temp": "103.5F", "RR": "24"},
        exam_findings={"chest": "Crackles and decreased breath sounds in the right lower lobe."},
        test_results={
            "chest_xray": "Consolidation in the right lower lobe.",
            "blood_cbc": "WBC 18,000",
        },
        correct_diagnosis="pneumonia",
        correct_treatment="antibiotics",
    ),
    PatientScenario(
        id="P005",
        initial_symptoms="Patient complains of frequent watery diarrhea, abdominal cramping, and vomiting for 24 hours.",
        vitals={"HR": "110", "BP": "100/60", "Temp": "99.5F", "RR": "18"},
        exam_findings={
            "abdomen": "Hyperactive bowel sounds, diffuse mild tenderness, poor skin turgor."
        },
        test_results={
            "stool_culture": "Pending. Rapid viral panel positive for Norovirus.",
            "blood_cbc": "WBC 8,000 (Normal), mild hemoconcentration",
        },
        correct_diagnosis="viral_gastroenteritis",
        correct_treatment="iv_fluids",
    ),
    PatientScenario(
        id="P006",
        initial_symptoms="Patient presents with acute onset of shortness of breath and pleuritic chest pain after a long flight.",
        vitals={"HR": "120", "BP": "115/75", "Temp": "99.0F", "RR": "28"},
        exam_findings={"chest": "Clear lungs. Tachycardia. Unilateral right calf swelling."},
        test_results={
            "d_dimer": "Elevated > 1000 ng/mL",
            "ctpa": "Embolus in the right main pulmonary artery.",
        },
        correct_diagnosis="pulmonary_embolism",
        correct_treatment="anticoagulation",
    ),
    PatientScenario(
        id="P007",
        initial_symptoms="Patient describes frequent urination, excessive thirst, and unintentional weight loss over 2 weeks.",
        vitals={"HR": "95", "BP": "110/70", "Temp": "98.4F", "RR": "26"},
        exam_findings={"general": "Fruity breath odor. Deep, rapid breathing (Kussmaul)."},
        test_results={
            "blood_glucose": "540 mg/dL",
            "abg": "pH 7.15, primary metabolic acidosis",
            "urine": "Positive for severe ketones.",
        },
        correct_diagnosis="diabetic_ketoacidosis",
        correct_treatment="iv_insulin_and_fluids",
    ),
    PatientScenario(
        id="P008",
        initial_symptoms="Patient presents with severe, sharp, unilateral flank pain radiating to the groin, and hematuria.",
        vitals={"HR": "105", "BP": "140/85", "Temp": "98.9F", "RR": "20"},
        exam_findings={
            "abdomen": "Right costovertebral angle (CVA) tenderness. Patient constantly shifting positions."
        },
        test_results={
            "urine": "Microscopic hematuria, no nitrites.",
            "ct_abdomen": "5mm calculus in the right proximal ureter.",
        },
        correct_diagnosis="nephrolithiasis",
        correct_treatment="pain_control_and_fluids",
    ),
    PatientScenario(
        id="P009",
        initial_symptoms="Patient reports progressive headache, neck stiffness, and photophobia developing over 24 hours.",
        vitals={"HR": "110", "BP": "125/80", "Temp": "102.5F", "RR": "18"},
        exam_findings={
            "neurological": "Positive Brudzinski's and Kernig's signs. Altered mental status."
        },
        test_results={
            "lumbar_puncture": "Cloudy CSF, high WBC (mainly neutrophils), low glucose, high protein.",
            "ct_head": "Normal, no increased ICP limits.",
        },
        correct_diagnosis="bacterial_meningitis",
        correct_treatment="iv_antibiotics",
    ),
    PatientScenario(
        id="P010",
        initial_symptoms="Patient presents with a red, swollen, intensely painful right big toe that started overnight.",
        vitals={"HR": "85", "BP": "130/80", "Temp": "99.2F", "RR": "16"},
        exam_findings={
            "extremity": "Right first metatarsophalangeal joint is erythematous, warm, and exquisitely tender to touch."
        },
        test_results={
            "joint_aspiration": "Negatively birefringent needle-shaped crystals. WBC 20,000.",
            "blood_uric_acid": "8.5 mg/dL",
        },
        correct_diagnosis="gout",
        correct_treatment="nsaids",
    ),
]
