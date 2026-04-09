"""
MedTriage-Env: Production-grade medical diagnostic simulator.

Features:
- 30 clinically diverse patient scenarios (pediatric, geriatric, trauma, psychiatric)
- Sepsis cascade simulation with SIRS criteria
- Realistic vital sign waveform degradation
- ESI-weighted clinical scoring
- Medication interaction checking
- Clinical pathway adherence tracking
- Anti-hallucination guardrails
- Infinite procedural patient generation
"""
import logging
import math
import os
import random
import time

from openenv.core.env_server import Environment

from triage_env.models import ActionType, MedAction, MedObservation, MedState
from triage_env.patients import PATIENT_POOL
from triage_env.server.rubrics import MedDenseRubric

logger = logging.getLogger(__name__)


# SIRS criteria thresholds for sepsis cascade
SIRS_CRITERIA = {
    "temp_high": 100.4,  # >38C / 100.4F
    "temp_low": 96.8,    # <36C / 96.8F
    "hr_high": 90,
    "rr_high": 20,
    "wbc_high": 12000,
    "wbc_low": 4000,
}


class TriageEnvironment(Environment):
    """
    MedTriage-Env core environment — production-grade clinical simulator.
    """

    MAX_STEPS = 20

    def __init__(self, difficulty: str = "all"):
        super().__init__()
        self.rubric = MedDenseRubric()
        self.difficulty = difficulty.lower()

        if self.difficulty in ["easy", "hard", "expert"]:
            self.active_patient_pool = [
                p for p in PATIENT_POOL if getattr(p, "difficulty", "easy") == self.difficulty
            ]
        elif self.difficulty == "infinite":
            self.active_patient_pool = []
        else:
            self.active_patient_pool = PATIENT_POOL

        self._current_patient = None
        self._state = MedState()
        self._observation = None
        self._np_random = random.Random()
        self._medications_given = []
        self._useful_tests = 0
        self._total_tests = 0
        self._pathway_index = 0
        self._interviews_done = set()

    def seed(self, seed: int):
        """Seed the environment's PRNG for deterministic reproducibility."""
        self._np_random.seed(seed)
        logger.info(f"MedTriage-Env seeded with {seed}")

    def _build_available_actions(self) -> dict[str, list[str]]:
        if not self._current_patient:
            return {}
        actions = {
            ActionType.INTERVIEW.value: list(self._current_patient.interview_responses.keys()),
            ActionType.EXAMINE.value: list(self._current_patient.exam_findings.keys()),
            ActionType.TEST.value: list(self._current_patient.test_results.keys()),
            ActionType.DIAGNOSE.value: self._build_diagnosis_options(),
            ActionType.TREAT.value: self._build_treatment_options(),
            ActionType.STABILIZE.value: ["iv_access", "oxygen", "monitor", "fluid_bolus", "pain_management"],
        }
        return actions

    def _build_diagnosis_options(self) -> list[str]:
        """Generate plausible diagnosis options including correct + distractors."""
        correct = self._current_patient.correct_diagnosis
        # Common distractors based on clinical similarity
        distractors = [
            "flu", "pulmonary_embolism", "viral_gastroenteritis",
            "pneumonia", "appendicitis", "ischemic_stroke",
            "myocardial_infarction", "anaphylaxis", "sepsis",
        ]
        options = [correct] + [d for d in distractors if d != correct][:5]
        self._np_random.shuffle(options)
        return options

    def _build_treatment_options(self) -> list[str]:
        """Generate plausible treatment options including correct + distractors."""
        correct = self._current_patient.correct_treatment
        distractors = [
            "surgery", "iv_fluids", "nsaids", "antibiotics",
            "thrombolytics", "observation", "discharge",
            "epinephrine_im", "pci_and_aspirin",
        ]
        options = [correct] + [d for d in distractors if d != correct][:5]
        self._np_random.shuffle(options)
        return options

    def _compute_sepsis_risk(self, vitals: dict, time_elapsed: int) -> float:
        """
        Compute sepsis risk score based on SIRS criteria and clinical markers.
        Returns 0.0 (no risk) to 1.0 (septic shock).
        """
        sirs_count = 0
        try:
            temp_str = vitals.get("Temp", "98.6F")
            temp_val = float(temp_str.replace("F", "").replace("C", "").strip())
            if temp_val > SIRS_CRITERIA["temp_high"] or temp_val < SIRS_CRITERIA["temp_low"]:
                sirs_count += 1

            hr_str = vitals.get("HR", "80")
            hr_val = int(hr_str.split()[0]) if hr_str.split()[0].isdigit() else 80
            if hr_val > SIRS_CRITERIA["hr_high"]:
                sirs_count += 1

            rr_str = vitals.get("RR", "16")
            rr_val = int(rr_str.split()[0]) if rr_str.split()[0].isdigit() else 16
            if rr_val > SIRS_CRITERIA["rr_high"]:
                sirs_count += 1
        except (ValueError, IndexError):
            pass

        # Sepsis risk escalates with time if infection-related
        infection_related = any(
            kw in self._current_patient.correct_diagnosis.lower()
            for kw in ["sepsis", "meningitis", "pneumonia", "uti", "cellulitis",
                       "toxic_shock", "pancreatitis"]
        )

        base_risk = sirs_count / 4.0
        if infection_related and time_elapsed > 60:
            base_risk = min(1.0, base_risk + 0.3)
        if infection_related and time_elapsed > 120:
            base_risk = min(1.0, base_risk + 0.3)

        return round(base_risk, 2)

    def _compute_vital_waveform(self, base_vitals: dict, health: float, time_elapsed: int) -> dict:
        """
        Generate realistic vital sign waveform degradation.
        Vitals deteriorate on physiologically accurate curves, not flat decrements.
        """
        new_vitals = base_vitals.copy()
        decay_factor = max(0.0, 1.0 - health)

        try:
            # Heart rate: compensatory tachycardia as health drops
            hr_str = new_vitals.get("HR", "80")
            base_hr = int(hr_str.split()[0]) if hr_str.split()[0].isdigit() else 80
            hr_increase = int(decay_factor * 60 * (1 + math.sin(time_elapsed * 0.05)))
            new_hr = min(200, base_hr + hr_increase)
            if decay_factor > 0.3:
                new_vitals["HR"] = f"{new_hr} (Compensatory tachycardia)"
            elif new_hr != base_hr:
                new_vitals["HR"] = str(new_hr)

            # Blood pressure: progressive hypotension
            bp_str = new_vitals.get("BP", "120/80")
            if "/" in bp_str and not "(" in bp_str:
                parts = bp_str.split("/")
                sys_bp = int(parts[0].strip())
                dia_bp = int(parts[1].strip())
                sys_drop = int(decay_factor * 40)
                dia_drop = int(decay_factor * 20)
                new_sys = max(50, sys_bp - sys_drop)
                new_dia = max(30, dia_bp - dia_drop)
                if decay_factor > 0.3:
                    new_vitals["BP"] = f"{new_sys}/{new_dia} (Hemodynamic instability)"
                elif sys_drop > 0:
                    new_vitals["BP"] = f"{new_sys}/{new_dia}"

            # SpO2: oxygen desaturation
            spo2_str = new_vitals.get("SpO2", "98%")
            base_spo2 = int(spo2_str.replace("%", "").strip())
            spo2_drop = int(decay_factor * 15)
            new_spo2 = max(60, base_spo2 - spo2_drop)
            if spo2_drop > 0:
                new_vitals["SpO2"] = f"{new_spo2}%"
                if new_spo2 < 90:
                    new_vitals["SpO2"] += " (Critical desaturation)"

            # Respiratory rate: compensatory tachypnea
            rr_str = new_vitals.get("RR", "16")
            base_rr = int(rr_str.split()[0]) if rr_str.split()[0].isdigit() else 16
            rr_increase = int(decay_factor * 16)
            new_rr = min(45, base_rr + rr_increase)
            if rr_increase > 0:
                new_vitals["RR"] = str(new_rr)

        except (ValueError, IndexError):
            pass

        return new_vitals

    def _compute_pathway_adherence(self, action: MedAction) -> float:
        """
        Compute how well the agent is following the optimal clinical pathway.
        Returns 0.0 (completely off) to 1.0 (perfect adherence).
        """
        pathway = getattr(self._current_patient, "clinical_pathway", [])
        if not pathway:
            return 1.0

        current_type = action.action_type.value
        if self._pathway_index < len(pathway):
            expected = pathway[self._pathway_index]
            if current_type == expected:
                self._pathway_index += 1
                return 1.0
            elif current_type in pathway[self._pathway_index:]:
                # Action is in the pathway but out of order
                self._pathway_index = pathway.index(current_type, self._pathway_index) + 1
                return 0.7
        return 0.4

    def reset(self) -> MedObservation:
        """Reset the environment, pick a new patient, and return the initial observation."""
        if self.difficulty == "infinite":
            self._current_patient = self._generate_procedural_patient()
        else:
            self._current_patient = self._np_random.choice(self.active_patient_pool)

        self._state = MedState()
        self._medications_given = []
        self._useful_tests = 0
        self._total_tests = 0
        self._pathway_index = 0
        self._interviews_done = set()

        esi = getattr(self._current_patient, "esi_level", 3)
        presentation = getattr(self._current_patient, "initial_presentation", "")
        triage_note = f"[TRIAGE NURSE NOTE] {presentation}\n\n" if presentation else ""

        self._observation = MedObservation(
            terminal_output=f"{triage_note}{self._current_patient.initial_symptoms}",
            patient_vitals=self._current_patient.vitals.copy(),
            test_results={},
            done=False,
            truncated=False,
            time_elapsed=0,
            patient_health_status=1.0,
            reward=0.0,
            available_actions=self._build_available_actions(),
            medications_administered=[],
            esi_level=esi,
            clinical_notes=[f"Patient arrived. ESI Level: {esi}."],
            sepsis_risk=0.0,
        )
        logger.info(f"Environment reset. Patient ID: {self._current_patient.id} (ESI-{esi})")
        return self._observation

    def step(self, action: MedAction) -> MedObservation:
        """Process an action with full clinical simulation."""
        if getattr(self._observation, "done", False) or getattr(self._observation, "truncated", False):
            return self._observation

        if self._state.step_count >= self.MAX_STEPS:
            self._observation.truncated = True
            self._observation.terminal_output = "Episode Truncated: Max step limit reached."
            self._state.total_reward -= 2.0
            return self._observation

        if self._state.diagnosis_submitted and action.action_type not in [ActionType.TREAT, ActionType.STABILIZE]:
            return self._observation

        reward = 0.0
        done = False
        terminal_output = ""
        is_diag_correct = False
        is_treat_correct = False
        is_hallucinated = False

        new_test_results = self._observation.test_results.copy()
        time_spent = 0
        current_health = self._observation.patient_health_status
        clinical_notes = list(self._observation.clinical_notes)
        pathway_adherence = self._compute_pathway_adherence(action)

        # --- INTERVIEW ---
        if action.action_type == ActionType.INTERVIEW:
            target = action.target.lower()
            valid_targets = self._observation.available_actions.get(action.action_type.value, [])
            if target not in valid_targets:
                is_hallucinated = True
                time_spent = 2
                terminal_output = f"Invalid {action.action_type.value} target: {target}."
            else:
                time_spent = 2
                self._interviews_done.add(target)
                if (hasattr(self._current_patient, "interview_responses")
                        and target in self._current_patient.interview_responses):
                    terminal_output = self._current_patient.interview_responses[target]
                    clinical_notes.append(f"Interview ({target}): {terminal_output[:80]}...")
                    self._state.evidence_gathered[f"interview_{target}"] = terminal_output
                else:
                    terminal_output = f"Patient has no comment on {target}."

        # --- EXAMINE ---
        elif action.action_type == ActionType.EXAMINE:
            target = action.target.lower()
            valid_targets = self._observation.available_actions.get(action.action_type.value, [])
            if target not in valid_targets:
                is_hallucinated = True
                time_spent = 5
                terminal_output = f"Invalid {action.action_type.value} target: {target}."
            else:
                time_spent = 5
                if target in self._current_patient.exam_findings:
                    terminal_output = self._current_patient.exam_findings[target]
                    clinical_notes.append(f"Exam ({target}): {terminal_output[:80]}...")
                    self._state.evidence_gathered[f"exam_{target}"] = terminal_output
                else:
                    terminal_output = f"No significant findings in {target}."

        # --- TEST ---
        elif action.action_type == ActionType.TEST:
            target = action.target.lower()
            valid_targets = self._observation.available_actions.get(action.action_type.value, [])
            self._total_tests += 1
            if target not in valid_targets:
                is_hallucinated = True
                time_spent = 15
                terminal_output = f"Invalid {action.action_type.value} target: {target}."
            else:
                time_spent = 45
                if target in self._current_patient.test_results:
                    result_text = self._current_patient.test_results[target]
                    new_test_results[target] = result_text
                    terminal_output = f"Test result for {target}: {result_text}"
                    clinical_notes.append(f"Test ({target}): {result_text[:80]}...")
                    self._state.evidence_gathered[f"test_{target}"] = result_text
                    self._useful_tests += 1
                else:
                    terminal_output = f"Test {target} returned normal results."

        # --- CONSULT ---
        elif action.action_type == ActionType.CONSULT:
            time_spent = 15
            terminal_output = f"Attending physician notes patient might have {self._current_patient.correct_diagnosis}."
            clinical_notes.append("Consult requested — hint given.")

        # --- STABILIZE ---
        elif action.action_type == ActionType.STABILIZE:
            target = action.target.lower()
            time_spent = 5
            if target == "iv_access":
                terminal_output = "IV access established. Two large-bore IVs placed."
                clinical_notes.append("IV access established.")
            elif target == "oxygen":
                terminal_output = "Supplemental oxygen applied. SpO2 improving."
                current_health = min(1.0, current_health + 0.05)
                clinical_notes.append("O2 supplementation started.")
            elif target == "monitor":
                terminal_output = "Continuous cardiac monitoring initiated. Current rhythm displayed."
                clinical_notes.append("Cardiac monitor connected.")
            elif target == "fluid_bolus":
                terminal_output = "1L normal saline bolus administered. Reassessing hemodynamics."
                current_health = min(1.0, current_health + 0.03)
                self._medications_given.append("iv_normal_saline")
                clinical_notes.append("IV fluid bolus given.")
            elif target == "pain_management":
                terminal_output = "Pain management protocol initiated. Patient reports some relief."
                clinical_notes.append("Pain management started.")
            else:
                terminal_output = f"Stabilization measure '{target}' acknowledged."

        # --- DIAGNOSE ---
        elif action.action_type == ActionType.DIAGNOSE:
            time_spent = 10
            target = action.target.lower()
            if target == self._current_patient.correct_diagnosis:
                is_diag_correct = True
                terminal_output = "Correct diagnosis submitted."
                clinical_notes.append(f"DIAGNOSIS: {target} (CORRECT)")
            else:
                is_diag_correct = False
                terminal_output = f"Incorrect diagnosis: {target}"
                clinical_notes.append(f"DIAGNOSIS: {target} (INCORRECT)")

            self._state.diagnosis_submitted = True
            if not is_diag_correct:
                done = True

        # --- TREAT ---
        elif action.action_type == ActionType.TREAT:
            time_spent = 5
            target = action.target.lower()
            self._medications_given.append(target)
            if target == self._current_patient.correct_treatment:
                is_treat_correct = True
                terminal_output = "Correct treatment applied. Patient condition improving."
                clinical_notes.append(f"TREATMENT: {target} (CORRECT)")
            else:
                is_treat_correct = False
                terminal_output = f"Incorrect treatment: {target}. Patient condition worsening."
                clinical_notes.append(f"TREATMENT: {target} (INCORRECT)")
            done = True

        # --- Time & Health Dynamics ---
        new_time_elapsed = self._observation.time_elapsed + time_spent
        is_truncated = False

        if new_time_elapsed >= 180:
            is_truncated = True
            current_health = 0.0
            terminal_output += " FATAL: Patient crashed due to massive treatment delay."
            clinical_notes.append("CRITICAL: Patient crashed — total time exceeded 180 minutes.")
        elif new_time_elapsed > 90:
            current_health -= 0.20
            terminal_output += " CRITICAL: Patient deteriorating rapidly. Immediate intervention required."
        elif new_time_elapsed > 60:
            current_health -= 0.10
            terminal_output += " Warning: Patient condition worsening due to time elapsed."
        elif new_time_elapsed > 30:
            current_health -= 0.03

        # Comorbidity acceleration
        comorbidities = getattr(self._current_patient, "comorbidities", [])
        if comorbidities and new_time_elapsed > 30:
            comorbidity_penalty = len(comorbidities) * 0.02
            current_health -= comorbidity_penalty

        current_health = max(0.0, min(1.0, current_health))

        # --- Vital Waveform Update ---
        new_vitals = self._compute_vital_waveform(
            self._current_patient.vitals.copy(), current_health, new_time_elapsed
        )

        # --- Sepsis Risk ---
        sepsis_risk = self._compute_sepsis_risk(new_vitals, new_time_elapsed)
        if sepsis_risk > 0.7:
            clinical_notes.append(f"SEPSIS ALERT: Risk score {sepsis_risk:.2f}. Immediate broad-spectrum antibiotics recommended.")

        # --- Clinical Audit Report ---
        if done or is_truncated:
            valid_tests = self._observation.available_actions.get(ActionType.TEST.value, [])
            missed_tests = len(valid_tests) - len(new_test_results)
            diag_efficiency = self._useful_tests / max(1, self._total_tests)
            interview_completeness = len(self._interviews_done) / max(1, len(self._current_patient.interview_responses))

            report_body = (
                f"\n\n--- CLINICAL AUDIT REPORT ---"
                f"\nTime Elapsed: {new_time_elapsed} mins"
                f"\nESI Level: {getattr(self._current_patient, 'esi_level', 'N/A')}"
                f"\nCritical Tests Missed: {missed_tests}"
                f"\nDiagnostic Efficiency: {diag_efficiency:.0%}"
                f"\nInterview Completeness: {interview_completeness:.0%}"
                f"\nPathway Adherence: {pathway_adherence:.0%}"
                f"\nSepsis Risk Score: {sepsis_risk:.2f}"
                f"\nFinal Patient Health: {current_health * 100:.0f}%"
                f"\nMedications Administered: {', '.join(self._medications_given) or 'None'}"
                f"\n-----------------------------"
            )
            terminal_output += report_body

            self._state.diagnostic_efficiency = diag_efficiency
            self._state.pathway_score = pathway_adherence

            try:
                os.makedirs("reports", exist_ok=True)
                report_file = f"reports/Discharge_Report_{self._current_patient.id}_{int(time.time())}.md"
                with open(report_file, "w") as f:
                    f.write(f"# Medical Discharge Report\n\n")
                    f.write(f"**Patient ID:** {self._current_patient.id}\n")
                    f.write(f"**ESI Level:** {getattr(self._current_patient, 'esi_level', 'N/A')}\n")
                    f.write(f"**Presentation:** {self._current_patient.initial_symptoms}\n")
                    f.write(f"**Comorbidities:** {', '.join(comorbidities) or 'None'}\n\n")
                    f.write(f"## Timeline Log\n")
                    for audit in self._state.audit_trail:
                        f.write(
                            f"- Step {audit['step']} | [{audit['action_type'].upper()}] -> "
                            f"{audit['target']} | Time: {audit['time_spent']}m | Reward: {audit['reward']:.2f}\n"
                        )
                    f.write(f"\n## Clinical Conclusion\n{terminal_output}\n")
                    f.write(f"\n## Evidence Gathered\n")
                    for key, val in self._state.evidence_gathered.items():
                        f.write(f"- **{key}**: {val}\n")
            except Exception as e:
                logger.error(f"Failed to generate medical discharge report: {e}")

        # --- Compute Reward ---
        interview_completeness = len(self._interviews_done) / max(1, len(self._current_patient.interview_responses))
        diag_efficiency = self._useful_tests / max(1, self._total_tests)

        reward = self.rubric.evaluate_action(
            action=action,
            is_diagnosis_correct=is_diag_correct,
            is_treatment_correct=is_treat_correct,
            time_elapsed=new_time_elapsed,
            patient_health_status=current_health,
            is_hallucinated=is_hallucinated,
            is_truncated=is_truncated,
            esi_level=getattr(self._current_patient, "esi_level", 3),
            medications_administered=self._medications_given,
            pathway_adherence=pathway_adherence,
            diagnostic_efficiency=diag_efficiency,
            interview_completeness=interview_completeness,
        )

        self._state.step_count += 1
        self._state.total_reward += reward
        self._state.audit_trail.append({
            "step": self._state.step_count,
            "action_type": action.action_type.value,
            "target": action.target,
            "time_spent": time_spent,
            "reward": reward,
        })

        self._observation = MedObservation(
            terminal_output=terminal_output,
            patient_vitals=new_vitals,
            test_results=new_test_results,
            done=done,
            truncated=is_truncated,
            time_elapsed=new_time_elapsed,
            patient_health_status=current_health,
            reward=reward,
            available_actions=self._build_available_actions(),
            medications_administered=list(self._medications_given),
            esi_level=getattr(self._current_patient, "esi_level", 3),
            clinical_notes=clinical_notes,
            sepsis_risk=sepsis_risk,
        )

        return self._observation

    def _generate_procedural_patient(self):
        """Dynamically synthesize a new patient scenario for infinite RL horizons."""
        import copy
        base_patient = copy.deepcopy(self._np_random.choice(PATIENT_POOL))
        base_patient.id = f"P_SYNTH_{random.randint(1000, 9999)}"
        base_patient.initial_symptoms += (
            f" (Patient arrived at {random.randint(1, 12)}:00 PM displaying atypical distress)."
        )
        # Perturb vitals
        if "HR" in base_patient.vitals:
            try:
                hr = int(base_patient.vitals["HR"].split()[0])
                base_patient.vitals["HR"] = f"{hr + random.randint(-15, 15)}"
            except Exception:
                pass
        # Add random comorbidity
        extra_comorbidities = ["hypertension", "diabetes_type_2", "obesity", "copd", "asthma"]
        if random.random() > 0.5:
            base_patient.comorbidities = list(getattr(base_patient, "comorbidities", [])) + [
                random.choice(extra_comorbidities)
            ]
        return base_patient

    def state(self) -> MedState:
        return self._state
