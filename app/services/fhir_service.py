"""
FHIR R4 Service for medical data conversion
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from fhir.resources.R4B.bundle import Bundle
from fhir.resources.R4B.composition import Composition
from fhir.resources.R4B.patient import Patient
from fhir.resources.R4B.practitioner import Practitioner
from fhir.resources.R4B.encounter import Encounter
from fhir.resources.R4B.observation import Observation
from fhir.resources.R4B.condition import Condition
from fhir.resources.R4B.medicationstatement import MedicationStatement
from fhir.resources.R4B.careplan import CarePlan
from fhir.resources.R4B.media import Media
from fhir.resources.R4B.codeableconcept import CodeableConcept
from fhir.resources.R4B.coding import Coding
from fhir.resources.R4B.reference import Reference
from fhir.resources.R4B.narrative import Narrative

from app.models.responses import TranscriptionResult, AnalysisResult
from app.core.logging import get_logger
from app.config import settings
import base64
import mimetypes

logger = get_logger(__name__)


class FHIRService:
    """FHIR R4 conversion for medical data"""
    
    def __init__(self):
        self.base_url = settings.api_base_url if hasattr(settings, 'api_base_url') else "https://api.numediq.de"
    
    async def create_fhir_bundle(
        self,
        transcript: TranscriptionResult,
        analysis: AnalysisResult,
        request_id: str,
        specialty: Optional[str] = None,
        conversation_type: str = "consultation"
    ) -> Dict[str, Any]:
        """
        Create a FHIR R4 Bundle from transcript and analysis
        
        Args:
            transcript: Transcription result
            analysis: LLM analysis result  
            request_id: Request ID
            specialty: Medical specialty
            conversation_type: Type of consultation
            
        Returns:
            FHIR Bundle as Dictionary
        """
        
        try:
            # FHIR-compliant ID by replacing invalid characters
            fhir_bundle_id = request_id.replace("_", "-")

            # Create bundle
            bundle = Bundle(
                id=fhir_bundle_id,
                type="document",
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
            
            entries = []
            
            # 1. Patient (placeholder)
            patient = self._create_patient_placeholder()
            entries.append({
                "fullUrl": f"{self.base_url}/Patient/{patient.id}",
                "resource": patient.dict()
            })
            
            # 2. Practitioner (placeholder)
            practitioner = self._create_practitioner_placeholder(specialty)
            entries.append({
                "fullUrl": f"{self.base_url}/Practitioner/{practitioner.id}",
                "resource": practitioner.dict()
            })
            
            # 3. Encounter (consultation)
            encounter = self._create_encounter(conversation_type, patient.id, practitioner.id)
            entries.append({
                "fullUrl": f"{self.base_url}/Encounter/{encounter.id}",
                "resource": encounter.dict()
            })
            
            # Link resources in Composition
            composition = self._create_composition(
                transcript=transcript, 
                analysis=analysis,
                patient_ref=Reference(reference=f"Patient/{patient.id}"),
                practitioner_ref=Reference(reference=f"Practitioner/{practitioner.id}"),
                encounter_ref=Reference(reference=f"Encounter/{encounter.id}"),
                specialty=specialty, 
                conversation_type=conversation_type
            )
            # Add Composition as the first entry
            entries.insert(0, {
                "fullUrl": f"{self.base_url}/Composition/{composition.id}",
                "resource": composition.dict()
            })
            
            # 4. Condition (diagnosis)
            if analysis.diagnosis:
                condition = self._create_condition(analysis.diagnosis, patient.id, encounter.id)
                composition.section[0].entry.append(Reference(reference=f"Condition/{condition.id}"))
                entries.append({
                    "fullUrl": f"{self.base_url}/Condition/{condition.id}",
                    "resource": condition.dict()
                })
            
            # 5. MedicationStatement (medication)
            if analysis.medication:
                medication_stmt = self._create_medication_statement(
                    analysis.medication, patient.id, encounter.id
                )
                composition.section[0].entry.append(Reference(reference=f"MedicationStatement/{medication_stmt.id}"))
                entries.append({
                    "fullUrl": f"{self.base_url}/MedicationStatement/{medication_stmt.id}",
                    "resource": medication_stmt.dict()
                })
            
            # 6. CarePlan (treatment plan)
            if analysis.treatment or analysis.follow_up:
                care_plan = self._create_care_plan(
                    analysis.treatment, analysis.follow_up, patient.id, encounter.id
                )
                composition.section[0].entry.append(Reference(reference=f"CarePlan/{care_plan.id}"))
                entries.append({
                    "fullUrl": f"{self.base_url}/CarePlan/{care_plan.id}",
                    "resource": care_plan.dict()
                })
            
            # Assemble bundle
            bundle.entry = entries
            
            logger.info(f"FHIR Bundle created with {len(entries)} resources")
            
            return bundle.dict()
            
        except Exception as e:
            logger.error(f"FHIR Bundle creation failed: {e}", exc_info=True)
            raise
    
    def _create_composition(
        self,
        transcript: TranscriptionResult,
        analysis: AnalysisResult,
        patient_ref: Reference,
        practitioner_ref: Reference,
        encounter_ref: Reference,
        specialty: Optional[str],
        conversation_type: str
    ) -> Composition:
        """Create the Composition resource"""
        
        # Create narrative HTML
        narrative_text = f"""
        <div xmlns="http://www.w3.org/1999/xhtml">
            <h1>Medical Consultation: {conversation_type.title()}</h1>
            <p><strong>Specialty:</strong> {specialty or 'N/A'}</p>
            <hr/>
            <h2>Summary</h2>
            <p>{analysis.summary}</p>
            
            {f'<h2>Diagnosis</h2><p>{analysis.diagnosis}</p>' if analysis.diagnosis else ''}
            {f'<h2>Treatment</h2><p>{analysis.treatment}</p>' if analysis.treatment else ''}
            {f'<h2>Medication</h2><p>{analysis.medication}</p>' if analysis.medication else ''}
            {f'<h2>Follow-up</h2><p>{analysis.follow_up}</p>' if analysis.follow_up else ''}
            
            <h2>Full Transcript</h2>
            <p><em>(Duration: {transcript.duration or 'Unknown'}s, Language: {transcript.language_detected or 'Unknown'})</em></p>
            <pre>{transcript.full_text}</pre>
        </div>
        """
        
        return Composition(
            id=str(uuid.uuid4()),
            status="final",
            type=CodeableConcept(
                coding=[Coding(
                    system="http://loinc.org",
                    code="11488-4",
                    display="Consultation note"
                )]
            ),
            subject=patient_ref,
            encounter=encounter_ref,
            date=datetime.utcnow().isoformat() + "Z",
            author=[practitioner_ref],
            title=f"Medical Consultation - {conversation_type.title()}",
            text=Narrative(
                status="generated",
                div=narrative_text
            ),
            section=[{
                "title": "Clinical Findings and Plan",
                "entry": []
            }]
        )
    
    def _create_patient_placeholder(self) -> Patient:
        """Create a placeholder Patient resource"""
        return Patient(
            id="patient-placeholder",
            active=True,
            name=[{
                "use": "usual",
                "text": "Patient (Anonymized)"
            }]
        )
    
    def _create_practitioner_placeholder(self, specialty: Optional[str]) -> Practitioner:
        """Create a placeholder Practitioner resource"""
        
        qualification = []
        if specialty:
            qualification.append({
                "code": {
                    "text": specialty.title()
                }
            })
        
        return Practitioner(
            id="practitioner-placeholder",
            active=True,
            name=[{
                "use": "official",
                "text": "Physician (Anonymized)"
            }],
            qualification=qualification if qualification else None
        )
    
    def _create_encounter(self, conversation_type: str, patient_id: str, practitioner_id: str) -> Encounter:
        """Create an Encounter resource"""
        
        class_code = {
            "consultation": "AMB",  # ambulatory
            "discharge": "IMP",     # inpatient
            "emergency": "EMER",    # emergency
            "notes": "AMB"
        }.get(conversation_type, "AMB")

        encounter_data = {
            "id": str(uuid.uuid4()),
            "status": "finished",
            "class": Coding(
                system="http://terminology.hl7.org/CodeSystem/v3-ActCode",
                code=class_code
            ),
            "subject": Reference(reference=f"Patient/{patient_id}"),
            "participant": [{
                "individual": Reference(reference=f"Practitioner/{practitioner_id}")
            }],
            "period": {
                "start": datetime.utcnow().isoformat() + "Z"
            }
        }
        
        return Encounter(**encounter_data)
    
    def _create_condition(self, diagnosis: str, patient_id: str, encounter_id: str) -> Condition:
        """Create a Condition resource for the diagnosis"""
        
        return Condition(
            id=str(uuid.uuid4()),
            clinicalStatus=CodeableConcept(coding=[Coding(
                system="http://terminology.hl7.org/CodeSystem/condition-clinical",
                code="active"
            )]),
            verificationStatus=CodeableConcept(coding=[Coding(
                system="http://terminology.hl7.org/CodeSystem/condition-ver-status",
                code="provisional" # Based on LLM analysis
            )]),
            category=[CodeableConcept(coding=[Coding(
                system="http://terminology.hl7.org/CodeSystem/condition-category",
                code="encounter-diagnosis"
            )])],
            code=CodeableConcept(text=diagnosis),
            subject=Reference(reference=f"Patient/{patient_id}"),
            encounter=Reference(reference=f"Encounter/{encounter_id}")
        )
    
    def _create_medication_statement(self, medication: str, patient_id: str, encounter_id: str) -> MedicationStatement:
        """Create a MedicationStatement resource"""
        return MedicationStatement(
            id=str(uuid.uuid4()),
            status="active",
            medicationCodeableConcept=CodeableConcept(text=medication),
            subject=Reference(reference=f"Patient/{patient_id}"),
            context=Reference(reference=f"Encounter/{encounter_id}"),
            dateAsserted=datetime.utcnow().isoformat() + "Z"
        )
    
    def _create_care_plan(
        self, 
        treatment: Optional[str], 
        follow_up: Optional[str], 
        patient_id: str, 
        encounter_id: str
    ) -> CarePlan:
        """Create a CarePlan for treatment and follow-up care"""
        
        description = []
        if treatment:
            description.append(f"Treatment: {treatment}")
        if follow_up:
            description.append(f"Follow-up: {follow_up}")
        
        return CarePlan(
            id=str(uuid.uuid4()),
            status="active",
            intent="plan",
            subject=Reference(reference=f"Patient/{patient_id}"),
            encounter=Reference(reference=f"Encounter/{encounter_id}"),
            created=datetime.utcnow().isoformat() + "Z",
            description=" | ".join(description)
        ) 