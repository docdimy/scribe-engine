"""
FHIR R4 Service for medical data conversion
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from fhir.resources.bundle import Bundle
from fhir.resources.composition import Composition
from fhir.resources.patient import Patient
from fhir.resources.practitioner import Practitioner
from fhir.resources.encounter import Encounter
from fhir.resources.observation import Observation
from fhir.resources.condition import Condition
from fhir.resources.medicationstatement import MedicationStatement
from fhir.resources.careplan import CarePlan
from fhir.resources.media import Media
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.reference import Reference
from fhir.resources.narrative import Narrative

from app.models.responses import TranscriptionResult, AnalysisResult
from app.core.logging import get_logger
from app.config import settings

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
            # Create bundle
            bundle = Bundle(
                id=request_id,
                type="document",
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
            
            entries = []
            
            # 1. Composition (main document)
            composition = self._create_composition(
                transcript, analysis, request_id, specialty, conversation_type
            )
            entries.append({
                "fullUrl": f"{self.base_url}/Composition/{composition.id}",
                "resource": composition.dict()
            })
            
            # 2. Patient (placeholder)
            patient = self._create_patient_placeholder()
            entries.append({
                "fullUrl": f"{self.base_url}/Patient/{patient.id}",
                "resource": patient.dict()
            })
            
            # 3. Practitioner (placeholder)
            practitioner = self._create_practitioner_placeholder(specialty)
            entries.append({
                "fullUrl": f"{self.base_url}/Practitioner/{practitioner.id}",
                "resource": practitioner.dict()
            })
            
            # 4. Encounter (consultation)
            encounter = self._create_encounter(conversation_type, patient.id, practitioner.id)
            entries.append({
                "fullUrl": f"{self.base_url}/Encounter/{encounter.id}",
                "resource": encounter.dict()
            })
            
            # 5. Media (audio transcript)
            media = self._create_media_resource(transcript, encounter.id)
            entries.append({
                "fullUrl": f"{self.base_url}/Media/{media.id}",
                "resource": media.dict()
            })
            
            # 6. Condition (diagnosis)
            if analysis.diagnosis:
                condition = self._create_condition(analysis.diagnosis, patient.id, encounter.id)
                entries.append({
                    "fullUrl": f"{self.base_url}/Condition/{condition.id}",
                    "resource": condition.dict()
                })
            
            # 7. MedicationStatement (medication)
            if analysis.medication:
                medication_stmt = self._create_medication_statement(
                    analysis.medication, patient.id, encounter.id
                )
                entries.append({
                    "fullUrl": f"{self.base_url}/MedicationStatement/{medication_stmt.id}",
                    "resource": medication_stmt.dict()
                })
            
            # 8. CarePlan (treatment plan)
            if analysis.treatment or analysis.follow_up:
                care_plan = self._create_care_plan(
                    analysis.treatment, analysis.follow_up, patient.id, encounter.id
                )
                entries.append({
                    "fullUrl": f"{self.base_url}/CarePlan/{care_plan.id}",
                    "resource": care_plan.dict()
                })
            
            # Assemble bundle
            bundle.entry = entries
            bundle.total = len(entries)
            
            logger.info(f"FHIR Bundle created with {len(entries)} resources")
            
            return bundle.dict()
            
        except Exception as e:
            logger.error(f"FHIR Bundle creation failed: {e}")
            raise
    
    def _create_composition(
        self,
        transcript: TranscriptionResult,
        analysis: AnalysisResult,
        request_id: str,
        specialty: Optional[str],
        conversation_type: str
    ) -> Composition:
        """Create the Composition resource"""
        
        # Create narrative HTML
        narrative_text = f"""
        <div xmlns="http://www.w3.org/1999/xhtml">
            <h1>Medical Consultation</h1>
            <h2>Summary</h2>
            <p>{analysis.summary}</p>
            
            {f'<h2>Diagnosis</h2><p>{analysis.diagnosis}</p>' if analysis.diagnosis else ''}
            {f'<h2>Treatment</h2><p>{analysis.treatment}</p>' if analysis.treatment else ''}
            {f'<h2>Medication</h2><p>{analysis.medication}</p>' if analysis.medication else ''}
            {f'<h2>Follow-up</h2><p>{analysis.follow_up}</p>' if analysis.follow_up else ''}
            
            <h2>Transcript</h2>
            <p><em>Duration: {transcript.duration or 'Unknown'} seconds</em></p>
            <p><em>Language: {transcript.language_detected or 'Unknown'}</em></p>
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
            subject=Reference(reference="Patient/patient-placeholder"),
            date=datetime.utcnow().isoformat() + "Z",
            author=[Reference(reference="Practitioner/practitioner-placeholder")],
            title=f"Medical Consultation - {conversation_type.title()}",
            text=Narrative(
                status="generated",
                div=narrative_text
            )
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
        
        return Encounter(
            id=str(uuid.uuid4()),
            status="finished",
            class_=Coding(
                system="http://terminology.hl7.org/CodeSystem/v3-ActCode",
                code=class_code
            ),
            subject=Reference(reference=f"Patient/{patient_id}"),
            participant=[{
                "individual": Reference(reference=f"Practitioner/{practitioner_id}")
            }],
            period={
                "start": datetime.utcnow().isoformat() + "Z"
            }
        )
    
    def _create_media_resource(self, transcript: TranscriptionResult, encounter_id: str) -> Media:
        """Create a Media resource for the transcript"""
        
        return Media(
            id=str(uuid.uuid4()),
            status="completed",
            type=CodeableConcept(
                coding=[Coding(
                    system="http://terminology.hl7.org/CodeSystem/media-type",
                    code="audio",
                    display="Audio"
                )]
            ),
            subject=Reference(reference="Patient/patient-placeholder"),
            encounter=Reference(reference=f"Encounter/{encounter_id}"),
            content={
                "contentType": "text/plain",
                "data": transcript.full_text.encode('utf-8').hex(),
                "title": "Audio Transcript"
            },
            note=[{
                "text": f"Automatically generated transcript. Language: {transcript.language_detected or 'Unknown'}"
            }]
        )
    
    def _create_condition(self, diagnosis: str, patient_id: str, encounter_id: str) -> Condition:
        """Create a Condition resource for the diagnosis"""
        
        return Condition(
            id=str(uuid.uuid4()),
            clinicalStatus=CodeableConcept(
                coding=[Coding(
                    system="http://terminology.hl7.org/CodeSystem/condition-clinical",
                    code="active"
                )]
            ),
            verificationStatus=CodeableConcept(
                coding=[Coding(
                    system="http://terminology.hl7.org/CodeSystem/condition-ver-status",
                    code="provisional"
                )]
            ),
            code=CodeableConcept(text=diagnosis),
            subject=Reference(reference=f"Patient/{patient_id}"),
            encounter=Reference(reference=f"Encounter/{encounter_id}"),
            recordedDate=datetime.utcnow().isoformat() + "Z"
        )
    
    def _create_medication_statement(self, medication: str, patient_id: str, encounter_id: str) -> MedicationStatement:
        """Create a MedicationStatement resource"""
        
        return MedicationStatement(
            id=str(uuid.uuid4()),
            status="recorded",
            medicationCodeableConcept=CodeableConcept(text=medication),
            subject=Reference(reference=f"Patient/{patient_id}"),
            context=Reference(reference=f"Encounter/{encounter_id}"),
            effectiveDateTime=datetime.utcnow().isoformat() + "Z"
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