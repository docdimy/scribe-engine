"""
FHIR R4 Models und Utilities
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


class FHIRValidationError(Exception):
    """FHIR Validierungsfehler"""
    pass


class FHIRCodeableConcept(BaseModel):
    """Vereinfachtes CodeableConcept für FHIR"""
    coding: Optional[List[Dict[str, str]]] = None
    text: Optional[str] = None


class FHIRReference(BaseModel):
    """FHIR Reference"""
    reference: str
    display: Optional[str] = None


class FHIRNarrative(BaseModel):
    """FHIR Narrative"""
    status: str = "generated"
    div: str


class FHIRMeta(BaseModel):
    """FHIR Meta Informationen"""
    version_id: Optional[str] = None
    last_updated: Optional[str] = None
    profile: Optional[List[str]] = None
    security: Optional[List[FHIRCodeableConcept]] = None
    tag: Optional[List[FHIRCodeableConcept]] = None


class FHIRExtension(BaseModel):
    """FHIR Extension"""
    url: str
    value: Optional[Any] = None  # valueString, valueBoolean, etc.


class FHIRIdentifier(BaseModel):
    """FHIR Identifier"""
    use: Optional[str] = None
    type: Optional[FHIRCodeableConcept] = None
    system: Optional[str] = None
    value: Optional[str] = None


class FHIRPeriod(BaseModel):
    """FHIR Period"""
    start: Optional[str] = None
    end: Optional[str] = None


class FHIRQuantity(BaseModel):
    """FHIR Quantity"""
    value: Optional[float] = None
    unit: Optional[str] = None
    system: Optional[str] = None
    code: Optional[str] = None


class FHIRValidator:
    """FHIR Validierungs-Utilities"""
    
    @staticmethod
    def validate_bundle(bundle_dict: Dict[str, Any]) -> bool:
        """Validiert ein FHIR Bundle"""
        
        required_fields = ["resourceType", "type", "entry"]
        
        for field in required_fields:
            if field not in bundle_dict:
                raise FHIRValidationError(f"Pflichtfeld '{field}' fehlt im Bundle")
        
        if bundle_dict["resourceType"] != "Bundle":
            raise FHIRValidationError("ResourceType muss 'Bundle' sein")
        
        # Validiere Entries
        entries = bundle_dict.get("entry", [])
        for i, entry in enumerate(entries):
            if "resource" not in entry:
                raise FHIRValidationError(f"Entry {i} hat keine 'resource'")
            
            resource = entry["resource"]
            if "resourceType" not in resource:
                raise FHIRValidationError(f"Resource in Entry {i} hat keinen 'resourceType'")
        
        return True
    
    @staticmethod
    def validate_composition(composition_dict: Dict[str, Any]) -> bool:
        """Validiert eine FHIR Composition"""
        
        required_fields = ["resourceType", "status", "type", "subject", "date", "author", "title"]
        
        for field in required_fields:
            if field not in composition_dict:
                raise FHIRValidationError(f"Pflichtfeld '{field}' fehlt in Composition")
        
        if composition_dict["resourceType"] != "Composition":
            raise FHIRValidationError("ResourceType muss 'Composition' sein")
        
        # Status Validation
        valid_statuses = ["preliminary", "final", "amended", "entered-in-error"]
        if composition_dict["status"] not in valid_statuses:
            raise FHIRValidationError(f"Ungültiger Status: {composition_dict['status']}")
        
        return True


class FHIRUtils:
    """FHIR Utility Funktionen"""
    
    @staticmethod
    def create_coding(system: str, code: str, display: str = None) -> Dict[str, str]:
        """Erstellt ein FHIR Coding"""
        coding = {
            "system": system,
            "code": code
        }
        if display:
            coding["display"] = display
        return coding
    
    @staticmethod
    def create_codeable_concept(text: str = None, codings: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Erstellt ein FHIR CodeableConcept"""
        concept = {}
        if codings:
            concept["coding"] = codings
        if text:
            concept["text"] = text
        return concept
    
    @staticmethod
    def create_reference(resource_type: str, resource_id: str, display: str = None) -> Dict[str, str]:
        """Erstellt eine FHIR Reference"""
        ref = {
            "reference": f"{resource_type}/{resource_id}"
        }
        if display:
            ref["display"] = display
        return ref
    
    @staticmethod
    def create_narrative(content: str) -> Dict[str, str]:
        """Erstellt ein FHIR Narrative"""
        return {
            "status": "generated",
            "div": f'<div xmlns="http://www.w3.org/1999/xhtml">{content}</div>'
        }
    
    @staticmethod
    def format_fhir_datetime(dt: datetime = None) -> str:
        """Formatiert ein Datetime für FHIR"""
        if dt is None:
            dt = datetime.utcnow()
        return dt.isoformat() + "Z"
    
    @staticmethod
    def generate_fhir_id() -> str:
        """Generiert eine FHIR-konforme ID"""
        import uuid
        return str(uuid.uuid4())


class MedicalCoding:
    """Medizinische Codes und Standards"""
    
    # LOINC Codes für medizinische Dokumente
    LOINC_CONSULTATION_NOTE = "11488-4"
    LOINC_DISCHARGE_SUMMARY = "18842-5"
    LOINC_EMERGENCY_NOTE = "34111-5"
    LOINC_PROGRESS_NOTE = "11506-3"
    
    # SNOMED CT Codes (Beispiele)
    SNOMED_CONSULTATION = "11429006"
    SNOMED_DIAGNOSIS = "408408009"
    SNOMED_TREATMENT = "276239002"
    
    # ICD-10 System URI
    ICD10_SYSTEM = "http://hl7.org/fhir/sid/icd-10"
    
    # LOINC System URI
    LOINC_SYSTEM = "http://loinc.org"
    
    # SNOMED CT System URI
    SNOMED_SYSTEM = "http://snomed.info/sct"
    
    @classmethod
    def get_document_type_coding(cls, conversation_type: str) -> Dict[str, str]:
        """Gibt passenden LOINC Code für Dokumententyp zurück"""
        
        type_mapping = {
            "consultation": cls.LOINC_CONSULTATION_NOTE,
            "discharge": cls.LOINC_DISCHARGE_SUMMARY,
            "emergency": cls.LOINC_EMERGENCY_NOTE,
            "notes": cls.LOINC_PROGRESS_NOTE
        }
        
        code = type_mapping.get(conversation_type, cls.LOINC_CONSULTATION_NOTE)
        
        return FHIRUtils.create_coding(
            system=cls.LOINC_SYSTEM,
            code=code,
            display="Medical consultation note"
        ) 