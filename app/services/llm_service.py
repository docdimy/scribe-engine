"""
Medical LLM Analysis Service
Provides medical content analysis using OpenAI models
"""

import asyncio
import json
import re
from typing import Dict, Any, Optional
from openai import AsyncOpenAI

from app.config import settings
from app.models.responses import AnalysisResult
from app.core.logging import get_logger

logger = get_logger(__name__)


class LLMService:
    """Medical Large Language Model service for content analysis"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.default_model = settings.default_llm_model or "gpt-4.1-nano"
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        
        # Medical specialty prompts
        self.specialty_prompts = {
            # General Medicine
            "general": "Provide comprehensive general medical analysis.",
            "internal": "Focus on internal medicine, systemic conditions, and comprehensive care.",
            "family": "Focus on family medicine, primary care, and preventive health.",
            
            # Internal Medicine Subspecialties
            "cardiology": "Focus on cardiovascular symptoms, heart conditions, and cardiac treatments.",
            "gastroenterology": "Focus on digestive system disorders, GI symptoms, and hepatic conditions.",
            "pulmonology": "Focus on respiratory conditions, lung diseases, and breathing disorders.",
            "nephrology": "Focus on kidney diseases, renal function, and electrolyte disorders.",
            "endocrinology": "Focus on hormonal disorders, diabetes, thyroid conditions, and metabolic diseases.",
            "rheumatology": "Focus on autoimmune diseases, joint disorders, and inflammatory conditions.",
            "hematology": "Focus on blood disorders, bleeding tendencies, and hematologic malignancies.",
            "oncology": "Focus on cancer symptoms, malignancies, and oncologic treatments.",
            "infectious": "Focus on infectious diseases, antimicrobial therapy, and tropical medicine.",
            "geriatrics": "Focus on age-related conditions, elderly care, and geriatric syndromes.",
            
            # Surgical Specialties
            "surgery": "Focus on surgical conditions, procedures, and perioperative care.",
            "trauma": "Focus on trauma surgery, emergency procedures, and injury management.",
            "orthopedics": "Focus on musculoskeletal conditions, bone/joint disorders, and orthopedic procedures.",
            "neurosurgery": "Focus on neurosurgical conditions, brain/spine surgery, and neurological procedures.",
            "cardiac_surgery": "Focus on cardiac surgical conditions, heart procedures, and thoracic surgery.",
            "vascular": "Focus on vascular conditions, arterial/venous disorders, and vascular procedures.",
            "plastic": "Focus on reconstructive surgery, aesthetic procedures, and wound healing.",
            
            # Organ-Specific Specialties
            "urology": "Focus on urological conditions, kidney/bladder disorders, and male reproductive health.",
            "gynecology": "Focus on women's health, gynecological conditions, and reproductive disorders.",
            "obstetrics": "Focus on pregnancy, childbirth, and maternal health conditions.",
            "ophthalmology": "Focus on eye conditions, vision disorders, and ocular treatments.",
            "ent": "Focus on ear, nose, throat conditions, head/neck disorders, and ENT procedures.",
            "dermatology": "Focus on skin conditions, dermatological symptoms, and skin cancer screening.",
            
            # Specialized Medicine
            "neurology": "Focus on neurological symptoms, brain/nerve conditions, and neurological treatments.",
            "psychiatry": "Focus on mental health symptoms, psychological conditions, and psychiatric treatments.",
            "pediatrics": "Focus on child-specific symptoms, pediatric conditions, and age-appropriate treatments.",
            "emergency": "Focus on urgent symptoms, emergency conditions, and immediate interventions.",
            "anesthesiology": "Focus on perioperative care, pain management, and anesthetic considerations.",
            "radiology": "Focus on imaging findings, radiological interpretations, and diagnostic procedures.",
            "pathology": "Focus on laboratory findings, histopathological results, and diagnostic testing.",
            "rehabilitation": "Focus on functional recovery, physical therapy, and rehabilitation medicine.",
            "pain": "Focus on chronic pain conditions, pain management strategies, and palliative care.",
            
            # Specialized Care
            "intensive": "Focus on critical care conditions, ICU management, and life-threatening situations.",
            "palliative": "Focus on end-of-life care, symptom management, and comfort measures.",
            "occupational": "Focus on work-related health issues, occupational diseases, and workplace safety.",
            "sports": "Focus on sports injuries, athletic performance, and exercise-related conditions."
        }
    
    async def analyze_medical_content(
        self,
        transcript: str,
        specialty: Optional[str] = None,
        conversation_type: str = "consultation",
        model: str = None
    ) -> AnalysisResult:
        """
        Analyze medical transcript content using LLM
        
        Args:
            transcript: The medical conversation transcript
            specialty: Medical specialty context
            conversation_type: Type of medical conversation
            model: LLM model to use (defaults to gpt-4.1-nano)
            
        Returns:
            AnalysisResult with structured medical analysis
        """
        
        if not model:
            model = self.default_model
        
        try:
            # Generate medical analysis prompt
            prompt = self._generate_medical_prompt(
                transcript, specialty, conversation_type
            )
            
            logger.info(f"Starting LLM analysis with model: {model}")
            
            # Call OpenAI API with retry logic
            response = await self._call_openai_with_retry(
                prompt=prompt,
                model=model,
                max_retries=3
            )
            
            # Parse structured response
            analysis = self._parse_medical_analysis(response)
            
            logger.info(f"LLM analysis completed successfully")
            
            return analysis
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            raise
    
    def _generate_medical_prompt(
        self,
        transcript: str,
        specialty: Optional[str],
        conversation_type: str
    ) -> str:
        """Generate optimized medical analysis prompt"""
        
        specialty_context = ""
        if specialty and specialty in self.specialty_prompts:
            specialty_context = f"\n\nSpecialty Context: {self.specialty_prompts[specialty]}"
        
        conversation_context = {
            "consultation": "This is a medical consultation between a healthcare provider and patient.",
            "discharge": "This is a hospital discharge conversation with care instructions.",
            "emergency": "This is an emergency medical assessment.",
            "notes": "These are medical notes or observations.",
            "informed_consent": "This is a medical education/informed consent conversation explaining procedures, risks, and treatment options to the patient."
        }.get(conversation_type, "This is a medical conversation.")
        
        prompt = f"""You are a medical AI assistant analyzing a healthcare conversation. 
{conversation_context}{specialty_context}

Please analyze the following medical transcript and provide a structured analysis:

TRANSCRIPT:
{transcript}

Please provide your analysis in the following JSON format:
{{
    "summary": "Brief clinical summary (2-3 sentences)",
    "chief_complaint": "Primary reason for visit/concern",
    "diagnosis": "Clinical assessment/diagnosis (if determinable)",
    "symptoms": ["list", "of", "symptoms", "mentioned"],
    "treatment": "Treatment plan/recommendations",
    "medication": "Medications mentioned or prescribed",
    "follow_up": "Follow-up care instructions",
    "risk_factors": ["identified", "risk", "factors"],
    "clinical_notes": "Additional clinical observations",
    "confidence_level": "high/medium/low based on clarity of information"
}}

Guidelines:
- Be conservative in diagnosis unless clearly stated
- Use medical terminology appropriately
- Note if information is incomplete or unclear
- Focus on clinically relevant information
- Maintain patient confidentiality principles

Provide ONLY the JSON response, no additional text."""
        
        return prompt
    
    async def _call_openai_with_retry(
        self,
        prompt: str,
        model: str,
        max_retries: int = 3
    ) -> str:
        """Call OpenAI API with retry logic"""
        
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a medical AI assistant specialized in analyzing healthcare conversations and providing structured medical insights."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"}
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"OpenAI API attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"OpenAI API failed after {max_retries} attempts: {e}")
                    raise
    
    def _parse_medical_analysis(self, response_content: str) -> AnalysisResult:
        """Parse and validate LLM response into structured format"""
        
        try:
            # Parse JSON response
            analysis_data = json.loads(response_content)
            
            # Validate and extract fields with defaults
            summary = analysis_data.get("summary", "Analysis not available")
            chief_complaint = analysis_data.get("chief_complaint")
            diagnosis = analysis_data.get("diagnosis")
            symptoms = analysis_data.get("symptoms", [])
            treatment = analysis_data.get("treatment")
            medication = analysis_data.get("medication")
            follow_up = analysis_data.get("follow_up")
            risk_factors = analysis_data.get("risk_factors", [])
            clinical_notes = analysis_data.get("clinical_notes")
            confidence_level = analysis_data.get("confidence_level", "medium")
            
            # Ensure lists are properly formatted
            if isinstance(symptoms, str):
                symptoms = [symptoms]
            if isinstance(risk_factors, str):
                risk_factors = [risk_factors]
            
            return AnalysisResult(
                summary=summary,
                chief_complaint=chief_complaint,
                diagnosis=diagnosis,
                symptoms=symptoms,
                treatment=treatment,
                medication=medication,
                follow_up=follow_up,
                risk_factors=risk_factors,
                clinical_notes=clinical_notes,
                confidence_level=confidence_level
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response content: {response_content}")
            
            # Fallback: extract information using regex
            return self._fallback_parse(response_content)
        
        except Exception as e:
            logger.error(f"Error parsing medical analysis: {e}")
            
            # Return minimal analysis
            return AnalysisResult(
                summary="Analysis parsing failed",
                symptoms=[],
                risk_factors=[],
                confidence_level="low"
            )
    
    def _fallback_parse(self, content: str) -> AnalysisResult:
        """Fallback parsing when JSON parsing fails"""
        
        # Try to extract basic information using regex patterns
        summary_match = re.search(r'"summary":\s*"([^"]*)"', content)
        diagnosis_match = re.search(r'"diagnosis":\s*"([^"]*)"', content)
        treatment_match = re.search(r'"treatment":\s*"([^"]*)"', content)
        
        return AnalysisResult(
            summary=summary_match.group(1) if summary_match else "Summary extraction failed",
            diagnosis=diagnosis_match.group(1) if diagnosis_match else None,
            treatment=treatment_match.group(1) if treatment_match else None,
            symptoms=[],
            risk_factors=[],
            confidence_level="low"
        )
    
    async def get_available_models(self) -> Dict[str, Any]:
        """
        Get available LLM models for medical analysis
        
        Returns:
            Dictionary with model information
        """
        return {
            "current_model": self.default_model,
            "available_models": [
                "gpt-4.1-nano",
                "gpt-4o-mini", 
                "gpt-4o"
            ],
            "model_capabilities": {
                "gpt-4.1-nano": "Optimized model for medical analysis with nano-scale precision",
                "gpt-4o-mini": "Lightweight model for basic medical analysis",
                "gpt-4o": "Full-featured model for complex medical reasoning"
            }
        } 