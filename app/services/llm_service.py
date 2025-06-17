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
    
    async def analyze(
        self,
        transcript: str,
        model: str,
        specialty: str,
        conversation_type: str
    ) -> AnalysisResult:
        """
        Analyze medical transcript content using an LLM.
        """
        try:
            prompt = self._generate_medical_prompt(
                transcript, specialty, conversation_type
            )
            
            logger.info(f"Starting LLM analysis with model: {model}")
            
            response_content = await self._call_openai_with_retry(
                prompt=prompt,
                model=model,
                max_retries=settings.max_retries
            )
            
            analysis = self._parse_medical_analysis(response_content)
            
            logger.info("LLM analysis completed successfully.")
            return analysis
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}", exc_info=True)
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
    "summary": "Brief clinical summary of the conversation (2-3 sentences).",
    "diagnosis": "Most likely clinical diagnosis or differential diagnoses. Be conservative if information is unclear.",
    "treatment": "Recommended treatment plan or actions taken.",
    "medication": "Medications mentioned, prescribed, or adjusted.",
    "follow_up": "Instructions for follow-up care or appointments.",
    "specialty_notes": "Key observations and notes relevant to the specified medical specialty.",
    "icd10_codes": ["ICD-10-CM code(s) relevant to the diagnosis, if identifiable."]
}}

Guidelines:
- If a field is not applicable or information is missing, use an empty string or an empty list.
- Provide ONLY the JSON response, with no additional text or explanations.
- The analysis should be objective and based solely on the provided transcript.
"""
        
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
            if not response_content:
                logger.warning("LLM response was empty. Returning empty analysis.")
                return AnalysisResult(summary="No analysis available.")

            analysis_data = json.loads(response_content)
            
            # Use .get() with default values to prevent KeyErrors
            return AnalysisResult(
                summary=analysis_data.get("summary", "Summary not available."),
                diagnosis=analysis_data.get("diagnosis"),
                treatment=analysis_data.get("treatment"),
                medication=analysis_data.get("medication"),
                follow_up=analysis_data.get("follow_up"),
                specialty_notes=analysis_data.get("specialty_notes"),
                icd10_codes=analysis_data.get("icd10_codes", [])
            )
        
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from LLM response.", exc_info=True)
            # Try a fallback if JSON is malformed but contains content
            return self._fallback_parse(response_content)
        except Exception as e:
            logger.error(f"Error parsing LLM analysis: {e}", exc_info=True)
            raise
    
    def _fallback_parse(self, content: str) -> AnalysisResult:
        """A simple fallback parser if JSON is invalid."""
        return AnalysisResult(
            summary="Could not parse structured analysis from the model.",
            specialty_notes=content # Put the raw response here for debugging
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