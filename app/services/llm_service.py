"""
LLM Service for Medical Analysis
"""
import instructor
from openai import AsyncOpenAI
from app.config import settings
from app.models.responses import AnalysisResult
from app.core.logging import get_logger

logger = get_logger(__name__)

class LLMService:
    """Service for analyzing medical transcripts using LLMs."""

    def __init__(self):
        # Apply the patch to the OpenAI client
        # enables response_model keyword
        self.openai_client = instructor.patch(AsyncOpenAI(api_key=settings.openai_api_key))
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens

    async def analyze(
        self,
        transcript: str,
        model: str,
        specialty: str,
        conversation_type: str,
        output_language: str = "en"
    ) -> AnalysisResult:
        """
        Analyze the transcript with a Large Language Model
        """
        logger.info(f"Starting LLM analysis with model: {model}, specialty: {specialty}, output lang: {output_language}")
        
        system_prompt = self._build_system_prompt(specialty, conversation_type, output_language)
        
        try:
            # Use the patched client with response_model
            analysis = await self.openai_client.chat.completions.create(
                model=model,
                response_model=AnalysisResult,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcript}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            logger.info("LLM analysis completed successfully.")
            return analysis
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}", exc_info=True)
            raise

    def _build_system_prompt(
        self, 
        specialty: str, 
        conversation_type: str, 
        output_language: str
    ) -> str:
        """Builds the dynamic system prompt for the LLM"""
        
        language_map = {
            "en": "English", "de": "German", "fr": "French", "es": "Spanish",
            "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "sv": "Swedish",
            "da": "Danish", "no": "Norwegian", "fi": "Finnish"
        }
        language_name = language_map.get(output_language.lower(), "English")

        prompt = f"""
You are a highly skilled medical assistant specializing in '{specialty}'. 
Your task is to analyze the following medical transcript from a '{conversation_type}'.
Provide a structured analysis of the conversation.

The final output, including all text fields, MUST be in {language_name}.

Your analysis should include:
- summary: A concise summary of the key points, findings, and patient complaints.
- diagnosis: The primary diagnosis or suspected conditions.
- treatment: The recommended treatment plan.
- medication: Any prescribed or mentioned medications.
- follow_up: Recommendations for future appointments or actions.
- specialty_notes: Any specific notes relevant to the specialty of '{specialty}'.
- icd10_codes: A list of relevant ICD-10 codes based on the diagnosis.
"""
        return prompt

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