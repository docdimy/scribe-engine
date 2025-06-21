"""
LLM Service for Medical Analysis
"""
import instructor
from openai import AsyncOpenAI
from typing import Dict, Any
from app.config import settings, ModelName
from app.models.responses import AnalysisResult
from app.core.logging import get_logger
import httpx
from openai.types.chat import ChatCompletion
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import assemblyai as aai
import json
import asyncio

logger = get_logger(__name__)

# Define which exceptions should trigger a retry
retryable_exceptions = (
    httpx.TimeoutException,
    # Add other transient exceptions if needed
)

# Define retry condition for OpenAI server errors (5xx)
def is_server_error(exception):
    """Return True if the exception is an OpenAI 5xx error"""
    from openai import APIStatusError
    return isinstance(exception, APIStatusError) and exception.status_code >= 500

class LLMService:
    """Service for analyzing medical transcripts using LLMs."""

    def __init__(self):
        # Apply the patch to the OpenAI client
        # enables response_model keyword
        self.openai_client = instructor.patch(AsyncOpenAI(api_key=settings.openai_api_key))
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.default_model = settings.default_llm_model

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        retry=(retry_if_exception_type(retryable_exceptions) | retry_if_exception_type(is_server_error))
    )
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

    async def label_speakers_with_lemur(
        self, 
        transcript: aai.Transcript, 
        specialty: str, 
        conversation_type: str
    ) -> Dict[str, str]:
        """
        Uses LeMUR to identify and label speakers as 'Provider' or 'Client'.
        Returns a mapping dictionary, e.g., {'A': 'Provider', 'B': 'Client'}.
        """
        if not transcript.utterances:
            logger.warning("No utterances found in transcript, cannot label speakers.")
            return {}

        # Extract the unique speaker labels from the transcript to give LeMUR clear instructions
        speaker_labels = sorted(list(set(utt.speaker for utt in transcript.utterances if utt.speaker)))
        if not speaker_labels:
            logger.warning("Could not find any speaker labels in the utterances.")
            return {}
        
        speaker_labels_str = ", ".join(f"'{label}'" for label in speaker_labels)

        prompt = (
            "You are an expert in analyzing medical conversations.\n"
            f"The following transcript contains speakers identified by the labels: {speaker_labels_str}.\n"
            "Based on the content of the conversation, please determine which speaker label corresponds to the 'Client' (e.g., patient, client) and which corresponds to the 'Provider' (e.g., doctor, practitioner, clinician, medical professional, nurse, therapist).\n"
            "Consider who is describing symptoms versus who is asking questions and providing medical advice.\n"
            "Your response MUST be a single, valid JSON object that maps each speaker label to one of the two roles: 'Client' or 'Provider'.\n"
            "For example: {\"A\": \"Provider\", \"B\": \"Client\"}"
        )
        
        context = f"This is a medical conversation within the specialty of '{specialty}'. The conversation type is a '{conversation_type}'."

        try:
            logger.info(f"Sending transcript (ID: {transcript.id}) to LeMUR for speaker labeling with context: {context}")
            # Call LeMUR via the transcript object. This needs to be run in a thread
            # as the SDK call itself is synchronous.
            lemur_response = await asyncio.to_thread(
                transcript.lemur.task,
                prompt=prompt,
                context=context,
                final_model="anthropic/claude-3-haiku",
                temperature=0.1, # Low temperature for factual, deterministic output
            )

            # Parse the JSON response from LeMUR
            # The response might be a plain string, so we need to be careful
            response_text = lemur_response.response.strip()
            logger.debug(f"Received raw response from LeMUR: {response_text}")
            
            speaker_map = json.loads(response_text)
            
            # Basic validation of the returned map
            if not isinstance(speaker_map, dict) or not all(isinstance(v, str) for v in speaker_map.values()):
                logger.error(f"LeMUR returned an invalid speaker map format: {speaker_map}")
                return {}

            logger.info(f"LeMUR successfully identified speaker roles: {speaker_map}")
            return speaker_map

        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from LeMUR response. Raw response: '{lemur_response.response}'", exc_info=True)
            return {}
        except Exception as e:
            logger.error(f"An error occurred while using LeMUR: {e}", exc_info=True)
            return {}

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
        Get available OpenAI models for analysis
        """
        # This is a placeholder; in a real-world scenario, you might
        # query the OpenAI API or have a more dynamic way of getting models.
        return {
            "default": self.default_model,
            "supported": [model.value for model in ModelName]
        } 