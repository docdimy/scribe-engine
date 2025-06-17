# 🏥 Scribe Engine - Medical Audio Transcription & Analysis

A scalable FastAPI-based microservice for processing audio data, STT integration and LLM-based medical analysis with FHIR R4 compliance.

## ✨ Features

- **🎙️ Audio Transcription** with OpenAI Whisper and AssemblyAI
- **🗣️ Speaker Diarization** for multi-person conversations
- **🤖 LLM Analysis** of medical consultations
- **🔗 FHIR R4 Integration** for standards-compliant outputs
- **🔐 Security** with API-Key/JWT authentication
- **📊 Monitoring** with Prometheus metrics
- **🌍 Multi-language** (DE, EN, FR, ES, IT, PT, NL, SV, DA, NO, FI)
- **⚡ Rate Limiting** for API protection
- **📋 Audit Logging** for compliance

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend PWA  │───▶│  Scribe Engine  │───▶│ External APIs   │
│                 │    │                 │    │                 │
│ • Upload Audio  │    │ • Validation    │    │ • OpenAI        │
│ • View Results  │    │ • Transcription │    │ • AssemblyAI    │
│ • Export FHIR   │    │ • Analysis      │    │                 │
└─────────────────┘    │ • FHIR Export   │    └─────────────────┘
                       │                 │
                       │ • Security      │    ┌─────────────────┐
                       │ • Monitoring    │───▶│ Infrastructure  │
                       │ • Logging       │    │                 │
                       └─────────────────┘    │ • Redis         │
                                             │ • Prometheus    │
                                             └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- OpenAI API Key
- AssemblyAI API Key

### 1. Clone Repository

```bash
git clone <repository-url>
cd scribe-engine
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env and set your API keys
```

### 3. Start with Docker Compose

```bash
docker-compose up -d
```

### 4. Test API

```bash
# Health Check
curl http://localhost:3001/health

# Audio Transcription (Example)
curl -X POST "http://localhost:3001/v1/transcribe" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: audio/mpeg" \
  --data-binary @sample-audio.mp3
```

## 📋 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/transcribe` | Audio transcription and analysis |
| `GET` | `/health` | Health check |
| `GET` | `/ready` | Readiness check |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/docs` | OpenAPI documentation (development only) |

### Main Endpoint: `/v1/transcribe`

**Parameters:**
- `diarization` (bool): Enable speaker diarization
- `specialty` (string): Medical specialty
- `conversation_type` (string): Type of consultation
- `output_format` (enum): json/xml/fhir
- `language` (string): ISO 639-1 code or "auto"
- `model` (enum): LLM model for analysis

**Request:**
```bash
curl -X POST "http://localhost:3001/v1/transcribe?diarization=true&specialty=cardiology&output_format=fhir" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: audio/mpeg" \
  --data-binary @consultation.mp3
```

**Response:**
```json
{
  "request_id": "abc123",
  "timestamp": "2024-01-15T10:30:00Z",
  "transcript": {
    "full_text": "Patient describes chest pain...",
    "segments": [...],
    "language_detected": "en",
    "confidence": 0.95,
    "duration": 180.5
  },
  "analysis": {
    "summary": "Patient with acute chest pain...",
    "diagnosis": "Suspected angina pectoris",
    "treatment": "Rest, sublingual nitro",
    "medication": "Aspirin 100mg daily",
    "follow_up": "Check-up in 1 week",
    "icd10_codes": ["I20.9"]
  },
  "fhir_bundle": {...},
  "processing_time_ms": 1500
}
```

## 🔧 Configuration

### Environment Variables

```env
# API Configuration
API_SECRET_KEY=your-secret-key
OPENAI_API_KEY=your-openai-key
ASSEMBLYAI_API_KEY=your-assemblyai-key

# Audio Processing
MAX_AUDIO_DURATION=600
MAX_FILE_SIZE_MB=50

# Rate Limiting
RATE_LIMIT_REQUESTS=10
RATE_LIMIT_WINDOW=60

# CORS for PWA
CORS_ORIGINS=https://your-pwa-domain.com
```

### Supported Audio Formats

- MP3 (`audio/mpeg`)
- WAV (`audio/wav`)
- MP4/M4A (`audio/mp4`, `audio/m4a`)
- OGG (`audio/ogg`)

### Languages

- German (de)
- English (en)
- French (fr)
- Spanish (es)
- Italian (it)
- Portuguese (pt)
- Dutch (nl)
- Swedish (sv)
- Danish (da)
- Norwegian (no)
- Finnish (fi)
- Auto-detection (auto)

## 🏥 FHIR R4 Integration

The service generates fully FHIR R4-compliant bundles with:

- **Composition**: Main consultation document
- **Patient**: Anonymized patient resource
- **Practitioner**: Anonymized physician resource
- **Encounter**: Consultation event
- **Media**: Audio transcript
- **Condition**: Diagnoses
- **MedicationStatement**: Medications
- **CarePlan**: Treatment plan

### FHIR Bundle Example

```json
{
  "resourceType": "Bundle",
  "type": "document",
  "entry": [
    {
      "resource": {
        "resourceType": "Composition",
        "status": "final",
        "type": {
          "coding": [{
            "system": "http://loinc.org",
            "code": "11488-4",
            "display": "Consultation note"
          }]
        },
        ...
      }
    },
    ...
  ]
}
```

## 🔒 Security

### Authentication

```bash
# API Key in header
curl -H "Authorization: Bearer your-api-key" ...

# Or JWT Token
curl -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." ...
```

### Data Protection

- **Encryption-at-Rest** for temporary data
- **Automatic deletion** after processing
- **Audit logging** of all API access
- **GDPR compliant** with data flow documentation

## 📊 Monitoring

### Prometheus Metrics

```
# HTTP Requests
http_requests_total{method="POST", endpoint="/v1/transcribe", status="200"}

# Request Duration
http_request_duration_seconds

# Audio Processing Duration
audio_processing_duration_seconds
```

### Health Checks

```bash
# Basic health check
curl http://localhost:3001/health

# Detailed readiness check
curl http://localhost:3001/ready
```

## 🧪 Testing

```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/

# All tests
python -m pytest
```

## 📦 Deployment

### Production Docker

```dockerfile
# Multi-stage build for smaller images
# Non-root user for security
# Health checks for container orchestration
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: scribe-engine
spec:
  replicas: 3
  selector:
    matchLabels:
      app: scribe-engine
  template:
    metadata:
      labels:
        app: scribe-engine
    spec:
      containers:
      - name: scribe-engine
        image: scribe-engine:latest
        ports:
        - containerPort: 3001
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-key
```

## 🐛 Troubleshooting

### Common Issues

**Audio validation failed:**
```
HTTP 415: Audio format not supported
→ Check Content-Type header and file format
```

**Rate limit reached:**
```
HTTP 429: Too many requests
→ Wait 60 seconds or increase rate limit
```

**FHIR validation failed:**
```
HTTP 500: Invalid FHIR bundle
→ Check logs for validation details
```

### View Logs

```bash
# Docker Compose
docker-compose logs -f scribe-engine

# Docker
docker logs -f scribe-engine

# Structured logs (JSON)
docker logs scribe-engine | jq .
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🔗 Links

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FHIR R4 Specification](https://hl7.org/fhir/R4/)
- [OpenAI API](https://platform.openai.com/docs)
- [AssemblyAI API](https://www.assemblyai.com/docs)

---

**🏥 Scribe Engine** - Transforming medical audio into structured, actionable insights. 