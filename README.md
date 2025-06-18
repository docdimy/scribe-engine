# 🏥 Scribe Engine - Medical Audio Transcription & Analysis

A scalable FastAPI-based microservice for processing audio data, STT integration and LLM-based medical analysis with FHIR R4 compliance.

## ✨ Features

- **🎙️ Audio Transcription** with OpenAI (`gpt-4o-mini-transcribe`) and AssemblyAI (`assemblyai-universal`)
- **🗣️ Automatic STT Backend Selection** based on diarization requirement
- **🤖 Multi-lingual LLM Analysis** with selectable input and output languages
- **🔗 Flexible FHIR R4 Integration** with selectable bundle types (`document` or `transaction`)
- **🔐 Security** with API-Key/JWT authentication
- **📊 Monitoring** with Prometheus metrics
- **🌍 Multi-language Support** for transcription and analysis
- **⚡ Rate Limiting** for API protection
- **📋 Audit Logging** for compliance

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │───▶│  Scribe Engine  │───▶│ External APIs   │
│ (PWA, EMR, ...) │    │                 │    │                 │
│                 │    │ • Validation    │    │ • OpenAI        │
│                 │    │ • Transcription │    │ • AssemblyAI    │
│                 │    │ • Analysis      │    │                 │
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
- AssemblyAI API Key (only if diarization is needed)

### 1. Clone Repository

```bash
git clone <repository-url>
cd scribe-engine
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory and add your API keys.

```bash
# .env
API_SECRET_KEY="your-super-secret-key-for-jwt-and-authentication"
OPENAI_API_KEY="sk-..."
ASSEMBLYAI_API_KEY="..."
```

### 3. Build and Start with Docker Compose

For development, use the `docker-compose.development.yml` file.

```bash
# Use --build the first time or after changing requirements.txt
docker-compose -f docker-compose.development.yml up --build -d
```

To stop the service:
```bash
docker-compose -f docker-compose.development.yml down
```

### 4. Test API

Check if the service is running:

```bash
curl http://localhost:3001/health
```

A successful response should look like this:
```json
{"status":"healthy","timestamp":"...","version":"1.0.0","uptime_seconds":...}
```

## 📋 API Documentation

### Main Endpoint: `POST /v1/transcribe`

This is the primary endpoint for all transcription and analysis tasks. It accepts a `multipart/form-data` request containing the audio file and uses query parameters for configuration.

#### Query Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio_file` | file | **Required** | The audio file to be processed. |
| `model` | enum | `gpt-4.1-nano` | The LLM to use for analysis. Can be `gpt-4.1-nano`, `gpt-4o-mini`, or `gpt-4o`. |
| `diarization` | bool | `false` | If `true`, enables speaker separation. This automatically uses AssemblyAI as the STT backend. |
| `language` | string | `auto` | ISO 639-1 code for the **input** audio language. `auto` enables automatic detection. |
| `output_language` | string | detected language | ISO 639-1 code for the **output** analysis language. Defaults to the detected input language. |
| `output_format` | enum | `json` | Desired output format. Can be `json`, `xml`, or `fhir`. |
| `fhir_bundle_type` | enum | `document` | The type of FHIR bundle to generate. Can be `document` or `transaction`. **Only used if `output_format=fhir`**. |
| `specialty` | string | `general` | Medical specialty to tailor the analysis (e.g., `cardiology`). |
| `conversation_type`| string | `consultation` | The type of conversation for better contextual analysis. |

---

#### Request Examples

##### Example 1: Simple JSON Analysis (German)

Transcribe a German audio file and get the analysis back in German.

```bash
curl -X POST "http://localhost:3001/v1/transcribe?language=de" \
  -H "Authorization: Bearer your-api-key" \
  -F "audio_file=@/path/to/your/german_audio.mp3"
```

##### Example 2: FHIR Document for Cardiology (English Output)

Transcribe a German audio file but generate an English analysis and a FHIR `document` bundle for a cardiology context.

```bash
curl -X POST "http://localhost:3001/v1/transcribe?language=de&output_language=en&output_format=fhir&specialty=cardiology" \
  -H "Authorization: Bearer your-api-key" \
  -F "audio_file=@/path/to/your/german_audio.mp3"
```

##### Example 3: FHIR Transaction with Diarization

Transcribe an audio file with multiple speakers, generating a FHIR `transaction` bundle ready to be posted to an EMR/KIS.

```bash
curl -X POST "http://localhost:3001/v1/transcribe?diarization=true&output_format=fhir&fhir_bundle_type=transaction" \
  -H "Authorization: Bearer your-api-key" \
  -F "audio_file=@/path/to/your/multi-speaker-audio.mp3"
```
---

## 🏥 FHIR R4 Integration

The service can generate fully FHIR R4-compliant bundles. Depending on the `fhir_bundle_type` parameter, the structure serves different purposes.

### `document` Bundle (Default)
- **Type:** `document`
- **Purpose:** Creates a static, unchangeable clinical document, much like a PDF. Ideal for archiving or sharing a snapshot in time.
- **Structure:** The bundle is centered around a `Composition` resource that acts as a table of contents, linking to all other resources.

### `transaction` Bundle
- **Type:** `transaction`
- **Purpose:** Creates a set of instructions for a FHIR server. It's an "atomic" operation: either all resources are created successfully, or none are. Ideal for writing data into an EMR or clinical data repository.
- **Structure:** The bundle contains a list of resources, each with a `request` field (`POST`, `PUT`) telling the receiving server what to do.

**Generated Resources:**
- **Composition** (only in `document` bundles)
- **Patient** (anonymized placeholder)
- **Practitioner** (anonymized placeholder)
- **Encounter**
- **Condition** (from analysis)
- **MedicationStatement** (from analysis)
- **CarePlan** (from analysis)

## 🔧 Configuration Details

### Supported Audio Formats

- MP3 (`audio/mpeg`)
- WAV (`audio/wav`)
- MP4/M4A (`audio/mp4`)
- OGG (`audio/ogg`)

### Languages

Supports all languages provided by the backend services. For explicit use, provide the ISO 639-1 code (e.g., `de`, `en`, `es`).

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