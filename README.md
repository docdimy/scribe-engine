<p align="center">
  <img src="app/static/scribe-engine.svg" alt="Scribe Engine Logo" width="400">
</p>
<h1 align="center">Medical Audio Transcription & Analysis</h1>

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

The service includes a simple web interface for easy testing. Once the container is running, open your browser and navigate to:

**➡️ <http://localhost:3001/>**

You can use this interface to record audio directly in the browser and send it to the API.

Alternatively, you can use `curl` to check if the service is running:
```bash
curl http://localhost:3001/health
```

A successful response should look like this:
```json
{"status":"healthy","timestamp":"...","version":"1.0.0","uptime_seconds":...}
```

## 📋 API Documentation

The API is documented via OpenAPI and can be explored at `http://localhost:3001/docs` when running in development mode.

For manual tests, it is recommended to use the **built-in web interface at <http://localhost:3001/>**. The following `curl` examples are for automated scripting.

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

The API is designed to be flexible and accepts a wide range of common audio formats. For best results regarding upload speed and processing efficiency, the **`Ogg/Opus`** format is recommended, as it offers excellent quality at a low file size.

The following MIME types and their corresponding file extensions are supported:

- **`audio/ogg`** (Recommended, typically `.ogg` or `.opus`)
- `audio/webm` (Typically `.webm`)
- `audio/mpeg` (Typically `.mp3`)
- `audio/wav` (Typically `.wav`)
- `audio/mp4` (Typically `.mp4`, `.m4a`)

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

- **Encryption-at-Rest:** All audio data is encrypted using a strong symmetric key (`Fernet`) before it is temporarily written to disk. The data is only decrypted in memory for processing and is immediately deleted afterwards.
- **Encryption-in-Transit:** Communication with all external APIs (OpenAI, AssemblyAI) is performed exclusively over HTTPS. For client-server communication, it is strongly recommended to use a reverse proxy that handles TLS termination in production.
- **Hardened HTTP Headers:** The application sends security headers like `Strict-Transport-Security` (HSTS), `Content-Security-Policy` (CSP), and `X-Content-Type-Options` to protect against common web vulnerabilities like clickjacking and cross-site scripting (XSS).
- **Automatic Deletion:** All temporary files and in-memory data related to a request are securely deleted immediately after processing is complete.
- **Audit Logging** of all API access
- **GDPR compliant** with data flow documentation

## 📊 Monitoring

The service exposes key performance and usage metrics in the Prometheus format. This allows for easy integration with monitoring dashboards (e.g., Grafana) and alerting systems.

### Accessing Metrics

The metrics are available at the `/metrics` endpoint.

```bash
curl http://localhost:3001/metrics
```

### Key Metrics

- `http_requests_total`: A counter for all HTTP requests, labeled by method, endpoint, and status code. Helps to monitor traffic and error rates.
- `http_request_duration_seconds`: A histogram measuring the latency of HTTP requests. Useful for performance analysis and SLO tracking.
- `audio_processing_duration_seconds`: A histogram specifically measuring the time taken for audio validation and saving.

### Health Checks

To ensure the service is running and ready to accept traffic, two health check endpoints are provided:
- `/health`: A simple check to confirm the service process is running.
- `/ready`: A more detailed check that can be extended to verify connections to downstream dependencies (like OpenAI, Redis, etc.).

## 📝 Project Plan

Further development steps, feature ideas and the overall roadmap are documented in the `projectplan.md` file in this repository.

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