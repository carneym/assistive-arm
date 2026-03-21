# SO-101 Assistive Arm — Voice Pipeline

Voice control pipeline for the SO-101 assistive robotic arm.
Designed for Jetson Orin Nano with ReSpeaker USB Mic Array v2.0.

## Architecture

```
Mic → WakeWord (Porcupine/OpenWakeWord) → VAD → Whisper STT → Intent Parser → Action
                                                                                 ↓
                                                                          TTS Feedback
```

## Quick Start (Jetson / Linux PC with mic)

```bash
# 1. Clone or copy this directory
# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) For Porcupine wake word — get free access key at:
#    https://console.picovoice.ai/
#    Then set: export PORCUPINE_ACCESS_KEY="your-key-here"

# 5. Run the pipeline
python -m voice_pipeline.main

# 6. Or run in test mode (no mic needed, uses simulated audio)
python -m voice_pipeline.main --test
```

## Project Structure

```
voice_pipeline/
├── main.py                 # Entry point — orchestrates the pipeline
├── core/
│   ├── __init__.py
│   ├── audio_capture.py    # Mic input (live) or file input (test)
│   ├── wake_word.py        # Wake word detection (Porcupine or keyword fallback)
│   ├── vad.py              # Voice Activity Detection (Silero VAD)
│   ├── stt.py              # Speech-to-text (Whisper via whisper.cpp or openai-whisper)
│   ├── intent_parser.py    # Command parsing (rule-based + optional LLM)
│   └── tts.py              # Text-to-speech feedback (Piper or pyttsx3 fallback)
├── config/
│   ├── __init__.py
│   └── settings.py         # All tuneable parameters in one place
├── tests/
│   ├── __init__.py
│   ├── test_intent_parser.py
│   └── test_pipeline_integration.py
└── audio_samples/           # Drop .wav files here for testing
```

## Configuration

All settings are in `voice_pipeline/config/settings.py`. Key parameters:

- `WAKE_WORD` — trigger phrase (default: "hey arm")
- `WHISPER_MODEL` — model size: "tiny", "base", "small" (default: "small")
- `SAMPLE_RATE` — audio sample rate (default: 16000)
- `CONFIRMATION_REQUIRED_TASKS` — tasks needing verbal "yes" before executing
- `INTENT_CONFIDENCE_THRESHOLD` — minimum confidence to execute without clarification

## Porting to Jetson

The code auto-detects the environment:
- **No mic?** → Falls back to file-based input or test mode
- **No GPU?** → Whisper runs on CPU (slower but works)
- **No Porcupine key?** → Falls back to keyword-based wake word detection

For best performance on Jetson, install `whisper-cpp-python` with CUDA support:
```bash
WHISPER_CUBLAS=1 pip install whisper-cpp-python
```

## Dependencies

See `requirements.txt`. Core deps:
- `openai-whisper` or `whisper-cpp-python` (STT)
- `pvporcupine` (wake word, optional — needs free API key)
- `sounddevice` + `numpy` (audio capture)
- `piper-tts` (TTS, optional — fallback to pyttsx3)
