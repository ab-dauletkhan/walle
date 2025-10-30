# 🚀 Rocket Voice Assistant

A voice-activated AI assistant that combines **Vosk Speech Recognition** with **Ollama LLM** for natural voice interactions.

## Features

- 🎤 **Wake word activation**: Say "hey rocket" to start
- 🗣️ **Natural speech recognition**: Speak your question/command
- 🤖 **AI responses**: Powered by Ollama (qwen3:4b)
- 💬 **Concise responses**: System prompt keeps answers short (3-4 sentences)
- 🔊 **Text-to-Speech**: Robot speaks responses using Mimic3
- ⏱️ **Smart timeout**: 3-second silence after you start speaking
- 🔄 **Continuous listening**: Ready for next command after each response

## Prerequisites

1. **Vosk** (already installed)
2. **Ollama** running with qwen3:4b model
   ```bash
   # Start Ollama server
   ollama serve
   
   # Pull the model (if not already installed)
   ollama pull qwen3:4b
   ```
3. **Mimic3 TTS** (Text-to-Speech) server running on port 59125
   - The robot will speak responses using this service
   - Must be running before starting the assistant

## Usage

### Basic Usage
```bash
cd /Users/edigeakimali/PycharmProjects/wall-e/stt/vosk-api/python
source .venv/bin/activate
python example/rocket_assistant.py
```

### How It Works

1. **Wait for the prompt**: `👂 Listening for 'hey rocket'...`
2. **Say the wake word**: "hey rocket"
3. **Speak your command**: e.g., "what is the weather like today?"
4. **Wait 3 seconds** (silence) for the system to process
5. **Get AI response**: The assistant will respond via Ollama
6. **Repeat**: System goes back to listening for "hey rocket"

### Example Interaction

```
👂 Listening for 'hey rocket'...

🎤 Wake word detected! Listening for your command...

============================================================
🗣️  You: what is the capital of France
============================================================
🤔 Thinking...

🤖 Rocket: The capital of France is Paris.
============================================================

🔊 Speaking...
[Robot speaks the response]

👂 Listening for 'hey rocket'...
```

## Command-Line Options

```bash
# Use a different wake word
python example/rocket_assistant.py -w "hello assistant"

# Change silence timeout to 2 seconds
python example/rocket_assistant.py -t 2.0

# Use a different Ollama model
python example/rocket_assistant.py -m llama2

# Disable TTS (text only, no voice)
python example/rocket_assistant.py --no-tts

# Use different TTS voice
python example/rocket_assistant.py --tts-voice en_US/vctk_low

# Custom TTS server URL
python example/rocket_assistant.py --tts-url http://localhost:8080

# Custom system prompt (for different response style)
python example/rocket_assistant.py --system-prompt "You are a pirate assistant. Answer like a pirate in 2-3 sentences."

# List available microphones
python example/rocket_assistant.py --list-devices

# Use specific microphone (device ID)
python example/rocket_assistant.py -d 1

# Full help
python example/rocket_assistant.py --help
```

## Troubleshooting

### "Cannot connect to Ollama"
- Make sure Ollama is running: `ollama serve`
- Check if running: `curl http://localhost:11434/api/tags`

### "TTS Error: Cannot connect to Mimic3"
- Make sure Mimic3 TTS is running on port 59125
- Check if running: `curl http://localhost:59125/api/voices`
- You can disable TTS with `--no-tts` flag if needed

### Wake word not detected
- Speak clearly: "hey rocket"
- Make sure your microphone is working
- Check microphone with: `python example/rocket_assistant.py --list-devices`

### Model not found
- List available models: `ollama list`
- Pull the model: `ollama pull qwen3:0.6b`

### No response after speaking
- Wait for 3 seconds of silence
- The system needs silence to know you're done speaking
- Adjust timeout with `-t` flag if needed

## Technical Details

- **Wake Word**: "hey rocket" (exact match, case-insensitive)
- **Speech Recognition**: Vosk (offline, English model)
- **Sentence Detection**: Vosk's natural pause detection + smart 3s timeout (only starts after you begin speaking)
- **LLM**: Ollama API (qwen3:4b by default)
- **System Prompt**: Configured for concise 3-4 sentence responses
- **TTS**: Mimic3 (en_UK/apope_low voice by default)
- **Audio Format**: 16kHz, mono, 16-bit PCM

## Tips for Best Results

1. **Speak clearly** and at a normal pace
2. **Pause briefly** after saying "hey rocket" before starting your command
3. **Speak your full command** - the system waits for you to start speaking
4. **Wait 3 seconds of silence** after finishing to signal you're done
5. **Avoid background noise** for better recognition
6. **Use complete sentences** for better LLM understanding

## Exit

Press `Ctrl+C` to stop the assistant at any time.

