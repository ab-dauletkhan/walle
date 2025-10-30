# 🔊 TTS Integration Summary

## What Was Added

The Rocket Assistant now has **Text-to-Speech (TTS)** capabilities using Mimic3!

### New Features

1. **Voice Output**: After the LLM responds, the robot will speak the response using Mimic3 TTS
2. **Configurable Voice**: Choose different voices (default: `en_UK/apope_low`)
3. **Optional TTS**: Can disable TTS with `--no-tts` flag for text-only mode
4. **Custom TTS Server**: Support for different Mimic3 server URLs

### New Dependencies

- **scipy** - For WAV file processing (automatically installed)
- **numpy** - For audio array handling (automatically installed)

### New Command-Line Arguments

```bash
--tts-url TTS_URL       # Mimic3 TTS server URL (default: http://localhost:59125)
--tts-voice TTS_VOICE   # TTS voice (default: en_UK/apope_low)
--no-tts                # Disable text-to-speech output
```

## How It Works

1. User says "hey rocket" + command
2. After 3 seconds of silence, command is sent to LLM
3. LLM generates text response
4. **[NEW]** Text response is sent to Mimic3 TTS API
5. **[NEW]** Received audio (WAV) is played through speakers
6. System returns to listening for "hey rocket"

## Example Flow

```
👂 Listening for 'hey rocket'...

🎤 Wake word detected! Listening for your command...

============================================================
🗣️  You: tell me a joke
============================================================
🤔 Thinking...

🤖 Rocket: Why did the robot go to therapy? Because it had too many bugs!
============================================================

🔊 Speaking...
[Robot speaks the joke using Mimic3 voice]

👂 Listening for 'hey rocket'...
```

## Technical Implementation

### TTS Request
```python
response = requests.get(
    "http://localhost:59125/api/tts",
    params={
        "text": "Hello world",
        "voice": "en_UK/apope_low"
    }
)
```

### Audio Playback
- Receives WAV audio from Mimic3
- Converts to numpy array using scipy.io.wavfile
- Plays using sounddevice.play()
- Waits for playback to complete before continuing

## Error Handling

- If Mimic3 is not running, shows warning but continues (text-only mode)
- All terminal output is preserved (text remains visible)
- TTS errors are shown with ⚠️ prefix but don't crash the program

## Testing

### Test TTS Service
```bash
# Check if Mimic3 is running
curl "http://localhost:59125/api/tts?voice=en_UK/apope_low&text=Hello" --head

# Test with the assistant (TTS enabled)
python example/rocket_assistant.py

# Test without TTS (text only)
python example/rocket_assistant.py --no-tts

# Test with different voice
python example/rocket_assistant.py --tts-voice en_US/vctk_low
```

## Files Modified

1. **example/rocket_assistant.py** - Added TTS functionality
   - New `speak_text()` method
   - Integration with Mimic3 API
   - Command-line arguments for TTS configuration
   
2. **ROCKET_ASSISTANT.md** - Updated documentation
   - Added TTS feature description
   - New command-line options
   - Troubleshooting for TTS issues

## Notes

- TTS is **enabled by default** if Mimic3 is running
- All text output **remains in terminal** (not removed)
- Works offline once Mimic3 is set up
- Compatible with any Mimic3 voice model

