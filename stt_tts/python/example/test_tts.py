#!/usr/bin/env python3
"""
Simple test script for Mimic3 TTS integration
"""

import requests
import sounddevice as sd
import numpy as np
import io
from scipy.io import wavfile

def test_tts(text="Hello, this is a test of the text to speech system.", 
             voice="en_UK/apope_low",
             tts_url="http://localhost:59125"):
    """Test TTS by speaking a text"""
    
    print(f"🔊 Testing TTS...")
    print(f"Text: {text}")
    print(f"Voice: {voice}")
    print(f"URL: {tts_url}")
    print()
    
    try:
        # Request TTS audio
        print("📡 Requesting audio from Mimic3...")
        response = requests.get(
            f"{tts_url}/api/tts",
            params={
                "text": text,
                "voice": voice
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Received audio from server")
            
            # Load WAV audio
            audio_data = io.BytesIO(response.content)
            sample_rate, audio_array = wavfile.read(audio_data)
            
            # Convert to float if needed
            if audio_array.dtype == np.int16:
                audio_array = audio_array.astype(np.float32) / 32768.0
            elif audio_array.dtype == np.int32:
                audio_array = audio_array.astype(np.float32) / 2147483648.0
            
            print(f"🎵 Audio info: {sample_rate}Hz, {len(audio_array)} samples")
            print("🔊 Playing audio...")
            
            # Play audio
            sd.play(audio_array, sample_rate)
            sd.wait()
            
            print("✅ Playback complete!")
            return True
        else:
            print(f"❌ Error: Server returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Cannot connect to Mimic3 at {tts_url}")
        print("   Make sure Mimic3 is running!")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Mimic3 TTS")
    parser.add_argument(
        "-t", "--text",
        type=str,
        default="Hello, this is a test of the text to speech system.",
        help="Text to speak"
    )
    parser.add_argument(
        "-v", "--voice",
        type=str,
        default="en_UK/apope_low",
        help="Voice to use"
    )
    parser.add_argument(
        "-u", "--url",
        type=str,
        default="http://localhost:59125",
        help="Mimic3 server URL"
    )
    
    args = parser.parse_args()
    
    success = test_tts(text=args.text, voice=args.voice, tts_url=args.url)
    exit(0 if success else 1)

