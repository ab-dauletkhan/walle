#!/usr/bin/env python3
"""
Ollama Diagnostics - Check your Ollama installation and available models
Run this first if you have any issues with WALL-E
"""

import sys
import json
from openai import OpenAI

def check_ollama_connection():
    """Check if Ollama is running and accessible"""
    print("🔍 Checking Ollama connection...")
    try:
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        # Try a simple API call
        models = client.models.list()
        print("✅ Ollama is running and accessible!")
        return client, True
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure Ollama is installed: https://ollama.ai")
        print("   2. Start Ollama: 'ollama serve'")
        print("   3. Check if it's running on port 11434")
        return None, False


def list_available_models(client):
    """List all available models in Ollama"""
    print("\n📋 Available models in your Ollama:")
    try:
        models = client.models.list()
        if not models.data:
            print("   ⚠️  No models found!")
            print("\n💡 Install a model first:")
            print("   ollama pull qwen2.5:0.5b    # Smallest, fastest")
            print("   ollama pull qwen2.5:1.5b    # Balanced")
            print("   ollama pull qwen2.5:3b      # Better quality")
            return []
        
        model_names = []
        for i, model in enumerate(models.data, 1):
            model_name = model.id
            model_names.append(model_name)
            print(f"   {i}. {model_name}")
        
        return model_names
    except Exception as e:
        print(f"   ❌ Error listing models: {e}")
        return []


def find_best_model(model_names):
    """Find the best model for WALL-E"""
    print("\n🎯 Recommended models for WALL-E:")
    
    # Preferred models in order
    preferred = [
        "qwen3:0.6b",
        "qwen2.5:0.5b", 
        "qwen2.5:1.5b",
        "qwen2:0.5b",
        "qwen:0.5b",
        "llama3.2:1b",
        "llama3.2:3b",
    ]
    
    found_models = []
    for pref in preferred:
        if pref in model_names:
            found_models.append(pref)
    
    if found_models:
        print(f"   ✅ Found: {', '.join(found_models)}")
        print(f"\n   👉 Best choice: {found_models[0]}")
        return found_models[0]
    else:
        print("   ⚠️  No recommended models found")
        if model_names:
            print(f"\n   Available alternatives: {', '.join(model_names[:3])}")
            return model_names[0]
        return None


def test_model(client, model_name):
    """Test if a model works"""
    print(f"\n🧪 Testing model: {model_name}")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Say 'Hello, I am WALL-E!' in one short sentence."}
            ],
            max_tokens=50
        )
        
        reply = response.choices[0].message.content
        print(f"   ✅ Model works!")
        print(f"   Response: {reply[:100]}")
        return True
    except Exception as e:
        print(f"   ❌ Model test failed: {e}")
        return False


def show_fix_instructions(recommended_model):
    """Show how to fix walle_enhanced.py"""
    print("\n" + "=" * 70)
    print("🔧 How to fix walle_enhanced.py:")
    print("=" * 70)
    
    if recommended_model:
        print(f"\n1. Open walle_enhanced.py")
        print(f"2. Find the line: MODEL_NAME = \"...\"")
        print(f"3. Change it to: MODEL_NAME = \"{recommended_model}\"")
        print(f"\n   Or use the fixed version: walle_enhanced_fixed.py")
    else:
        print("\n⚠️  Please install a model first:")
        print("   ollama pull qwen2.5:0.5b")
        print("\n   Then run this script again!")


def main():
    print("=" * 70)
    print("🤖 WALL-E Ollama Diagnostics")
    print("=" * 70)
    
    # Step 1: Check connection
    client, connected = check_ollama_connection()
    if not connected:
        sys.exit(1)
    
    # Step 2: List models
    model_names = list_available_models(client)
    if not model_names:
        print("\n❌ No models available. Please install a model first:")
        print("   ollama pull qwen2.5:0.5b")
        sys.exit(1)
    
    # Step 3: Find best model
    recommended = find_best_model(model_names)
    
    # Step 4: Test the model
    if recommended:
        test_model(client, recommended)
    
    # Step 5: Show fix instructions
    show_fix_instructions(recommended)
    
    print("\n" + "=" * 70)
    print("✅ Diagnostics complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
