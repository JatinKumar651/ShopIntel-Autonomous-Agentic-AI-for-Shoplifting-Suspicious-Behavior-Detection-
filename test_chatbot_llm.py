#!/usr/bin/env python
"""Quick test script to verify Gemini API is working for chatbot."""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from chatbot_agent import ChatbotAgent

print("=" * 60)
print("CHATBOT LLM TEST")
print("=" * 60)

# Initialize chatbot
print("\n🤖 Initializing ChatbotAgent...")
chatbot = ChatbotAgent()

print(f"✅ ChatbotAgent initialized")
print(f"API Key present: {bool(chatbot.gemini_api_key)}")
print(f"API URL: {chatbot.gemini_api_url[:50]}...")

# Test message
test_message = "I just caught a customer trying to hide items in their bag. I'm nervous and not sure what to do. Should I confront them?"

print(f"\n📝 Test message: {test_message[:60]}...")
print(f"\n🔄 Generating response (this calls LLM)...\n")

response = chatbot.generate_response(test_message)

print("\n" + "=" * 60)
print("RESPONSE:")
print("=" * 60)
print(response)
print("\n" + "=" * 60)

# Check if it's a fallback response (static keywords)
fallback_keywords = ["Take a moment", "Your safety and well-being", "I understand you're feeling worried"]
is_fallback = any(keyword in response for keyword in fallback_keywords)

if is_fallback:
    print("⚠️ WARNING: Response appears to be fallback (static local guidance)")
    print("❌ LLM was NOT called successfully")
else:
    print("✅ Response appears to be from LLM (dynamic generated content)")
    print("✅ LLM WAS called successfully")

print("=" * 60)
