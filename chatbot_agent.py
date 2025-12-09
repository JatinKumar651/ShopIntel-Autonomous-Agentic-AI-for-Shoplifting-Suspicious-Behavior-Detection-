"""
Chatbot Agent: Provides emotional support and guidance for staff/store owners
dealing with suspicious activity incidents.
Uses Gemini API for intelligent responses.
"""

import json
import requests
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from dotenv import load_dotenv

load_dotenv()


class ChatbotAgent:
    """Chatbot for supporting staff with emotional guidance and incident analysis."""
    
    def __init__(self, chat_history_file: str = "chat_history.json"):
        self.chat_history_file = Path(chat_history_file)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.gemini_api_url = os.getenv("GEMINI_API_URL", "")
        self._ensure_chat_history_exists()
        self.system_prompt = """You are a compassionate and professional support agent helping store staff and owners 
deal with suspicious activity incidents. Your role is to:

1. **Listen and Understand**: Carefully understand the incident the person is describing
2. **Provide Emotional Support**: Show empathy, validate their concerns, and help them stay calm
3. **Guide Them**: Provide practical advice on how to handle the situation
4. **Reassure**: Remind them that they did the right thing by reporting the incident
5. **Professional Guidance**: Offer best practices for similar situations in the future

Your responses should be:
- Warm, empathetic, and supportive
- Clear and actionable
- Professional but friendly
- Concise but thorough

Always start by acknowledging what they experienced, then provide guidance."""
    
    def _ensure_chat_history_exists(self):
        """Create chat history file if it doesn't exist."""
        if not self.chat_history_file.exists():
            self.chat_history_file.write_text(json.dumps([]))
    
    def _get_conversation_context(self, limit: int = 10) -> str:
        """Get recent chat history as context for the conversation."""
        try:
            chat_history = json.loads(self.chat_history_file.read_text())
            recent = chat_history[-limit:]
            
            if not recent:
                return ""
            
            context = "Recent conversation history:\n"
            for msg in recent:
                context += f"\nStaff: {msg['user_message']}\nAgent: {msg['agent_response'][:200]}..."
            
            return context
        except Exception:
            return ""
    
    def generate_response(self, user_message: str) -> Optional[str]:
        """
        Generate a supportive response using Gemini API.
        
        Args:
            user_message: The user's message describing their incident
            
        Returns:
            Generated response from Gemini API or fallback message on API failure
        """
        print(f"\n🤖 Generating response for: {user_message[:50]}...")
        
        # Always try API first if credentials available
        if self.gemini_api_key and self.gemini_api_url:
            print(f"🔑 API credentials found, attempting LLM call...")
            api_response = self._call_gemini_api(user_message)
            if api_response:
                print(f"✅ LLM response received successfully")
                return api_response
            else:
                # Log API failure and fallback
                print(f"⚠️ Gemini API failed, falling back to local response")
        else:
            print(f"⚠️ No API credentials configured")
        
        # Fallback to local guidance if API not configured or failed
        print(f"📝 Using fallback response")
        return self._fallback_response(user_message)
    
    def _call_gemini_api(self, user_message: str) -> Optional[str]:
        """
        Call Gemini API for response generation.
        
        Args:
            user_message: User's message
            
        Returns:
            Response text or None if API call fails
        """
        try:
            # Verify credentials
            if not self.gemini_api_key or not self.gemini_api_url:
                print(f"⚠️ Missing API credentials")
                return None
            
            # Get recent conversation context
            context = self._get_conversation_context()
            
            # Build the full prompt
            full_prompt = f"""{self.system_prompt}

{context}

Staff member's current message: {user_message}

Please provide a supportive and helpful response."""
            
            # Use Google Generative Language API format (same as guidance_agent)
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": full_prompt
                            }
                        ]
                    }
                ]
            }
            
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.gemini_api_key
            }
            
            print(f"📡 Calling Gemini API...")
            response = requests.post(self.gemini_api_url, json=payload, headers=headers, timeout=15)
            
            print(f"📡 API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"📡 API Response Keys: {data.keys()}")
                
                # Parse Gemini response format
                api_text = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                )
                
                if api_text and api_text.strip():
                    print(f"✅ Got LLM response: {len(api_text)} chars")
                    return api_text
                else:
                    print(f"⚠️ Empty response from API")
                    return None
            else:
                print(f"❌ API returned status {response.status_code}")
                print(f"Response: {response.text[:500]}")
                return None
        
        except requests.exceptions.Timeout:
            print(f"⚠️ Gemini API timeout (15s)")
            return None
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Gemini API request failed: {str(e)}")
            return None
        except Exception as e:
            print(f"⚠️ Error calling Gemini API: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _fallback_response(self, user_message: str) -> str:
        """Provide a fallback response when API is unavailable."""
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ['scared', 'afraid', 'nervous', 'worried', 'anxious']):
            return """I understand you're feeling worried about this incident. That's a completely natural reaction. 
Your safety and well-being come first. Here's what I recommend:

1. **Take a moment**: Breathe deeply and center yourself. What you experienced was handled properly.
2. **You did the right thing**: Reporting suspicious activity protects your store and team.
3. **Follow protocols**: Stick to your store's established procedures for handling such incidents.
4. **Debrief with management**: Talk to your manager or owner about what happened. They can provide support and guidance.
5. **Remember**: Security incidents happen everywhere. This doesn't reflect on you or your abilities.

You handled this responsibly. Is there anything specific about the incident you'd like help processing?"""
        
        elif any(word in message_lower for word in ['threatened', 'threat', 'danger', 'dangerous', 'violent']):
            return """I'm sorry you experienced a threatening situation. Your safety is paramount, and you absolutely did the right thing by reporting it.

**Immediate steps:**
1. **Ensure your safety**: Step away from the situation if still at risk
2. **Contact management**: Inform your supervisor or store owner immediately
3. **Document details**: Write down what happened while it's fresh
4. **Support**: Consider speaking with HR or employee assistance if available

**Remember:**
- This is NOT your fault
- Your quick thinking helped protect the store and team
- These incidents, while unsettling, are handled by trained security professionals

You showed great composure. Take care of yourself today. Would you like to talk about specific aspects of what happened?"""
        
        elif any(word in message_lower for word in ['confused', 'confused about', 'what should', 'help me', 'guidance']):
            return """I'm here to help you through this. Let me guide you:

**When facing suspicious activity:**
1. **Stay calm** - Your composure helps the situation
2. **Observe carefully** - Note details: appearance, behavior, items of interest
3. **Alert management** - Inform your supervisor or store owner immediately
4. **Don't confront** - Let trained security handle confrontations
5. **Document** - Write down timestamps and what you observed
6. **Support colleagues** - Help teammates who may also be shaken

**After the incident:**
- Debrief with management
- Take a break if needed
- Reflect on what went well (you caught it!)
- Discuss prevention for next time

You handled this professionally. What specific aspect would you like guidance on?"""
        
        else:
            return """Thank you for sharing this with me. I want you to know that handling suspicious activity incidents takes courage and awareness.

**Here's what I want to emphasize:**
- You did the right thing by being alert
- Reporting incidents protects everyone
- Security teams are trained to handle these situations
- Your well-being matters - take care of yourself

**Next steps:**
1. Inform your manager/owner if you haven't already
2. Document what you observed
3. Follow your store's incident protocol
4. Take a moment to decompress

You showed great responsibility. Remember, you're part of a security-conscious team, and that's valuable.

Would you like to talk through any specific concerns about what happened?"""
    
    def save_message(self, user_message: str, agent_response: str) -> Dict:
        """Save a message exchange to chat history."""
        try:
            chat_history = json.loads(self.chat_history_file.read_text())
            
            message_record = {
                "timestamp": datetime.now().isoformat(),
                "user_message": user_message,
                "agent_response": agent_response
            }
            
            chat_history.append(message_record)
            self.chat_history_file.write_text(json.dumps(chat_history, indent=2))
            
            return message_record
        except Exception as e:
            print(f"⚠️ Could not save message: {e}")
            return {}
    
    def get_chat_history(self) -> List[Dict]:
        """Get all chat history."""
        try:
            if self.chat_history_file.exists():
                return json.loads(self.chat_history_file.read_text())
            return []
        except Exception as e:
            print(f"⚠️ Error reading chat history: {e}")
            return []
    
    def clear_chat_history(self) -> bool:
        """Clear all chat history."""
        try:
            self.chat_history_file.write_text(json.dumps([]))
            return True
        except Exception as e:
            print(f"⚠️ Could not clear chat history: {e}")
            return False
    
    def get_chat_summary(self) -> Dict:
        """Get summary statistics of chat history."""
        try:
            history = self.get_chat_history()
            return {
                "total_messages": len(history),
                "last_message": history[-1]['timestamp'] if history else None,
                "messages": history
            }
        except Exception:
            return {"total_messages": 0, "last_message": None, "messages": []}


if __name__ == "__main__":
    print("Chatbot Agent module loaded. Import and use ChatbotAgent in your app.")
