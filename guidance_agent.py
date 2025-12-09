import os
import requests
from dotenv import load_dotenv

load_dotenv()

from typing import Dict


class GuidanceAgent:
    """Generate AI guidance only when alert is triggered."""

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.api_url = os.getenv("GEMINI_API_URL")

        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY in .env")

        if not self.api_url:
            raise ValueError("Missing GEMINI_API_URL in .env")

    # Public entry function
    def generate_guidance(self, event: Dict) -> str:
        try:
            return self._ai_guidance(event)
        except Exception:
            return self._local_guidance(event)

    # Primary LLM call
    def _ai_guidance(self, event: Dict) -> str:
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "You are a security guidance assistant. Generate a structured, short, "
                                "professional guidance report based on this event:\n"
                                f"{event}\n\n"
                                "Sections required:\n"
                                "1. Incident summary\n"
                                "2. Immediate actions\n"
                                "3. Investigation steps\n"
                                "4. Prevention recommendations\n"
                                "5. Short staff message\n"
                            )
                        }
                    ]
                }
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }

        resp = requests.post(self.api_url, json=payload, headers=headers, timeout=15)

        if resp.status_code != 200:
            raise RuntimeError(resp.text)

        data = resp.json()

        return (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        ) or "AI guidance unavailable."

    # Fallback if Gemini fails
    def _local_guidance(self, event: Dict) -> str:
        action = (event.get("action") or "").lower()
        conf = event.get("confidence", 0)

        lines = []
        lines.append(f"Incident summary: {event.get('action', 'Unknown action')} (confidence {conf*100:.1f}%)")
        lines.append("")

        lines.append("Immediate actions:")
        if any(k in action for k in ("hand behind back", "concealment", "pocket", "hand near torso")):
            lines += [
                "- Avoid confrontation; ensure safety.",
                "- Notify security personnel.",
                "- Save the relevant video clip.",
            ]
        elif any(k in action for k in ("bending", "crouch")):
            lines += [
                "- Maintain visibility of the subject.",
                "- Save footage and notify loss-prevention.",
            ]
        else:
            lines += [
                "- Record and preserve evidence.",
                "- Report to the manager on duty.",
            ]

        lines.append("")
        lines.append("Investigation steps:")
        lines += [
            "- Save video 10 seconds before/after event.",
            "- Check POS logs for correlation.",
            "- Document staff observations.",
        ]

        lines.append("")
        lines.append("Prevention recommendations:")
        lines += [
            "- Improve staff presence in aisles.",
            "- Optimize camera coverage.",
            "- Conduct targeted training.",
        ]

        lines.append("")
        lines.append("Short staff message:")
        lines.append(
            "Stay alert and prioritize safety. Event recorded — notify loss prevention."
        )

        return "\n".join(lines)
