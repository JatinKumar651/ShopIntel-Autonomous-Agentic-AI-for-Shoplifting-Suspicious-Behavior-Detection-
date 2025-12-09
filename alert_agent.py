"""
Alert Agent: Detects suspicious activity and sends automated alerts.
Logs events with timestamp, confidence, and action description.
"""

import json
import smtplib
import requests
import cv2
import numpy as np
import base64
import threading
from datetime import datetime
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import Optional, Dict, List
from config import EMAIL_CONFIG, SLACK_CONFIG, ALERT_CONFIG
from guidance_agent import GuidanceAgent


class ActionAnalyzer:
    """Analyzes pose keypoints to describe the suspicious action."""
    
    @staticmethod
    def analyze_action(keypoints: List[List[float]]) -> str:
        """
        Analyze keypoint positions and describe the suspicious action.
        
        YOLOv8 17 keypoints (normalized 0-1):
        0: Nose, 1: L-Eye, 2: R-Eye, 3: L-Ear, 4: R-Ear
        5: L-Shoulder, 6: R-Shoulder, 7: L-Elbow, 8: R-Elbow
        9: L-Wrist, 10: R-Wrist, 11: L-Hip, 12: R-Hip
        13: L-Knee, 14: R-Knee, 15: L-Ankle, 16: R-Ankle
        """
        if not keypoints or len(keypoints) < 17:
            return "Pose detection inconclusive"
        
        # Extract key body parts
        l_wrist = keypoints[9]
        r_wrist = keypoints[10]
        l_hip = keypoints[11]
        r_hip = keypoints[12]
        nose = keypoints[0]
        l_shoulder = keypoints[5]
        r_shoulder = keypoints[6]
        l_elbow = keypoints[7]
        r_elbow = keypoints[8]
        
        actions = []
        
        # Check for concealment behavior
        if l_wrist and r_wrist:
            avg_wrist_x = (l_wrist[0] + r_wrist[0]) / 2
            hip_x = (l_hip[0] + r_hip[0]) / 2 if (l_hip and r_hip) else 0.5
            
            # Wrist far behind body (concealment)
            if avg_wrist_x < hip_x - 0.15:
                actions.append("Hand behind back (concealment risk)")
            # Wrist at torso level unnaturally
            if l_wrist and l_wrist[1] > 0.4 and l_wrist[1] < 0.7:
                if abs(l_wrist[0] - hip_x) < 0.1:
                    actions.append("Hand near torso (possible pocket placement)")
        
        # Check for crouching/bending
        if nose and l_hip and r_hip:
            hip_y = (l_hip[1] + r_hip[1]) / 2
            # Low nose = bending down
            if nose[1] > 0.5 and hip_y > 0.6:
                actions.append("Bending/crouching posture")
        
        # Check for unnatural arm angles (elbows)
        if l_elbow and r_elbow and l_shoulder and r_shoulder:
            # Elbows tucked unnaturally close to body
            l_elbow_dist = abs(l_elbow[0] - l_shoulder[0])
            r_elbow_dist = abs(r_elbow[0] - r_shoulder[0])
            if l_elbow_dist < 0.05 and r_elbow_dist < 0.05:
                actions.append("Cramped arm position (constrained movement)")
        
        # Check for asymmetric posture (one arm stiff, one moving)
        if l_wrist and r_wrist:
            wrist_spread = abs(l_wrist[0] - r_wrist[0])
            if wrist_spread < 0.1:
                actions.append("Arms held close together (concealment posture)")
        
        if not actions:
            actions.append("Pose indicates potential shoplifting behavior")
        
        return " + ".join(actions)


class AlertLogger:
    """Logs suspicious events to file."""
    
    def __init__(self, log_file: str = "suspicious_events.json"):
        self.log_file = Path(log_file)
        self._ensure_log_exists()
    
    def _ensure_log_exists(self):
        if not self.log_file.exists():
            self.log_file.write_text(json.dumps([]))
    
    def log_event(self, confidence: float, action: str, frame_num: int) -> Dict:
        """Log a suspicious event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "confidence": round(confidence, 3),
            "action": action,
            "frame": frame_num
        }
        
        events = json.loads(self.log_file.read_text())
        events.append(event)
        self.log_file.write_text(json.dumps(events, indent=2))
        
        return event
    
    def get_recent_events(self, minutes: int = 5) -> List[Dict]:
        """Get suspicious events from last N minutes."""
        if not self.log_file.exists():
            return []
        
        events = json.loads(self.log_file.read_text())
        cutoff_time = datetime.now().replace(microsecond=0) if not events else datetime.now()
        cutoff_time = cutoff_time.replace(second=0, microsecond=0)
        cutoff_time_seconds = cutoff_time.timestamp() - (minutes * 60)
        
        recent = []
        for e in events:
            try:
                evt_time = datetime.fromisoformat(e["timestamp"])
                if evt_time.timestamp() > cutoff_time_seconds:
                    recent.append(e)
            except Exception:
                pass
        
        return recent
    
    def clear_log(self):
        """Clear event log."""
        self.log_file.write_text(json.dumps([]))


class EmailAlertSender:
    """Sends email alerts."""
    
    def __init__(self, sender_email: str, sender_password: str, recipient_emails: List[str]):
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_emails = recipient_emails
    
    def send_alert(self, event: Dict, store_name: str = "Store", frame_image=None) -> bool:
        """Send email alert for suspicious activity with optional frame image."""
        try:
            subject = f"🚨 SUSPICIOUS ACTIVITY DETECTED - {store_name}"
            
            body = f"""
ALERT: Suspicious Activity Detected

Store: {store_name}
Time: {event['timestamp']}
Confidence: {event['confidence'] * 100:.1f}%
Action: {event['action']}
Frame: {event['frame']}

---
This is an automated alert from the Suspicious Activity Detection System.
Please investigate immediately.
            """
            
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ", ".join(self.recipient_emails)
            msg['Subject'] = subject
            # Append guidance if present
            if event.get('guidance'):
                body = body + "\n\n\n" + event['guidance']

            msg.attach(MIMEText(body, 'plain'))
            
            # Attach frame image if provided
            if frame_image is not None:
                try:
                    if isinstance(frame_image, np.ndarray):
                        success, encoded_image = cv2.imencode('.jpg', frame_image)
                        if success:
                            img_data = encoded_image.tobytes()
                            img = MIMEImage(img_data, 'jpeg')
                            img.add_header('Content-Disposition', 'attachment', filename='suspicious_frame.jpg')
                            msg.attach(img)
                except Exception as img_err:
                    print(f"⚠️ Could not attach image to email: {img_err}")
            
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            return True
        except Exception as e:
            print(f"❌ Email alert failed: {e}")
            return False


class SlackAlertSender:
    """Sends Slack alerts with optional frame images."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send_alert(self, event: Dict, store_name: str = "Store", frame_image=None) -> bool:
        """Send Slack alert for suspicious activity with frame image."""
        try:
            # Build Slack fields and include guidance excerpt if available
            fields = [
                {"title": "Store", "value": store_name, "short": True},
                {"title": "Time", "value": event['timestamp'], "short": True},
                {"title": "Confidence", "value": f"{event['confidence'] * 100:.1f}%", "short": True},
                {"title": "Frame", "value": str(event['frame']), "short": True},
                {"title": "Action", "value": event['action'], "short": False}
            ]

            if event.get('guidance'):
                guidance_excerpt = event['guidance'][:800]
                fields.append({"title": "AI Guidance (excerpt)", "value": guidance_excerpt, "short": False})

            message = {
                "text": "🚨 SUSPICIOUS ACTIVITY DETECTED",
                "attachments": [
                    {
                        "color": "danger",
                        "fields": fields
                    }
                ]
            }
            
            response = requests.post(self.webhook_url, json=message, timeout=5)
            
            # Send frame image as a separate message if provided
            if frame_image is not None and response.status_code == 200:
                self._send_frame_image(frame_image)
            
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Slack alert failed: {e}")
            return False
    
    def _send_frame_image(self, frame_image) -> bool:
        """Send frame image to Slack via webhook."""
        try:
            if isinstance(frame_image, np.ndarray):
                success, encoded_image = cv2.imencode('.jpg', frame_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not success:
                    return False
                img_bytes = encoded_image.tobytes()
            else:
                img_bytes = frame_image
            
            # Convert to base64 for embedding
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            # Create image message
            image_msg = {
                "attachments": [
                    {
                        "fallback": "Suspicious Activity Frame",
                        "image_url": f"data:image/jpeg;base64,{img_base64}",
                        "text": "Frame with highest confidence"
                    }
                ]
            }
            
            response = requests.post(self.webhook_url, json=image_msg, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"⚠️ Could not send frame image to Slack: {e}")
            return False


class AlertAgent:
    """Main alert agent: coordinates logging and alerting."""
    
    def __init__(self, 
                 log_file: str = "suspicious_events.json",
                 enable_email: bool = False,
                 enable_slack: bool = False):
        
        self.logger = AlertLogger(log_file)
        self.action_analyzer = ActionAnalyzer()
        self.store_name = ALERT_CONFIG.get("store_name", "Store")
        self.alert_threshold = ALERT_CONFIG.get("alert_threshold", 1)
        self.alert_cooldown = ALERT_CONFIG.get("alert_cooldown", 60)
        
        self.email_sender = None
        if enable_email and EMAIL_CONFIG.get("enabled"):
            self.email_sender = EmailAlertSender(
                EMAIL_CONFIG.get('sender_email'),
                EMAIL_CONFIG.get('sender_password'),
                EMAIL_CONFIG.get('recipient_emails', [])
            )
        
        self.slack_sender = None
        if enable_slack and SLACK_CONFIG.get("enabled"):
            self.slack_sender = SlackAlertSender(SLACK_CONFIG.get('webhook_url'))
        
        self.last_alert_time = None
        # Guidance agent for generating action/investigation guidance
        try:
            self.guidance_agent = GuidanceAgent()
        except Exception:
            self.guidance_agent = None
    
    def process_detection(self, confidence: float, keypoints: List[List[float]], 
                         frame_num: int, frame_image=None) -> Dict:
        """
        Process a suspicious detection and trigger alerts if needed.
        Args:
            confidence: Detection confidence score
            keypoints: Keypoint coordinates
            frame_num: Frame number
            frame_image: Optional frame image (numpy array or bytes)
        Returns:
            event dict
        """
        action = self.action_analyzer.analyze_action(keypoints)
        event = self.logger.log_event(confidence, action, frame_num)

        # Store frame image for alert sending (guidance generation deferred
        # until we are actually about to send alerts to avoid unnecessary API calls)
        event['frame_image'] = frame_image
        
        # Check if we should send alert
        self._send_alerts_if_needed(event)
        
        return event
    
    def _send_alerts_if_needed(self, event: Dict):
        """Send alerts if threshold is met and cooldown allows (non-blocking)."""
        import time
        now = time.time()
        
        # Check cooldown
        if self.last_alert_time and (now - self.last_alert_time) < self.alert_cooldown:
            return
        
        # Check threshold
        recent = self.logger.get_recent_events(minutes=1)
        if len(recent) >= self.alert_threshold:
            # Update last alert time immediately
            self.last_alert_time = now
            
            # Send alerts in background thread (non-blocking)
            alert_thread = threading.Thread(
                target=self._send_alerts_background,
                args=(event,),
                daemon=True
            )
            alert_thread.start()
    
    def _send_alerts_background(self, event: Dict):
        """Background thread function to send alerts without blocking video stream."""
        try:
            # Generate guidance only when about to send alerts (not on every detection)
            if self.guidance_agent and not event.get('guidance'):
                try:
                    guidance_text = self.guidance_agent.generate_guidance(event)
                    if guidance_text:
                        event['guidance'] = guidance_text
                except Exception as e:
                    print(f"⚠️ Guidance generation at alert time failed: {e}")
            
            frame_image = event.pop('frame_image', None)
            
            # Convert frame image to base64 for storage in JSON
            frame_base64 = None
            if frame_image is not None:
                try:
                    if isinstance(frame_image, np.ndarray):
                        success, encoded_image = cv2.imencode('.jpg', frame_image)
                        if success:
                            frame_base64 = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
                            event['frame_base64'] = frame_base64
                except Exception as img_err:
                    print(f"⚠️ Could not encode frame image: {img_err}")
            
            email_sent = False
            if self.email_sender:
                email_sent = self.email_sender.send_alert(event, self.store_name, frame_image=frame_image)
            
            if self.slack_sender:
                self.slack_sender.send_alert(event, self.store_name, frame_image=frame_image)
            
            # Mark event as email sent if email was successfully sent
            if email_sent:
                event['email_sent'] = True
                # Update the event in the log file
                try:
                    events = json.loads(self.logger.log_file.read_text())
                    for e in events:
                        if e['timestamp'] == event['timestamp']:
                            e['email_sent'] = True
                            if event.get('guidance'):
                                e['guidance'] = event['guidance']
                            if frame_base64:
                                e['frame_base64'] = frame_base64
                            break
                    self.logger.log_file.write_text(json.dumps(events, indent=2))
                except Exception as log_err:
                    print(f"⚠️ Could not update event log with email_sent flag: {log_err}")
        except Exception as e:
            print(f"❌ Background alert sending failed: {e}")
    
    def get_event_summary(self) -> Dict:
        """Get summary of recent events."""
        recent = self.logger.get_recent_events(minutes=5)
        return {
            "total_events": len(recent),
            "events": recent
        }


if __name__ == "__main__":
    print("Alert Agent module loaded. Import and use AlertAgent in your app.")
