"""
Configuration for Alert System
Edit these values with your own credentials.
"""

# Email Alert Configuration
EMAIL_CONFIG = {
    "enabled": True,  # Set to True to enable email alerts
    "sender_email": "youremail@gmail.com",  # Your Gmail address
    "sender_password": "your gmail app password",  # Gmail App Password (16 characters)
    "recipient_emails": ["youremail@gmail.com"],  # Who to notify
}

# Slack Configuration
SLACK_CONFIG = {
    "enabled": False,  # Set to True to enable Slack alerts
    "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",  # Your Slack Webhook URL
}

# Default Detection Settings
DETECTION_CONFIG = {
    "confidence_threshold": 0.75,
    "imgsz": 320,
    "device": "auto",
    "frame_skip": 1,
}

# Alert Settings
ALERT_CONFIG = {
    "store_name": "Home",
    "alert_threshold": 1,  # Alert after N suspicious events in 1 minute
    "alert_cooldown": 90,  # Seconds between alerts (prevent spam)
}
