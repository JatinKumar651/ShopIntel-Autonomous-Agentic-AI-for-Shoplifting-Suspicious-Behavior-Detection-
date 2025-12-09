# Streamlit UI for Suspicious Activity Detection

This Streamlit demo lets you upload a video (or use your webcam) and runs frame-level inference using the project's YOLO + XGBoost pipeline with **automatic alerting**.

## Key Features

- **Real-Time Detection** — Upload videos or use webcam feed
- **Auto-Alerts** — Sends email/Slack notifications when suspicious activity is detected
- **Event Logging** — Logs all detections with timestamps, confidence, and action descriptions
- **Action Analysis** — Describes what kind of suspicious behavior was detected (concealment, crouching, etc.)
- **Performance Tuning** — Adjust inference size, frame-skip, and device (GPU/CPU)

## How to Run

### 1. Set up Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Start the Streamlit App
```powershell
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## Alert Configuration

### Setup Email Alerts

1. **Edit `config.py`** and set:
   ```python
   EMAIL_CONFIG = {
       "enabled": True,  # Set to True to enable
       "sender_email": "your_gmail@gmail.com",
       "sender_password": "your_app_password",  # 16-char App Password
       "recipient_emails": ["alert@example.com", "security@store.com"],
   }
   ```

2. **Get Gmail App Password:**
   - Go to [Google Account Security](https://myaccount.google.com/security)
   - Enable 2-Factor Authentication
   - Generate an "App Password" for Mail
   - Copy the 16-character password to `config.py`

3. **In the Streamlit UI**, check the **"Send Email Alerts"** toggle to activate

### Setup Slack Alerts

1. **Create Slack App:** Visit [api.slack.com/apps](https://api.slack.com/apps)
2. **Enable Incoming Webhooks** → Create New Webhook → Select channel
3. **Copy the Webhook URL** and paste into `config.py`:
   ```python
   SLACK_CONFIG = {
       "enabled": True,  # Set to True to enable
       "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
   }
   ```
4. **In the Streamlit UI**, check the **"Send Slack Alerts"** toggle to activate

---

## Settings Explained

| Setting | Effect |
|---------|--------|
| **Confidence Threshold** | Only flag detections above this confidence (0.75 = 75% sure) |
| **Inference Image Size** | Lower = faster but less accurate; 320 is fast, 640 is balanced, 1280 is accurate |
| **Process Every Nth Frame** | Skip frames for speed; 3 = process every 3rd frame (3× faster) |
| **Device** | `cuda` = GPU (fast), `cpu` = CPU (slow but always works), `auto` = auto-detect |
| **Alert Threshold** | Trigger alert after N suspicious events in 1 minute |

---

## Example: Fast Setup for Real-Time Monitoring

**Sidebar Settings:**
- Device: `cuda` (or `cpu`)
- Inference Image Size: `320`
- Process Every Nth Frame: `2`
- Alerts: Email + Slack enabled
- Alert Threshold: `1` (alert immediately on first detection)

---

## Output Files

- **suspicious_events.json** — JSON log of all detections with timestamps, confidence, actions
- View it to analyze patterns, false positives, or replay incidents

---

## Troubleshooting

**"No module named 'streamlit'"**
```powershell
pip install -r requirements.txt
```

**Gmail alerts not working**
- Make sure 2FA is enabled
- Use the 16-character App Password, not your Gmail password
- Check firewall/antivirus isn't blocking SMTP

**Slack alerts not working**
- Verify webhook URL is correct
- Test webhook: `curl -X POST -H 'Content-type: application/json' --data '{"text":"Test"}' <WEBHOOK_URL>`

**Slow FPS**
- Lower `Inference Image Size` to 320
- Increase `Process Every Nth Frame` to 3–5
- Ensure GPU is available and being used

---

**That's it!** Your surveillance system now automatically alerts you to suspicious activity. 🚨

