import streamlit as st
import cv2
import tempfile
import time
from pathlib import Path
import numpy as np
import json

from model_runner import Detector
from alert_agent import AlertAgent
from guidance_agent import GuidanceAgent
from chatbot_agent import ChatbotAgent
from config import DETECTION_CONFIG, ALERT_CONFIG, EMAIL_CONFIG, SLACK_CONFIG

@st.cache_resource
def get_detector(yolo_path, xgb_path, conf, device, imgsz):
    return Detector(yolo_path=yolo_path or None, xgb_path=xgb_path or None, conf_threshold=conf, device=device, imgsz=imgsz)


st.set_page_config(page_title="ShopIntel", layout="wide", initial_sidebar_state="expanded")

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 0;
        }
        
        /* Header and Title Section */
        .header-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
        }
        
        .main-logo {
            font-size: 4em;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        
        .main-title {
            text-align: center;
            font-size: 2.8em;
            font-weight: 900;
            color: #ffffff;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
            margin: 0;
        }
        
        .main-subtitle {
            text-align: center;
            font-size: 1.1em;
            color: #e0e7ff;
            margin-top: 0.5rem;
            font-weight: 500;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
        }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(102, 126, 234, 0.5);
        }
        
        .stButton > button:active {
            transform: translateY(0);
        }
        
        /* Active navigation button styling */
        .nav-btn-active {
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%) !important;
            font-size: 1.1em !important;
            font-weight: 700 !important;
            box-shadow: 0 8px 20px rgba(67, 233, 123, 0.4) !important;
            transform: scale(1.08) !important;
        }
        
        .nav-btn-active:hover {
            transform: scale(1.12) !important;
            box-shadow: 0 12px 28px rgba(67, 233, 123, 0.6) !important;
        }
        
        .nav-btn-inactive {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-size: 0.95em;
            font-weight: 600;
            opacity: 0.8;
        }
        
        .nav-btn-inactive:hover {
            opacity: 1;
            transform: translateY(-2px);
        }
        
        /* Cards and containers */
        .card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
            border-left: 4px solid #667eea;
        }
        
        .card-detection {
            border-left: 4px solid #667eea;
        }
        
        .card-chatbot {
            border-left: 4px solid #f093fb;
        }
        
        .card-logs {
            border-left: 4px solid #4facfe;
        }
        
        .card-email {
            border-left: 4px solid #43e97b;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #1a202c;
            font-weight: 800;
        }
        
        h1 {
            border-bottom: 3px solid #667eea;
            padding-bottom: 0.5rem;
        }
        
        /* Alert boxes */
        .stAlert {
            border-radius: 8px;
            padding: 1rem;
        }
        
        .alert-success {
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
            color: #1a202c;
            border-radius: 8px;
        }
        
        .alert-warning {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            color: #1a202c;
            border-radius: 8px;
        }
        
        .alert-error {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
            color: white;
            border-radius: 8px;
        }
        
        /* Input elements */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select {
            border-radius: 8px;
            border: 2px solid #e2e8f0;
            padding: 10px;
        }
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border: 2px solid #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        /* Sidebar control spacing fix */
        [data-testid="stSidebar"] .stTextInput,
        [data-testid="stSidebar"] .stSelectbox,
        [data-testid="stSidebar"] .stCheckbox,
        [data-testid="stSidebar"] .stRadio {
            margin-bottom: 1.5rem;
            position: relative;
            z-index: 1;
        }
        
        /* Ensure dropdown menus don't get covered */
        [data-testid="stSidebar"] div[role="listbox"],
        [data-testid="stSidebar"] div[role="combobox"] {
            z-index: 10;
            position: relative;
        }
        
        /* Expander styling */
        .streamlit-expanderContent {
            border-radius: 8px;
            background: #f8f9fa;
        }
        
        /* Chat message styling */
        .chat-message {
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .chat-message-user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: 3rem;
            border-radius: 18px 18px 4px 18px;
        }
        
        .chat-message-assistant {
            background: #f0f4f8;
            color: #1a202c;
            margin-right: 3rem;
            border-radius: 18px 18px 18px 4px;
        }
        
        /* Divider styling */
        hr {
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #667eea, transparent);
            margin: 1.5rem 0;
        }
        
        /* Page indicator */
        .page-indicator {
            text-align: center;
            padding: 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
            font-weight: 600;
            margin-bottom: 1.5rem;
        }
        
        /* Video container */
        .video-container {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.2);
        }
        
        /* Stats boxes */
        .stat-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
        
        .stat-number {
            font-size: 2.5em;
            font-weight: 900;
            margin: 0.5rem 0;
        }
        
        .stat-label {
            font-size: 1em;
            opacity: 0.9;
        }
    </style>
""", unsafe_allow_html=True)

# Display enhanced header
st.markdown("""
    <div class="header-container">
        <div class="main-title">ShopIntel</div>
        <div class="main-subtitle">Autonomous Agentic AI for Shoplifting & Suspicious Behavior Detection</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

with st.sidebar:
    st.markdown("---")
    st.markdown("Features")
    if "page" not in st.session_state:
        st.session_state.page = "detection"
    
    # Create custom navigation buttons with visual impact
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.page == "detection":
            st.markdown("""
                <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                            border-radius: 10px; padding: 12px; text-align: center; 
                            font-weight: 700; font-size: 1.1em; color: white;
                            box-shadow: 0 8px 20px rgba(67, 233, 123, 0.4);
                            transform: scale(1.08);">
                    🎥 Detection
                </div>
            """, unsafe_allow_html=True)
        else:
            if st.button("🎥Detection", key="nav_detection", use_container_width=True):
                st.session_state.page = "detection"
                st.rerun()
    
    with col2:
        if st.session_state.page == "chatbot":
            st.markdown("""
                <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                            border-radius: 10px; padding: 12px; text-align: center; 
                            font-weight: 700; font-size: 1.1em; color: white;
                            box-shadow: 0 8px 20px rgba(67, 233, 123, 0.4);
                            transform: scale(1.08);">
                    💬 Chatbot
                </div>
            """, unsafe_allow_html=True)
        else:
            if st.button("💬 Chatbot", key="nav_chatbot", use_container_width=True):
                st.session_state.page = "chatbot"
                st.rerun()
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.session_state.page == "logs":
            st.markdown("""
                <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                            border-radius: 10px; padding: 12px; text-align: center; 
                            font-weight: 700; font-size: 1.1em; color: white;
                            box-shadow: 0 8px 20px rgba(67, 233, 123, 0.4);
                            transform: scale(1.08);">
                    📋 Logs
                </div>
            """, unsafe_allow_html=True)
        else:
            if st.button("📋 Logs", key="nav_logs", use_container_width=True):
                st.session_state.page = "logs"
                st.rerun()
    
    with col4:
        if st.session_state.page == "email":
            st.markdown("""
                <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                            border-radius: 10px; padding: 12px; text-align: center; 
                            font-weight: 700; font-size: 1.1em; color: white;
                            box-shadow: 0 8px 20px rgba(67, 233, 123, 0.4);
                            transform: scale(1.08);">
                    📧 Email
                </div>
            """, unsafe_allow_html=True)
        else:
            if st.button("📧 Email", key="nav_email", use_container_width=True):
                st.session_state.page = "email"
                st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Detection Controls")
    conf = st.slider(" Confidence Threshold", 0.0, 1.0, DETECTION_CONFIG.get("confidence_threshold", 0.75), 0.01)
    run_mode = st.radio("📹 Mode", ["Upload video"])
    device_choice = st.selectbox("💻 Device", ["auto", "cuda", "cpu"], index=0)
    imgsz = st.slider(" Inference Size (px)", 320, 1280, DETECTION_CONFIG.get("imgsz", 640), step=32)
    frame_skip = st.slider(" Frame Skip (higher = faster)", 1, 10, DETECTION_CONFIG.get("frame_skip", 1))
    
    st.markdown("---")
    st.markdown("### 🔔 Alert Settings")
    enable_alerts = st.checkbox(" Enable Auto-Alerts", value=True)
    
    st.markdown("#### 📧 Email Alerts")
    enable_email = st.checkbox(" Send Email Alerts", value=EMAIL_CONFIG.get("enabled", False))
    if enable_email and not EMAIL_CONFIG.get("enabled"):
        st.warning("⚠️ Email alerts disabled in config.py. Edit config.py to enable.")

# Initialize session state variables
if "stop_stream" not in st.session_state:
    st.session_state.stop_stream = False

if "detection_started" not in st.session_state:
    st.session_state.detection_started = False

if "detector" not in st.session_state:
    st.session_state.detector = None

if "cap" not in st.session_state:
    st.session_state.cap = None

if "alert_agent" not in st.session_state:
    st.session_state.alert_agent = None

if "video_path" not in st.session_state:
    st.session_state.video_path = None

# Main layout - content on left, sidebar on right
main_col, side = st.columns([3, 1])

with main_col:
    # Route to pages: if not on detection page, render that page and stop
    if st.session_state.get("page", "detection") != "detection":
        if st.session_state.get("page") == "chatbot":
            st.markdown('<div class="page-indicator">💬 Support Chatbot</div>', unsafe_allow_html=True)
            st.markdown("**For store staff & owners**: Talk about any suspicious activity incidents. Get emotional support and guidance.")
            
            # Initialize chatbot agent
            if "chatbot_agent" not in st.session_state:
                st.session_state.chatbot_agent = ChatbotAgent()
            
            chatbot = st.session_state.chatbot_agent
            
            # Chat history controls
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📋 View Chat History", key="view_chat_history", use_container_width=True):
                    st.session_state.show_chat_history = not st.session_state.get("show_chat_history", False)
            
            with col2:
                if st.button("🗑️ Delete Chat History", key="delete_chat_history", use_container_width=True):
                    if chatbot.clear_chat_history():
                        st.success("✅ Chat history cleared.")
                        st.rerun()
                    else:
                        st.error("❌ Could not clear chat history.")
            
            with col3:
                if st.button("🔄 Clear Session", key="clear_session", use_container_width=True):
                    st.session_state.chat_messages = []
                    st.success("✅ Current session cleared.")
                    st.rerun()
            
            # Display chat history if toggled
            if st.session_state.get("show_chat_history", False):
                st.divider()
                st.subheader("📜 Chat History")
                history = chatbot.get_chat_history()
                if history:
                    for i, msg in enumerate(history):
                        with st.expander(f"Message {i+1}: {msg['timestamp'][:19]}"):
                            st.write("**You:** " + msg['user_message'])
                            st.write("**Agent:** " + msg['agent_response'])
                else:
                    st.info("No chat history yet.")
                st.divider()
            
            # Chat interface
            st.subheader("💭 Tell me about your experience")
            
            # Initialize chat session storage if needed
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []
            
            # Display previous messages in this session
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            
            # User input - using chat_input for Enter key support and auto-clear
            user_input = st.chat_input("Share your incident... I'm here to listen and help.")
            
            if user_input:
                # Add user message to session
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": user_input
                })
                
                # Display user message immediately
                with st.chat_message("user"):
                    st.write(user_input)
                
                # Generate response
                with st.spinner("💭 Thinking... I'm here to help."):
                    response = chatbot.generate_response(user_input)
                
                if response:
                    # Add agent response to session
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    # Save to persistent chat history
                    chatbot.save_message(user_input, response)
                    
                    # Display the response immediately
                    with st.chat_message("assistant"):
                        st.write(response)
                else:
                    st.error("❌ Could not generate response. Please try again.")
        
        elif st.session_state.get("page") == "logs":
            st.markdown('<div class="page-indicator">📋 Event Logs</div>', unsafe_allow_html=True)
            st.markdown("View and manage all detection events.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 View All Logs (JSON)", key="view_all_logs", use_container_width=True):
                    try:
                        log_file = Path("suspicious_events.json")
                        if log_file.exists():
                            events = json.loads(log_file.read_text())
                            with st.expander("📄 Full Event Log (JSON)", expanded=True):
                                st.json(events)
                        else:
                            st.info("Log file not created yet.")
                    except Exception as e:
                        st.error(f"Error reading logs: {e}")
            
            with col2:
                if st.button("Delete All Logs", key="delete_all_logs"):
                    try:
                        Path("suspicious_events.json").write_text("[]")
                        st.success("✅ All event logs cleared.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not clear logs: {e}")
            
            st.divider()
            st.subheader("Individual Log Entries")
            try:
                log_file = Path("suspicious_events.json")
                if log_file.exists():
                    events = json.loads(log_file.read_text())
                    if events:
                        for i, e in enumerate(events):
                            with st.expander(f"🔍 Event {i+1}: {e['timestamp']} — {e['action']} ({e['confidence']*100:.1f}%)"):
                                st.json(e)
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button("Delete This Event", key=f"delete_event_{i}"):
                                        try:
                                            events.pop(i)
                                            Path("suspicious_events.json").write_text(json.dumps(events, indent=2))
                                            st.success("✅ Event deleted.")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Could not delete event: {e}")
                    else:
                        st.info("No events logged yet.")
                else:
                    st.info("Log file not created yet.")
            except Exception as e:
                st.error(f"Error reading logs: {e}")
        
        elif st.session_state.get("page") == "email":
            st.markdown('<div class="page-indicator">📧 Email History</div>', unsafe_allow_html=True)
            st.markdown("View and manage emails sent for detected suspicious activities.")
            
            try:
                log_file = Path("suspicious_events.json")
                if log_file.exists():
                    events = json.loads(log_file.read_text())
                    # Filter only events where email was actually sent
                    email_events = [e for e in events if e.get('email_sent', False)]
                    
                    if email_events:
                        st.write(f"**Total Emails Sent:** {len(email_events)}")
                        for i, e in enumerate(email_events):
                            with st.expander(f"📧 Email {i+1}: {e['timestamp']} — {e['action']}"):
                                # Display frame image if available
                                if e.get('frame_base64'):
                                    try:
                                        import base64
                                        frame_bytes = base64.b64decode(e['frame_base64'])
                                        st.image(frame_bytes, caption="Suspicious Activity Frame", use_column_width=True)
                                    except Exception as img_err:
                                        st.warning(f"Could not display frame image: {img_err}")
                                
                                # Show full event details
                                # st.write("**Event Details:**")
                                # st.json(e)
                                
                                # Display guidance if available
                                if e.get('guidance'):
                                    st.write("**AI Guidance:**")
                                    st.info(e['guidance'])
                                
                                # Show reconstructed email content if available
                                if "email_sent" in e or e.get('confidence'):
                                    st.write("**Email Content Preview:**")
                                    email_preview = f"""
**Subject:** Suspicious Activity Detected - {e['action']}

**Body:**
Timestamp: {e['timestamp']}
Action: {e['action']}
Confidence: {e['confidence']*100:.1f}%
Keypoints: {e.get('keypoints', 'N/A')}
{'Guidance: ' + e.get('guidance', '') if e.get('guidance') else ''}
                                    """
                                    st.text(email_preview)
                                
                                # Delete button for individual email
                                if st.button("Delete This Email", key=f"delete_email_{i}"):
                                    try:
                                        events.pop(events.index(e))
                                        Path("suspicious_events.json").write_text(json.dumps(events, indent=2))
                                        st.success("✅ Email record deleted.")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Could not delete email: {e}")
                    else:
                        st.info("No emails sent yet.")
                else:
                    st.info("Email log file not created yet.")
            except Exception as e:
                st.error(f"Error reading email history: {e}")

        st.stop()
    
    # Detection Page Header
    st.markdown('<div class="page-indicator">🎥 Live Detection & Video Processing</div>', unsafe_allow_html=True)
    
    if run_mode == "Upload video":
        st.markdown("### 📤 Upload Video File")
        uploaded = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

        if uploaded is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            video_path = tfile.name

            # Store in session state
            st.session_state.video_path = video_path
            
            # Initialize detector once (cached)
            if st.session_state.detector is None:
                st.session_state.detector = get_detector(None, None, conf, device_choice, imgsz)

            # Initialize video capture
            if st.session_state.cap is None:
                st.session_state.cap = cv2.VideoCapture(video_path)

            # Initialize alert agent
            if st.session_state.alert_agent is None:
                st.session_state.alert_agent = AlertAgent(
                    log_file="suspicious_events.json",
                    enable_email=enable_alerts and enable_email,
                    enable_slack=False
                ) if enable_alerts else None

            cap = st.session_state.cap
            detector = st.session_state.detector
            alert_agent = st.session_state.alert_agent

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25

            # Create placeholders (outside loop so they don't get recreated)
            img_placeholder = st.image([])
            status = st.empty()
            progress = st.progress(0)

            # Start/Stop buttons - kept at top
            cols = st.columns(2)
            with cols[0]:
                if st.button("▶️ Start Detection", key="start_btn_upload", use_container_width=True):
                    st.session_state.detection_started = True
                    st.session_state.stop_stream = False
            with cols[1]:
                if st.button("⏹️ Stop Detection", key="stop_btn_upload", use_container_width=True):
                    st.session_state.detection_started = False
                    st.session_state.stop_stream = True

            # Process video only if detection has started
            if st.session_state.detection_started:
                frame_idx = 0
                last_time = time.time()
                cached_annotated = None
                cached_summary = {"suspicious": 0, "normal": 0}
                cached_suspicious = []

                while cap.isOpened() and st.session_state.detection_started and not st.session_state.stop_stream:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # optionally skip frames to increase throughput
                    if frame_idx % frame_skip == 0:
                        annotated, summary, suspicious_events = detector.predict_frame(frame)
                        cached_annotated = annotated
                        cached_summary = summary
                        cached_suspicious = suspicious_events
                    else:
                        # reuse last annotated frame to avoid extra inference
                        annotated = cached_annotated if cached_annotated is not None else frame
                        summary = cached_summary
                        suspicious_events = cached_suspicious

                    # Process alerts - send to sidebar
                    if alert_agent and suspicious_events:
                        for evt in suspicious_events:
                            event_data = alert_agent.process_detection(
                                evt['confidence'],
                                evt['keypoints'],
                                frame_idx,
                                frame_image=annotated  # Pass annotated frame
                            )
                            
                            # Send alert to sidebar instead of main area
                            with side:
                                st.warning(
                                    f"🚨 **SUSPICIOUS ACTIVITY** @ {event_data['timestamp']}\n"
                                    f"Confidence: {event_data['confidence']*100:.1f}%\n"
                                    f"Action: {event_data['action']}"
                                )

                    # convert BGR to RGB for Streamlit
                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                    # display
                    img_placeholder.image(rgb, use_container_width=True)

                    frame_idx += 1
                    if total_frames:
                        progress.progress(min(frame_idx / total_frames, 1.0))

                    now = time.time()
                    elapsed = now - last_time
                    if elapsed > 0:
                        status.text(f"Frame: {frame_idx} | FPS: {1/elapsed:.1f}")
                    last_time = now

                    # slight pause to allow UI to update
                    time.sleep(0.01)

                cap.release()
                st.session_state.cap = None
                status.text("Detection finished.")
                
                if alert_agent:
                    event_summary = alert_agent.get_event_summary()
                    with side:
                        st.info(f"📊 Total suspicious events logged: {event_summary['total_events']}")
            else:
                if uploaded is not None:
                    st.info("📹 Video loaded. Click **Start Detection** to begin processing.")

    else:
        # Webcam mode
        st.markdown("### 📹 Webcam Live Stream")
        if st.session_state.detector is None:
            st.session_state.detector = get_detector(None, None, conf, device_choice, imgsz)
        
        if st.session_state.cap is None:
            st.session_state.cap = cv2.VideoCapture(0)

        # Initialize alert agent
        if st.session_state.alert_agent is None:
            st.session_state.alert_agent = AlertAgent(
                log_file="suspicious_events.json",
                enable_email=enable_alerts and enable_email,
                enable_slack=False
            ) if enable_alerts else None

        detector = st.session_state.detector
        cap = st.session_state.cap
        alert_agent = st.session_state.alert_agent

        img_placeholder = st.image([])
        status = st.empty()

        # Start/Stop buttons - kept at top
        cols = st.columns(2)
        with cols[0]:
            if st.button("▶️ Start Detection", key="start_btn_webcam", use_container_width=True):
                st.session_state.detection_started = True
                st.session_state.stop_stream = False
        with cols[1]:
            if st.button("⏹️ Stop Detection", key="stop_btn_webcam", use_container_width=True):
                st.session_state.detection_started = False
                st.session_state.stop_stream = True
        
        # Process webcam only if detection has started
        if st.session_state.detection_started:
            frame_idx = 0
            last_time = time.time()
            cached_annotated = None
            cached_summary = {"suspicious": 0, "normal": 0}
            cached_suspicious = []

            while cap.isOpened() and st.session_state.detection_started and not st.session_state.stop_stream:
                ret, frame = cap.read()
                if not ret:
                    status.text("No camera frame available.")
                    break
                if frame_idx % frame_skip == 0:
                    annotated, summary, suspicious_events = detector.predict_frame(frame)
                    cached_annotated = annotated
                    cached_summary = summary
                    cached_suspicious = suspicious_events
                else:
                    annotated = cached_annotated if cached_annotated is not None else frame
                    summary = cached_summary
                    suspicious_events = cached_suspicious

                # Process alerts - send to sidebar
                if alert_agent and suspicious_events:
                    for evt in suspicious_events:
                        event_data = alert_agent.process_detection(
                            evt['confidence'],
                            evt['keypoints'],
                            frame_idx,
                            frame_image=annotated  # Pass annotated frame
                        )
                        
                        # Send alert to sidebar instead of main area
                        with side:
                            st.warning(
                                f"🚨 **SUSPICIOUS ACTIVITY** @ {event_data['timestamp']}\n"
                                f"Confidence: {event_data['confidence']*100:.1f}%\n"
                                f"Action: {event_data['action']}"
                            )

                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                img_placeholder.image(rgb, use_container_width=True)

                frame_idx += 1
                now = time.time()
                elapsed = now - last_time
                if elapsed > 0:
                    status.text(f"Frame: {frame_idx} | FPS: {1/elapsed:.1f}")
                last_time = now

                time.sleep(0.01)

            cap.release()
            st.session_state.cap = None
        else:
            st.info("📹 Webcam ready. Click **Start Detection** to begin.")

with st.sidebar.expander("🚀 How to Use ShopIntel", expanded=False):
    st.markdown("""
### 🧠 1. Load the Models
- Place your YOLO model file **`best.pt`** in the project directory.
- The system automatically loads the detection and pose-estimation weights.

---

### 🎥 2. Select Video Source
Choose your input:
- 📷 **Webcam**
- 🎞️ **Local Video File**

---

### ⚙️ 3. Configure System Settings
Use the sidebar to adjust:
- 📧 Email alert settings  
- 💬 Slack alert settings  
- 📊 Alert threshold  
- 🔐 Gemini API integration (optional)  
- 🧩 Advanced detection/pose parameters (if enabled)

---

### ▶️ 4. Start Real-Time Monitoring
- Click **Start Detection** to begin.
- The system will automatically:
  - 🧍 Detect individuals  
  - 🦾 Track pose keypoints  
  - 🚨 Flag suspicious activities  
  - 📝 Log every event locally  

---

### 🔔 5. Alert Triggering
- Alerts are sent **only when the defined threshold is reached**.
- Each alert contains:
  - 🖼️ Captured frame  
  - ⏱️ Timestamp  
  - 📈 Confidence score  
  - 🤖 AI-generated Gemini guidance  

---

### 📬 6. Email Alert History
- View detailed alert history under **Email Logs**.  
- Each log entry shows:
  - Date & time  
  - Action detected  
  - Confidence level  
  - Snapshot preview  
- 🗑️ Use **Delete** to remove individual records or **Clear All** to reset.

---

### 📁 7. Suspicious Activity Log Management
- All suspicious detections are saved in **Suspicious Activity Logs**.  
- Contains:
  - Frame number  
  - Timestamp  
  - Detected action  
  - Confidence scoring  
- 🗑️ Remove individual entries or clear the full log anytime.

---

### 💬 8. Chat Assistant (AI Support)
- Access the **Chat Assistant** for:
  - 📘 Guidance on suspicious events  
  - 🧠 General support & troubleshooting  
  - 🤖 AI-generated advice using Gemini  
- Ideal for quick answers and staff training.

---

### ⏹️ 9. Stop Detection
- Click **Stop Detection** to end monitoring and release resources.

---

### 💡 Tips
- Use a **GPU-enabled system** for optimal real-time performance.  
- Maintain stable camera positions for accurate pose estimation.  
- Ensure stable internet for Gemini-guided assistance & alerting.  
""")
