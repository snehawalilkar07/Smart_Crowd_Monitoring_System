import streamlit as st
import cv2, torch, torch.nn as nn, torchvision.models as models, numpy as np, pandas as pd, tempfile, time, os
from datetime import datetime
from email.message import EmailMessage
import smtplib, ssl
from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env file

# =================== PAGE CONFIG ===================
st.set_page_config("🚦 DeepVision Crowd Monitor", layout="wide")
st.title("🚦 DeepVision : Smart Crowd Monitoring System")

# =================== SESSION STATE ===================
for key, val in {
    "running": False,
    "cap": None,
    "counts": [],
    "alert_sent": False,
    "last_frames": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# =================== SIDEBAR ===================
st.sidebar.header("⚙️ Monitoring Settings")

# --- Video Input ---
st.sidebar.subheader("📹 Video Input")
video_file = st.sidebar.file_uploader("Upload a video file", type=["mp4","avi"])

# --- Alert Settings ---
st.sidebar.subheader("🚨 Alert Settings")
threshold = st.sidebar.slider("Crowd Alert Threshold", 5, 1000, 100,
                              help="Trigger alert when crowd count exceeds this value")

# --- Processing Settings ---
st.sidebar.subheader("⚡ Processing Settings")
fps_slider = st.sidebar.slider("Processing FPS", 1, 30, 5,
                               help="Frames per second to process (higher FPS = heavier CPU)")

alert_email = st.sidebar.text_input("Alert Email", placeholder="you@example.com",
                                    help="Optional: Email to receive alerts")

# --- How-to-use ---
with st.expander("ℹ️ How to use this system", expanded=True):
    st.markdown("""
    1. Upload a video.  
    2. Set alert threshold and FPS.  
    3. Enter alert email (optional).  
    4. Click 'Start Monitoring' on main screen to begin.  
    5. Click 'Stop Monitoring' to pause.  
    6. Original and Overlay frames appear after monitoring starts.
    """)

# =================== START / STOP BUTTONS ===================
st.markdown("### ▶ Controls")
col1, col2 = st.columns(2)
with col1:
    start_btn = st.button("Start Monitoring")
with col2:
    stop_btn = st.button("Stop Monitoring")

status_msg = st.empty()  # status below buttons

# =================== MODEL ===================
class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.frontend = nn.Sequential(*list(vgg.features)[:23])
        self.backend = nn.Sequential(
            nn.Conv2d(512,512,3,padding=2,dilation=2), nn.ReLU(),
            nn.Conv2d(512,512,3,padding=2,dilation=2), nn.ReLU(),
            nn.Conv2d(512,512,3,padding=2,dilation=2), nn.ReLU(),
            nn.Conv2d(512,256,3,padding=2,dilation=2), nn.ReLU(),
            nn.Conv2d(256,128,3,padding=2,dilation=2), nn.ReLU(),
            nn.Conv2d(128,64,3,padding=2,dilation=2), nn.ReLU()
        )
        self.output_layer = nn.Conv2d(64,1,1)
    def forward(self, x):
        return self.output_layer(self.backend(self.frontend(x)))

csr_model = CSRNet()
csr_model.load_state_dict(torch.load("model_fp16.pth", map_location="cpu"))
csr_model.eval()

# =================== ALERT EMAIL FUNCTION ===================
def send_alert_email(count, frame):
    EMAIL = os.getenv("ALERT_EMAIL")
    PASS = os.getenv("ALERT_PASSWORD")
    if not EMAIL or not PASS or not alert_email:
        return
    path = f"alert_{int(time.time())}.jpg"
    cv2.imwrite(path, frame)
    msg = EmailMessage()
    msg["Subject"] = "🚨 CROWD ALERT"
    msg["From"] = EMAIL
    msg["To"] = alert_email
    msg.set_content(f"Crowd Count: {count}\nTime: {datetime.now()}")
    with open(path,"rb") as f:
        msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename=path)
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com",465,context=ctx) as s:
        s.login(EMAIL,PASS)
        s.send_message(msg)

# =================== START MONITORING ===================
if start_btn:
    if not video_file:
        st.warning("Please upload a video first.")
    else:
        t = tempfile.NamedTemporaryFile(delete=False)
        t.write(video_file.read())
        st.session_state.cap = cv2.VideoCapture(t.name)
        st.session_state.running = True
        st.session_state.alert_sent = False
        st.session_state.counts = []
        st.session_state.last_frames = None
        status_msg.success("🚀 Monitoring started!")

# =================== STOP MONITORING ===================
if stop_btn:
    st.session_state.running = False
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None
    status_msg.info("⏹ Monitoring stopped")

# =================== PLACEHOLDERS (always defined) ===================
frames_col = st.columns([1,1])
with frames_col[0]:
    st.markdown("### 🎥 Original Frame")
    frame_original = st.empty()
with frames_col[1]:
    st.markdown("### 🌡️ Overlay + Heatmap")
    frame_overlay = st.empty()

metrics_col = st.columns(3)
with metrics_col[0]: total_alerts_box = st.empty()
with metrics_col[1]: avg_count_box = st.empty()
with metrics_col[2]: peak_count_box = st.empty()

status_col = st.columns([1,1])
with status_col[0]: count_box = st.empty()
with status_col[1]: alert_box = st.empty()

chart_box = st.empty()

# =================== VIDEO PROCESSING ===================
if st.session_state.running and st.session_state.cap:
    cap = st.session_state.cap
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.session_state.running = False
            cap.release()
            break

        # Prepare input
        img = cv2.resize(frame, (640,480))
        x = torch.tensor(img).permute(2,0,1).unsqueeze(0)/255
        with torch.no_grad():
            density_map = csr_model(x)
            count = float(density_map.sum().item())
        st.session_state.counts.append(count)

        # ALERT logic
        if count > threshold:
            alert_text = "⚠️ CROWD ALERT"
            color_bgr = (0,0,255)  # red for overlay
            color_html = "255,0,0" # for HTML
            if not st.session_state.alert_sent:
                send_alert_email(count, frame)
                st.session_state.alert_sent = True
        else:
            alert_text = "SAFE"
            color_bgr = (0,255,0)  # green for overlay
            color_html = "0,255,0"
            st.session_state.alert_sent = False

        # OVERLAY + HEATMAP
        density = density_map[0,0].cpu().numpy()
        heatmap = density / (density.max() + 1e-6)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        # Dynamic count text
        cv2.putText(
            overlay,
            f"Count: {int(count)}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color_bgr,
            2
        )

        # UPDATE UI
        frame_original.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        frame_overlay.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), channels="RGB")
        count_box.metric("👥 Crowd Count", int(count))
        alert_box.markdown(f"<h3 style='color:rgb({color_html})'>{alert_text}</h3>", unsafe_allow_html=True)
        chart_box.line_chart(pd.DataFrame(st.session_state.counts, columns=["Count"]))
        total_alerts_box.metric("🚨 Total Alerts", int(sum(np.array(st.session_state.counts) > threshold)))
        avg_count_box.metric("📈 Avg Crowd", int(np.mean(st.session_state.counts)))
        peak_count_box.metric("🔥 Peak Crowd", int(np.max(st.session_state.counts)))

        # Save last frames
        st.session_state.last_frames = (frame, overlay)
        time.sleep(1/fps_slider)

# =================== PRESERVE LAST FRAME ===================
if not st.session_state.running and st.session_state.last_frames:
    o, ov = st.session_state.last_frames
    frame_original.image(cv2.cvtColor(o, cv2.COLOR_BGR2RGB), channels="RGB")
    frame_overlay.image(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB), channels="RGB")
