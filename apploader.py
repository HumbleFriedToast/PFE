import threading
import webview
import subprocess
import time

# Start Streamlit app in a separate thread
def start_streamlit():
    subprocess.Popen(["streamlit", "run", "Gui2.py","--server.headless","true"], shell=True)
    time.sleep(3)  # Give Streamlit time to start

# Launch Streamlit in background
threading.Thread(target=start_streamlit, daemon=True).start()

# Open in WebView (point to localhost)
webview.create_window(
    "Tatouage Numérique",
    "http://localhost:8501",
    width=800,
    height=650,
    frameless=True,   # <- Removes window frame (title bar and borders)
    easy_drag=False   # Optional: prevents dragging if you have no UI to grab
)

webview.start()
