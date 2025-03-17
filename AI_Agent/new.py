import openai
import streamlit as st
import os
import subprocess
import shutil
import sys
import yt_dlp

# Set OpenAI API Key (Replace with your key)
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key


# Create a directory for downloads
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Session state to store messages
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []


def research_query(user_query):
    """Fetch research-based responses from OpenAI's GPT model."""
    if not openai.api_key:
        st.error("❌ OpenAI API key is missing.")
        return

    st.session_state.chat_log.append(("🧠 You", user_query))  # Log user query

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": user_query}],
            max_tokens=500
        )
        answer = response["choices"][0]["message"]["content"]
        st.session_state.chat_log.append(("🤖 AI", answer))  # Log AI response
        return answer
    except Exception as e:
        error_msg = f"❌ Error fetching research: {str(e)}"
        st.session_state.chat_log.append(("⚠️ Error", error_msg))
        return error_msg


def download_youtube_video(url):
    """Download a YouTube video using yt_dlp."""
    try:
        ydl_opts = {
            'outtmpl': os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s'),
            'quiet': False,
            'noplaylist': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        success_msg = f"✅ Video downloaded successfully in `{DOWNLOAD_DIR}`."
        st.session_state.chat_log.append(("🎥 Download", success_msg))
        return success_msg
    except Exception as e:
        error_msg = f"❌ Error downloading video: {str(e)}"
        st.session_state.chat_log.append(("⚠️ Download Error", error_msg))
        return error_msg


import os
import shutil
import subprocess
import sys
import streamlit as st

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []
import os
import subprocess
import requests
import sys
import streamlit as st

def get_python_files_from_github(repo_url):
    """Fetch the list of .py files from the GitHub repo using GitHub API."""
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    user_name = repo_url.split("/")[-2]
    api_url = f"https://api.github.com/repos/{user_name}/{repo_name}/contents"

    response = requests.get(api_url)
    if response.status_code == 200:
        files = response.json()
        python_files = [file["name"] for file in files if file["name"].endswith(".py")]
        return python_files
    return []

def clone_repo(github_url):
    """Clone the GitHub repository and show updates in the UI."""
    repo_name = github_url.split("/")[-1].replace(".git", "")

    if os.path.exists(repo_name):
        shutil.rmtree(repo_name)  # Remove existing repo before cloning

    st.write(f"🚀 Cloning `{repo_name}`...")
    process = subprocess.run(["git", "clone", github_url], capture_output=True, text=True)

    if process.returncode == 0:
        st.write(f"✅ Repository `{repo_name}` cloned successfully!")
        return repo_name
    else:
        st.write(f"❌ Clone Error: {process.stderr}")
        return None

def install_dependencies(repo_path):
    """Check and install dependencies if found."""
    python_requirements = os.path.join(repo_path, "requirements.txt")
    node_package = os.path.join(repo_path, "package.json")

    if os.path.exists(python_requirements):
        st.write("📦 Installing Python dependencies...")
        subprocess.run(["pip", "install", "-r", python_requirements], cwd=repo_path)

    if os.path.exists(node_package):
        st.write("📦 Installing Node.js dependencies...")
        subprocess.run(["npm", "install"], cwd=repo_path)

def detect_executable(repo_path):
    """Find the correct Python file to execute."""
    python_files = [f for f in os.listdir(repo_path) if f.endswith(".py")]
    
    # Prioritize common entry points
    entry_points = ["manage.py", "app.py", "main.py"]
    for entry in entry_points:
        if entry in python_files:
            return entry, "python"

    # If there's only one Python file, run it
    if len(python_files) == 1:
        return python_files[0], "python"

    return None, None  # No valid file detected

def fix_import_error():
    """Fix ImportError related to `asyncio.coroutine` by upgrading `motor`."""
    st.write("⚠️ Fixing ImportError: Updating `motor` package...")
    subprocess.run(["pip", "install", "--upgrade", "motor"], text=True, check=True)
    st.write("✅ `motor` package updated successfully!")

def run_project(repo_path, file_name=None, project_type=None, retry=False):
    """Run the detected project file, fix errors if necessary, and display output persistently."""
    
    if not file_name or not project_type:
        file_name, project_type = detect_executable(repo_path)

    if not file_name:
        st.error("⚠️ No executable file found (e.g., `manage.py`, `app.py`, `main.py`).")
        return

    st.info(f"🚀 Running `{file_name}`...")

    command = [sys.executable, os.path.join(repo_path, file_name)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    output_text = ""  # Store all output for persistent display
    output_window = st.empty()  # Create a placeholder for the live output

    # Stream real-time output
    for line in process.stdout:
        output_text += line
        output_window.text_area("🔍 Live Output:", output_text, height=300)  # Display all output

    stderr_output = process.stderr.read()
    
    if stderr_output:
        output_text += f"\n⚠️ Execution Error:\n{stderr_output}"
        output_window.text_area("🔍 Live Output:", output_text, height=300)
        st.error("❌ Execution encountered an error.")
        
        # Auto-fix ImportError for `asyncio.coroutine`
        if "ImportError: cannot import name 'coroutine' from 'asyncio'" in stderr_output and not retry:
            fix_import_error()
            st.info("♻️ Retrying execution...")
            return run_project(repo_path, file_name, project_type, retry=True)  # Retry once

    # Ensure the final output remains visible before success message
    output_window.text_area("🔍 Final Output:", output_text, height=300)
    st.success("✅ Execution completed successfully!")  # Show success message at the end


import streamlit as st
import openai
import os

# Function to generate the dashboard code based on the user input (e.g., a finance dashboard)
def generate_dashboard_code(description):
    """Generate a dynamic code snippet for a dashboard based on description using OpenAI."""
    try:
        prompt = f"Generate code for a {description} dashboard that shows relevant data, including visualizations like bar graphs, pie charts, line graphs, and tables. The dashboard should be responsive, clean, and have interactive components where necessary. It should include features relevant to {description} like sales data, user analytics, stock prices, etc."

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use the new gpt-3.5-turbo model
            messages=[
                {"role": "system", "content": "You are an AI that generates code."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.5
        )

        code = response['choices'][0]['message']['content']
        return code.strip()
    
    except Exception as e:
        return f"❌ Error generating code: {str(e)}"

# Function to generate the dashboard image
def generate_dashboard_image(description, code_snippet):
    """Generate a dynamic dashboard image using OpenAI based on the user's description and generated code."""
    try:
        # Keep the prompt length under 1000 characters by truncating both description and code
        description = description[:500]  # Shorten description if needed
        code_snippet = code_snippet[:500]  # Shorten the code snippet if necessary
        
        # Create the prompt
        prompt = f"Create a visual image of a {description} dashboard based on the following code: {code_snippet}."
        
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        
        image_url = response['data'][0]['url']
        return image_url
    except Exception as e:
        return f"❌ Error generating dashboard image: {str(e)}"


import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st
import os

# Function to send email
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email(to_email, subject, body):
    from_email = "ummarakhan60@gmail.com"  # Replace with your email
    app_password = "horf ybfy fwsn czxr"  # Replace with your application-specific password

    # Setting up the MIME
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Establishing the connection with the SMTP server
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)  # You can use another SMTP server if needed
        server.starttls()  # Secure the connection
        server.login(from_email, app_password)  # Login using the app-specific password
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()  # Logout from the server
        return "Message sent successfully!"
    except Exception as e:
        return f"Error: {str(e)}"
        
import datetime
import json
import os
import streamlit as st
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from google.oauth2 import service_account
import datetime


# Path to the service account JSON file
load_dotenv()

# Get service account file path from environment variable
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")

# Authenticate using the service account
def authenticate_google_account():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=['https://www.googleapis.com/auth/calendar']
    )
    return build('calendar', 'v3', credentials=creds)

# Function to schedule an event
def schedule_event(event_details):
    service = authenticate_google_account()

    event = {
        'summary': event_details['title'],
        'location': event_details.get('location', ''),
        'description': event_details.get('description', ''),



        'start': {
            'dateTime': event_details['start_time'].isoformat(),
            'timeZone': 'America/Los_Angeles',
        },
        'end': {
            'dateTime': event_details['end_time'].isoformat(),
            'timeZone': 'America/Los_Angeles',
        },
    }

    try:
        event_result = service.events().insert(
            calendarId='primary', body=event
        ).execute()

        print(f"✅ Event '{event_result['summary']}' added to Google Calendar on {event_result['start']['dateTime']}")
        return f"✅ Event '{event_details['title']}' added successfully!"
    except Exception as e:
        print(f"❌ Error scheduling event: {str(e)}")
        return f"❌ Error: {str(e)}"



from bs4 import BeautifulSoup






import streamlit as st
import requests
from PyPDF2 import PdfReader

def extract_text_from_url(url):
    """Fetch and extract text content from a given URL."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        article_text = " ".join([p.get_text() for p in paragraphs])
        return article_text if len(article_text) > 100 else None  # Avoid empty/short content
    except Exception as e:
        return f"Error fetching URL: {str(e)}"

def summarize_with_openai(text):
    """Use OpenAI API to summarize the extracted text."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI that summarizes articles concisely."},
                {"role": "user", "content": f"Summarize this text:\n{text}"}
            ],
            max_tokens=200
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error summarizing text: {str(e)}"

def summarize_document(url=None, file_path=None):
    """Fetch text from a URL or file and summarize using OpenAI."""
    if url:
        text = extract_text_from_url(url)
    elif file_path:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
    else:
        return "❌ No valid input provided."

    if text:
        return summarize_with_openai(text)
    return "❌ Unable to extract meaningful text."


import streamlit as st
import openai
import os
import moviepy.editor as mp
from pydub import AudioSegment

displayed_messages =[]

# Set upload directory
UPLOAD_FOLDER = "uploads"
AUDIO_FOLDER = "audio"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

def save_uploaded_file(uploaded_file):
    """Save uploaded file and return its path."""
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return file_path

def extract_audio(video_path):
    """Extract audio from a video file and return the audio path."""
    audio_path = os.path.join(AUDIO_FOLDER, os.path.splitext(os.path.basename(video_path))[0] + ".mp3")
    
    try:
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        return audio_path
    except Exception as e:
        return f"❌ Error extracting audio: {str(e)}"

def transcribe_audio(file_path):
    """Transcribe the audio file using OpenAI Whisper."""
    try:
        response = openai.Audio.transcribe(
            model="whisper-1",
            file=open(file_path, "rb")
        )
        return response.get("text", "❌ No transcription available.")
    except Exception as e:
        return f"❌ Error transcribing: {str(e)}"

import streamlit as st
import os
import datetime



import streamlit as st
import json
import os
from datetime import datetime

CHAT_LOG_FILE = "chat_logs.txt"

def save_chat_logs():
    """Save chat logs to a file."""
    with open(CHAT_LOG_FILE, "w") as f:
        json.dump(st.session_state["chat_logs"], f)

def load_chat_logs():
    """Load chat logs from a file."""
    if os.path.exists(CHAT_LOG_FILE):
        with open(CHAT_LOG_FILE, "r") as f:
            try:
                st.session_state["chat_logs"] = json.load(f)
            except json.JSONDecodeError:
                st.session_state["chat_logs"] = []
    else:
        st.session_state["chat_logs"] = []



import speech_recognition as sr
from datetime import datetime
from pydub import AudioSegment
import tempfile
import sounddevice as sd
import numpy as np
import tempfile
import os
import scipy.io.wavfile as wav

def transcribe_audio_openai(file_path):
    """Transcribes audio using OpenAI's Whisper API."""
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]

def record_and_transcribe(duration=5, samplerate=44100):
    """Records audio using sounddevice and transcribes it using OpenAI's API."""

    # Check if a microphone is available
    try:
        input_devices = sd.query_devices()
        input_device_index = sd.default.device[0]  # Get default input device
        if input_device_index is None or input_device_index < 0:
            st.error("⚠️ No microphone detected. Please check your audio settings.")
            return None
    except Exception:
        st.error("⚠️ No microphone detected. Please check your audio settings.")
        return None

    st.info(f"🎤 Speak now... Recording for {duration} seconds...")

    try:
        # Record audio
        audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished
        st.success("✅ Recorded! Transcribing...")

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            wav.write(temp_audio.name, samplerate, audio_data)
            temp_audio_path = temp_audio.name

        # Transcribe using OpenAI Whisper
        transcription = transcribe_audio_openai(temp_audio_path)
        os.remove(temp_audio_path)  # Clean up the temporary file
        return transcription

    except Exception as e:
        return f"⚠️ Error in transcription: {e}"








def main():
    """Main Streamlit app."""
    st.set_page_config(layout="wide")
    st.title("🎙️ AI Transcription Assistant")

    # Load chat history from file
    if "chat_logs" not in st.session_state:
        load_chat_logs()

    st.sidebar.title("📜 Chat History")


    # Speech-to-Text Button
    if st.button("🎤 Click to Speak"):
        transcribed_text = record_and_transcribe()
        if transcribed_text:
            st.session_state["messages"].append({"role": "user", "text": transcribed_text})
            st.session_state["messages"].append({"role": "ai", "text": f"Transcribed: {transcribed_text}"})
            st.rerun()

    # Display chat logs in sidebar (time-wise)
    for idx, log in enumerate(st.session_state["chat_logs"]):
        timestamp = log[-1].get("timestamp", "Unknown Time") if log else "Unknown Time"

        with st.sidebar.expander(f"Chat {idx+1} - {timestamp}"):
            for msg in log:
                role = "🧠 You:" if msg["role"] == "user" else "🤖 AI:"
                st.write(f"{role} {msg['text']}")
            if st.button(f"Open Chat {idx+1}", key=f"open_chat_{idx}"):
                st.session_state["messages"] = log
                st.rerun()

    st.sidebar.markdown("---")

    if st.sidebar.button("➕ New Chat"):
        st.session_state["messages"] = []
        st.rerun()

    if st.sidebar.button("🔄 Restart"):
        st.session_state.clear()
        load_chat_logs()  # Reload logs after clearing session
        st.rerun()

    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state["messages"] = []
        st.rerun()

    if st.sidebar.button("💾 Save Chat"):
        if "messages" in st.session_state and st.session_state["messages"]:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for msg in st.session_state["messages"]:
                msg["timestamp"] = timestamp
            st.session_state["chat_logs"].append(st.session_state["messages"])
            save_chat_logs()
            st.sidebar.success("Chat saved!")

    if st.sidebar.button("📂 Load Last Chat"):
        if st.session_state["chat_logs"]:
            st.session_state["messages"] = st.session_state["chat_logs"][-1]
            st.rerun()

    # Load latest chat session automatically
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    uploaded_file = st.file_uploader(
        "📂 Upload an audio or video file", 
        type=["mp3", "wav", "m4a", "mp4", "mov", "avi", "flac", "mpeg4"]
    )
    if uploaded_file:
        file_path = save_uploaded_file(uploaded_file)
        st.session_state["last_uploaded_file"] = file_path
        st.success(f"✅ File saved: {file_path}")
    
    # Chat container to display conversation history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state["messages"]:
            if message["role"] == "user":
                st.markdown(
                    f"<div style='text-align: right; background-color: #d4edda; padding: 10px; border-radius: 10px; margin: 5px 0; width: fit-content; float: right;'>"
                    f"<strong>🧠 You:</strong> {message['text']}</div><div style='clear:both;'></div>", 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='text-align: left; background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin: 5px 0; width: fit-content; float: left;'>"
                    f"<strong>🤖 AI:</strong> {message['text']}</div><div style='clear:both;'></div>", 
                    unsafe_allow_html=True
                )
    
    




    # User input
    user_input = st.chat_input("📝 Type your command:")

    if user_input:
        # Store user message and display instantly
        st.session_state["messages"].append({"role": "user", "text": user_input})
        st.session_state["chat_logs"].append(st.session_state["messages"].copy())
       

      
        # Process user command
        response = ""
        if user_input.lower().startswith("transcribe this"):
            file_path = st.session_state.get("last_uploaded_file")
            if file_path:
                if file_path.endswith((".mp4", ".mov", ".avi", ".mpeg4")):
                    st.info("🔄 Extracting audio from video...")
                    file_path = extract_audio(file_path)  # Convert video to audio
                
                if file_path and os.path.exists(file_path):
                    response = transcribe_audio(file_path)
                else:
                    response = "⚠️ No valid file found after extraction."
            else:
                response = "⚠️ No file uploaded. Please upload an audio/video file first."

        elif user_input.lower().startswith("send msg to "):
            email_part = user_input[len("send msg to "):].strip()
            if "@" in email_part and email_part.count(" ") > 0:
                email_address, message = email_part.split(' ', 1)
                with st.spinner(f"⏳ Sending message to {email_address}..."):
                    response = send_email(email_address, "Your Subject Here", message)
            else:
                response = "Invalid email format or missing message."

        elif user_input.lower().startswith("execute this "):
            github_url = user_input[len("execute this "):].strip()
            if github_url.startswith("https://github.com/") and github_url.endswith(".git"):
                with st.spinner("⏳ Cloning repository..."):
                    repo_name = clone_repo(github_url)
                    if repo_name:
                        with st.spinner("🔍 Detecting project files..."):
                            repo_path = os.path.abspath(repo_name)
                            file_name, project_type = detect_executable(repo_path)
                        if file_name:
                            with st.spinner("⚙️ Installing dependencies..."):
                                install_dependencies(repo_path)
                            with st.spinner("🚀 Running project..."):
                                run_project(repo_path)
                            response = "✅ Project executed successfully."
                        else:
                            response = "⚠️ No valid execution command found."
                    else:
                        response = "❌ Error cloning repository."
            else:
                response = "⚠️ Invalid GitHub URL."

        elif user_input.lower().startswith("download this "):
            video_url = user_input[len("download this "):].strip()
            with st.spinner("⏳ Downloading..."):
                response = download_youtube_video(video_url)

        elif user_input.lower().startswith("create dashboard"):
            dashboard_description = user_input[len("create dashboard"):].strip() or "sales data"
            with st.spinner("⏳ Generating dashboard code..."):
                code_snippet = generate_dashboard_code(dashboard_description)
            if code_snippet:
                with st.spinner("⏳ Generating dashboard image..."):
                    dashboard_image_url = generate_dashboard_image(dashboard_description, code_snippet)
                    st.image(dashboard_image_url, caption="Generated Dashboard Image", use_column_width=True)
                    st.subheader("📚 Generated Code")
                    st.code(code_snippet, language="python")
                response = "✅ Dashboard generated successfully."
            else:
                response = "❌ Error generating dashboard code."

        elif user_input.lower().startswith("schedule this"):
            event_details = {
    "title": "Meeting with team",
    "start_time": datetime.now() + timedelta(hours=3),
    "end_time": datetime.now() + timedelta(hours=4),
    "location": "Zoom",
    "description": "Discussing project updates."
}

            response = schedule_event(event_details)

        elif user_input.lower().startswith("summarize this"):
            content_to_summarize = user_input[len("summarize this"):].strip()
            
            if content_to_summarize.startswith("http"):
                response = summarize_document(url=content_to_summarize)
            elif len(content_to_summarize) > 0:
                response = summarize_document(input_text=content_to_summarize)
            else:
                uploaded_file = st.file_uploader("Choose a file to summarize", type=["txt", "pdf"])
                if uploaded_file:
                    response = summarize_document(file=uploaded_file)
                else:
                    response = "⚠️ No content provided to summarize."

        else:
            with st.spinner("⏳ Thinking..."):
                response = research_query(user_input)

        # Store AI response
        if response:
            st.session_state["messages"].append({"role": "ai", "text": response})

        # Rerun the app to update UI instantly
        st.rerun()

if __name__ == "__main__":
    main()

