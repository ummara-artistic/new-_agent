import openai
from datetime import datetime
from timezonefinder import TimezoneFinder
import geopy.geocoders
import streamlit as st
import os
import subprocess
import shutil
import sys
import yt_dlp
import streamlit as st
import speech_recognition as sr
from datetime import datetime
from pydub import AudioSegment
import tempfile
import sounddevice as sd
import webbrowser
import numpy as np
import tempfile
import os
import scipy.io.wavfile as wav

from bs4 import BeautifulSoup
import openai
import os
import moviepy.editor as mp
from pydub import AudioSegment
import datetime
import streamlit as st
import requests
from PyPDF2 import PdfReader
import json
import os
import streamlit as st
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from google.oauth2 import service_account
import datetime
import os
import shutil
import subprocess
import sys
import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st
import os

# Function to send email
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


# Set OpenAI API Key (Replace with your key)
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize session state variables if they don't exist
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []




# Create a directory for downloads
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Session state to store messages
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []


import openai
import streamlit as st
import pytz
from datetime import datetime
from timezonefinder import TimezoneFinder
import geopy.geocoders

geolocator = geopy.geocoders.Nominatim(user_agent="timezone_finder")
tf = TimezoneFinder()

def get_current_time(location):
    try:
        # Get the latitude and longitude of the location
        geo_info = geolocator.geocode(location)
        if not geo_info:
            return f"Sorry, I couldn't find the timezone for '{location}'."

        # Get timezone from coordinates
        timezone_str = tf.timezone_at(lng=geo_info.longitude, lat=geo_info.latitude)
        if not timezone_str:
            return f"Sorry, I couldn't determine the timezone for '{location}'."

        # Get the current time in that timezone
        timezone = pytz.timezone(timezone_str)
        current_time = datetime.now(timezone)

        # Format the time response
        time_str = current_time.strftime('%Y-%m-%d %H:%M:%S %Z%z')
        return f"The current time in {location.capitalize()} is {time_str}."

    except Exception as e:
        return f"An error occurred: {str(e)}"

def research_query(user_query):
    """Fetch research-based responses from OpenAI's GPT model, with time zone support."""
    if not openai.api_key:
        st.error("❌ OpenAI API key is missing.")
        return
    
    # Ensure chat_log exists in session state
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    st.session_state.chat_log.append(("🧠 You", user_query))  # Log user query

    # Check if user is asking about time in a specific place
    if any(keyword in user_query.lower() for keyword in ["what time is it", "current time in"]):
        location = user_query.split("in")[-1].strip()  # Extract location after "in"
        if location:
            answer = get_current_time(location)
            st.session_state.chat_log.append(("⏰ AI", answer))  # Log AI response
            st.write(answer)  # Display AI response in Streamlit
            return answer

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": user_query}],
            max_tokens=500
        )
        answer = response["choices"][0]["message"]["content"]
        st.session_state.chat_log.append(("🤖 AI", answer))  # Log AI response
        st.write(answer)  # Display AI response in Streamlit
        return answer
    except Exception as e:
        error_msg = f"❌ Error fetching research: {str(e)}"
        st.session_state.chat_log.append(("⚠️ Error", error_msg))
        st.error(error_msg)  # Display error in Streamlit
        return error_msg

import re


def is_ffmpeg_installed():
    """Check if ffmpeg is installed and accessible in PATH."""
    return shutil.which("ffmpeg") is not None

def extract_url(user_input):
    """Extracts the URL from the input text if it matches the expected format."""
    match = re.search(r"download this (https?://\S+)", user_input, re.IGNORECASE)
    return match.group(1) if match else None


def download_youtube_video(video_url):
    try:
        if not is_ffmpeg_installed():
            st.error("❌ Error: `ffmpeg` is not installed or not in PATH. Please install it.")
            return

        # Define temporary download directory in Streamlit Cloud
        temp_dir = "downloads"
        os.makedirs(temp_dir, exist_ok=True)

        # yt-dlp options for downloading a single video
        ydl_opts = {
            'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),  # Save as Title.mp4
            'format': 'bestvideo+bestaudio/best',  # Best available quality
            'merge_output_format': 'mp4',  # Save in MP4 format
            'noplaylist': True,  # Ensure only the single video is downloaded
            'quiet': False
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            filename = ydl.prepare_filename(info_dict).replace(".webm", ".mp4")  # Adjust extension if needed
        
        st.success("✅ Download complete!")
        
        # Provide a download button for the user
        with open(filename, "rb") as file:
            st.download_button(label="📥 Click to Download", data=file, file_name=os.path.basename(filename), mime="video/mp4")

    except Exception as e:
        st.error(f"❌ Error: {e}")


if "chat_log" not in st.session_state:
    st.session_state.chat_log = []


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
    """Check and install dependencies for different project types."""
    python_requirements = os.path.join(repo_path, "requirements.txt")
    node_package = os.path.join(repo_path, "package.json")
    dotnet_project = os.path.join(repo_path, "project.csproj")
    sql_schema = os.path.join(repo_path, "schema.sql")

    if os.path.exists(python_requirements):
        st.write("📦 Installing Python dependencies...")
        subprocess.run(["pip", "install", "-r", python_requirements], cwd=repo_path)

    if os.path.exists(node_package):
        st.write("📦 Installing Node.js dependencies...")
        subprocess.run(["npm", "install"], cwd=repo_path)

    if os.path.exists(dotnet_project):
        st.write("📦 Restoring .NET dependencies...")
        subprocess.run(["dotnet", "restore"], cwd=repo_path)

    if os.path.exists(sql_schema):
        st.write("📦 SQL schema detected. Ensure you import it into your database.")

def detect_executable(repo_path):
    """Detect the correct executable file based on project type."""
    files = os.listdir(repo_path)

    # Python
    python_files = [f for f in files if f.endswith(".py")]
    for entry in ["manage.py", "app.py", "main.py"]:
        if entry in python_files:
            return entry, "python"
    if len(python_files) == 1:
        return python_files[0], "python"

    # Node.js
    if "server.js" in files:
        return "server.js", "node"
    if "index.js" in files:
        return "index.js", "node"

    # React.js
    if "package.json" in files and os.path.isdir(os.path.join(repo_path, "src")):
        return "react", "react"

    # C
    c_files = [f for f in files if f.endswith(".c")]
    if len(c_files) == 1:
        return c_files[0], "c"

    # C++
    cpp_files = [f for f in files if f.endswith(".cpp")]
    if len(cpp_files) == 1:
        return cpp_files[0], "cpp"

    # C# (.NET)
    cs_files = [f for f in files if f.endswith(".csproj")]
    if len(cs_files) == 1:
        return cs_files[0], "dotnet"

    return None, None  # No valid file detected

def compile_and_run_c_cpp(repo_path, file_name, lang):
    """Compile and run C or C++ files."""
    binary_name = "output.exe" if os.name == "nt" else "./output"

    if lang == "c":
        compile_cmd = ["gcc", os.path.join(repo_path, file_name), "-o", os.path.join(repo_path, binary_name)]
    else:
        compile_cmd = ["g++", os.path.join(repo_path, file_name), "-o", os.path.join(repo_path, binary_name)]

    compile_process = subprocess.run(compile_cmd, capture_output=True, text=True)
    
    if compile_process.returncode == 0:
        st.write(f"✅ {lang.upper()} file compiled successfully! Running it now...")
        subprocess.run([os.path.join(repo_path, binary_name)], cwd=repo_path)
    else:
        st.write(f"❌ Compilation Error: {compile_process.stderr}")

import streamlit as st
import subprocess
import os
import sys

def run_project(repo_path, file_name=None, project_type=None):
    """Run the detected project file and persist output in the UI."""
    
    if not file_name or not project_type:
        file_name, project_type = detect_executable(repo_path)

    if not file_name:
        st.error("⚠️ No executable file found.")
        return

    st.info(f"🚀 Running `{file_name}` ({project_type})...")

    # Determine the command based on project type
    if project_type == "python":
        command = [sys.executable, os.path.join(repo_path, file_name)]
    elif project_type == "node":
        command = ["node", os.path.join(repo_path, file_name)]
    elif project_type == "c":
        command = ["gcc", os.path.join(repo_path, file_name), "-o", "output"]  # Compile first
    elif project_type == "cpp":
        command = ["g++", os.path.join(repo_path, file_name), "-o", "output"]
    elif project_type == "csharp":
        command = ["dotnet", "run"]
    else:
        st.error("❌ Unsupported project type.")
        return

    # For compiled languages, execute the compiled binary
    if project_type in ["c", "cpp"]:
        subprocess.run(command, check=True)
        command = ["./output"]  # Run compiled binary

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Initialize session state for persistent output
    if "output_text" not in st.session_state:
        st.session_state.output_text = ""

    output_window = st.empty()  # UI placeholder

    # Read output in real-time and keep it visible
    for line in iter(process.stdout.readline, ''):
        st.session_state.output_text += line  # Append new output
        output_window.text_area("🔍 Live Output:", st.session_state.output_text, height=400)  # Keep updating

    # Capture and display errors
    stderr_output = process.stderr.read()
    if stderr_output:
        st.session_state.output_text += f"\n⚠️ Execution Error:\n{stderr_output}"
        st.error("❌ Execution encountered an error.")

    # Ensure final output remains visible
    output_window.text_area("🔍 Final Output:", st.session_state.output_text, height=400)

    if not stderr_output:
        st.success("✅ Execution completed successfully!")

# **Persistent Output on Reload**
if "output_text" in st.session_state and st.session_state.output_text:
    st.text_area("🔍 Previous Output:", st.session_state.output_text, height=400)



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



def send_email(to_email, subject, body):
    from_email = "ummarakhan60@gmail.com"  # Replace with your email
    app_password = "horf ybfy fwsn czxr"  # Replace with your application-specific password

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, app_password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        return "✅ Message sent successfully!"
    except Exception as e:
        return f"❌ Error: {str(e)}"

        


# Path to the service account JSON file
load_dotenv()

# Get service account file path from environment variable
# Load the service account credentials
import os

import os
from google.oauth2 import service_account
from googleapiclient.discovery import build

import os
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build

from google.oauth2 import service_account
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os

import streamlit as st
import re
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials

# Google Calendar API Setup
# Google Calendar API Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCOPES = ["https://www.googleapis.com/auth/calendar"]

# Load credentials properly
os.path.join(BASE_DIR, "config", "D:\Ai-agent-main\Ai-agent-main\config\continual-works-430510-v1-e672c0aea3c1.json")

def schedule_event(summary, start_time):
    """
    Schedules an event in Google Calendar.
    
    :param summary: Event title
    :param start_time: Datetime object for the start time
    :return: Google Calendar event link
    """
    service = build("calendar", "v3", credentials=CREDS)
    calendar_id = "primary"

    end_time = start_time + timedelta(hours=1)  # Default duration: 1 hour
    event = {
        "summary": summary,
        "start": {"dateTime": start_time.isoformat(), "timeZone": "UTC"},
        "end": {"dateTime": end_time.isoformat(), "timeZone": "UTC"},
    }

    event_result = service.events().insert(calendarId=calendar_id, body=event).execute()
    return event_result.get("htmlLink")

def extract_time_from_message(message):
    """
    Extracts time from a message like 'Schedule this meeting at 2 PM today'.
    Returns a datetime object.
    """
    match = re.search(r"(\d{1,2})\s?(AM|PM|am|pm)?\s?(today|tomorrow)?", message, re.IGNORECASE)
    
    if match:
        hour = int(match.group(1))
        period = match.group(2)
        day_modifier = match.group(3)

        # Convert to 24-hour format
        if period and period.lower() == "pm" and hour != 12:
            hour += 12
        if period and period.lower() == "am" and hour == 12:
            hour = 0

        # Determine the date
        event_date = datetime.utcnow().date()
        if day_modifier and day_modifier.lower() == "tomorrow":
            event_date += timedelta(days=1)

        event_time = datetime(event_date.year, event_date.month, event_date.day, hour, 0)
        return event_time

    return None


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








# Main function
def load_chat_logs():
    if "chat_logs" not in st.session_state:
        st.session_state["chat_logs"] = []

# Main function
def main():
    """Main Streamlit app."""

    # Ensure API key is stored globally
    if "openai.key" not in st.session_state:
        st.session_state["openai.key"] = None

    st.sidebar.title("🔑 API & Settings")

    # API Key Input (Always Visible)
    openai_key = st.sidebar.text_input("🔐 OpenAI API Key", type="password")

    # Save API key and set globally
    if openai_key:
        st.session_state["openai.key"] = openai_key
        openai.api_key = openai_key  # ✅ Assign globally
        st.sidebar.success("✅ OpenAI API Key saved!")

    # Section: Optional Settings (Always Visible)
    st.sidebar.subheader("⚙️ Optional Settings")

    with st.sidebar.expander("📹 Zoom Integration (Optional)"):
        st.session_state["zoom_id"] = st.text_input("Zoom ID")
        st.session_state["zoom_client_id"] = st.text_input("Zoom Client ID")
        st.session_state["zoom_client_secret"] = st.text_input("Zoom Client Secret", type="password")

    with st.sidebar.expander("📧 Email Integration (Optional)"):
        st.session_state["sender_email"] = st.text_input("Sender Email")
        st.session_state["email_app_password"] = st.text_input("Email App Password", type="password")

    # If no API key, show warning and STOP execution before displaying main UI
    if not st.session_state["openai.key"]:
        st.warning("⚠️ Please enter your OpenAI API Key to access the main app features.")
        return  # Stop execution until API key is provided

    # Load chat logs
    load_chat_logs()


    
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
    user_input = st.chat_input("📝 Type your command:",key = 'user_input')

    if user_input:
        # Store user message and display instantly
        st.session_state["messages"].append({"role": "user", "text": user_input})
        st.session_state["chat_logs"].append(st.session_state["messages"].copy())
       

      
        # Process user command
        response = ""



        if user_input.lower().startswith("schedule this"):
         event_time = extract_time_from_message(user_input)
            event_time = None
            

        if event_time:
            event_details = {
                "summary": "Meeting",
                "start_time": event_time
            }
            event_link = schedule_event(event_details["summary"], event_details["start_time"])
            st.success(f"✅ Meeting scheduled successfully! [View in Google Calendar]({event_link})")
        



        elif user_input.lower().startswith("download this"):
         video_url = extract_url(user_input)
         if video_url:
            download_youtube_video(video_url)
            st.success(f"✅ Download complete! Check your Downloads folder.")




         
            
      

        elif user_input.lower().startswith("transcribe this"):
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
                    st.success(response)
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
        

if __name__ == "__main__":
    main()
