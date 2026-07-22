# ASL Recognition Pipeline with Gemini Intelligence

A modern, real-time American Sign Language (ASL) recognition system that leverages Computer Vision and Large Language Models (Gemini 2.5 Flash) to provide a seamless translation experience.

## 🚀 Features
- **Real-time ASL Detection**: Powered by Roboflow inference.
- **Stability Logic**: Gestures must be held for 2 seconds to be committed, preventing flickering and accidental inputs.
- **Gemini 2.5 Intelligence**: Automatically corrects common CV misinterpretations (e.g., confusing 'P' and 'H') and autocompletes sentences using context.
- **Modern Web Interface**: Glassmorphic UI with Tailwind CSS and DaisyUI.
- **Audio-Visual Feedback**: Interactive progress indicators and audio cues when a character is typed.
- **Tabbed Experience**: Dedicated views for recognition and session history.

## 🛠️ Tech Stack
- **Backend**: FastAPI (Python)
- **Frontend**: HTML5, Tailwind CSS, JavaScript (WebSockets)
- **Computer Vision**: OpenCV, Roboflow Inference
- **LLM**: Gemini 2.5 Flash (`google-genai`)

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd SignLanguageDetection
   ```

2. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn websockets google-genai python-dotenv opencv-python supervision inference jinja2
   ```

3. **Configure Environment**:
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   ROBOFLOW_API_KEY=Nz2Z5jwsCQBEkSutNlJS
   ```

4. **Run the Application**:
   ```bash
   python app.py
   ```

## 📖 How it Works
1. **Detection**: The browser sends video frames to the FastAPI backend via WebSockets.
2. **Commitment**: If the model detects the same sign for 2 continuous seconds, the character is added to the "current word".
3. **Correction & Completion**: Once 2 words are formed, the sentence is sent to Gemini 2.5 Flash. It reviews the characters for likely typos (knowing ASL CV quirks) and suggests completions.

## 📝 License
MIT
