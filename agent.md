# Agent Instruction: SignLanguageDetection Pipeline

## Project Overview
A real-time American Sign Language (ASL) recognition system using a FastAPI backend, a modern web frontend, and Gemini 2.5 Flash for contextual correction and predictive typing.

## Current Progress
- `main.py`: Contains working Roboflow inference code (API Key: `Nz2Z5jwsCQBEkSutNlJS`, Model: `asl-ybz8z/2`).
- Plan approved: 2-second stability hold, Tabbed UI, Gemini integration after 2 words.

## Technical Architecture

### Backend (FastAPI)
- **WebSockets**: `/ws/stream` for binary frame processing.
- **Gesture Processor**: Logic to track confidence and duration (2s threshold).
- **LLM Helper**: Gemini 2.0/2.5 Flash (`google-genai`) to fix CV errors (P vs H, J/Z missing) and autocomplete. Uses `system_instruction` and JSON mode.
- **Templates**: Jinja2 for server-side rendering of the tabbed interface.

### Frontend (Tailwind + JS)
- **Base Template**: `templates/base.html` with Tailwind/DaisyUI.
- **Recognition View**: Live video + SVG circular progress bar + Audio feedback.
- **Interactions**: Tab key to accept AI suggestions, Space key for manual word breaks.
- **History View**: Log of previous sessions/sentences.
- **WebSocket Client**: Captures frames from `<video>` and sends to backend.

## Roadmap & Tasks
1. [x] **Setup**: Initialize directory structure (`engine/`, `static/`, `templates/`) and install dependencies.
2. [x] **Refactor Engine**: Port logic into `engine/gesture_processor.py` with stability tracking.
3. [x] **LLM Integration**: Implement `engine/llm_helper.py` with `system_instruction`.
4. [x] **Frontend Foundation**: Create `base.html` and `recognition.html` with tab navigation.
5. [x] **Real-time Pipeline**: Implement WebSocket frame handling and character commitment logic.
6. [x] **Feedback Loop**: Add audio "ping", visual progress, and "Tab to accept" suggestion.

## Key Constraints
- Gemini API key must be in `.env`.
- Use `google-genai` package.
- Trigger LLM only after 2 words are formed and a word-break is detected.
- Maintain modern "Glassmorphism" aesthetic.
