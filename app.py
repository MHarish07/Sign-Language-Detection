import os
import json
import base64
import cv2
import numpy as np
import asyncio
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

from engine.gesture_processor import GestureProcessor
from engine.llm_helper import LLMHelper

load_dotenv()

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize Engine
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "Nz2Z5jwsCQBEkSutNlJS")
MODEL_ID = "asl-ybz8z/2"
processor = GestureProcessor(model_id=MODEL_ID, api_key=ROBOFLOW_API_KEY)
llm = LLMHelper()

# App State
state = {
    "raw_sentence": "",
    "corrected_sentence": "",
    "history": [],
    "current_llm_task": None
}

async def run_llm_task(raw_sentence, websocket):
    try:
        # Minimal debounce
        await asyncio.sleep(0.4)
        
        # Trigger LLM if we have any content (removed 2-word constraint)
        if not raw_sentence.strip():
            return

        llm_result = await llm.process_sentence(raw_sentence)
        corrected = llm_result.get("corrected", "")
        state["corrected_sentence"] = corrected
        
        await websocket.send_text(json.dumps({
            "type": "update",
            "llm_suggestion": llm_result.get("suggestion", ""),
            "llm_corrected": corrected,
            "full_sentence": raw_sentence
        }))
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Async LLM Task Error: {e}")

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("recognition.html", {"request": request, "active_tab": "recognition"})

@app.get("/history", response_class=HTMLResponse)
async def get_history(request: Request):
    return templates.TemplateResponse("history.html", {"request": request, "active_tab": "history", "history": state["history"]})

@app.get("/learn", response_class=HTMLResponse)
async def get_learn(request: Request):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return templates.TemplateResponse("learn.html", {"request": request, "active_tab": "learn", "alphabet": alphabet})

@app.get("/practice", response_class=HTMLResponse)
async def get_practice(request: Request, char: str = "", mode: str = "random"):
    return templates.TemplateResponse("practice.html", {
        "request": request, 
        "active_tab": "practice", 
        "initial_char": char,
        "mode": mode
    })

@app.get("/api/practice/paragraph")
async def get_practice_paragraph():
    prompt = "Generate a short, simple paragraph (2-3 sentences) for ASL finger-spelling practice. Use simple words. CRITICAL: Do not use the letters 'J' or 'Z' at all. Return only the paragraph text."
    try:
        # We'll use the existing LLM helper but with a direct prompt
        text = await llm.generate_text(prompt)
        text = text.strip().upper()
        # Double check exclusion
        text = text.replace('J', '').replace('Z', '')
        return {"paragraph": text}
    except Exception as e:
        print(f"LLM Paragraph Error: {e}")
        return {"paragraph": "THE QUICK BROWN FOX FALLS OVER THE LAZY DOG".replace('J', '').replace('Z', '').upper()}

@app.get("/api/practice/levels")
async def get_practice_levels():
    levels = [
        "CAT", "DOG", "FISH", "BIRD", "HOME",
        "HELLO WORLD", "I LIKE ASL", "APPLES ARE RED",
        "THE SUN IS BRIGHT", "KEEP PRACTICING HARD",
        "WATER IS CLEAR AND COOL", "FAST LEARNING IS FUN"
    ]
    # Filter out J and Z just in case
    filtered_levels = [l.upper().replace('J', '').replace('Z', '') for l in levels]
    return {"levels": filtered_levels}

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket, mode: str = "recognition"):
    await websocket.accept()
    print(f"WebSocket connected (mode: {mode})")
    
    try:
        while True:
            message = await websocket.receive()
            
            if "bytes" in message:
                image_bytes = message["bytes"]
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                    
                detections, gesture, committed, progress = processor.process_frame(frame)
                
                if committed and gesture and mode == "recognition":
                    char = str(gesture)
                    if char.lower() == "space": 
                        char = " "
                    
                    state["raw_sentence"] += char
                    
                    # Immediate Feedback
                    await websocket.send_text(json.dumps({
                        "type": "update",
                        "current_gesture": gesture,
                        "progress": progress,
                        "committed": True,
                        "char": char,
                        "full_sentence": state["raw_sentence"]
                    }))
                    
                    # Async Gemini logic
                    if state["current_llm_task"] and not state["current_llm_task"].done():
                        state["current_llm_task"].cancel()
                    
                    state["current_llm_task"] = asyncio.create_task(run_llm_task(state["raw_sentence"], websocket))
                else:
                    # In practice mode, we still send "committed" but don't update global state
                    # We just send the update to the client
                    resp = {
                        "type": "update",
                        "detections": detections,
                        "current_gesture": gesture,
                        "progress": progress,
                        "committed": committed
                    }
                    if committed:
                        resp["char"] = gesture
                    
                    if mode == "recognition":
                        resp["full_sentence"] = state["raw_sentence"]
                        
                    await websocket.send_text(json.dumps(resp))
                
            elif "text" in message:
                data = json.loads(message["text"])
                
                if data['type'] == 'apply_correction':
                    if state["corrected_sentence"]:
                        state["raw_sentence"] = state["corrected_sentence"]
                        state["corrected_sentence"] = "" # Clear after applying
                        await websocket.send_text(json.dumps({
                            "type": "update",
                            "full_sentence": state["raw_sentence"],
                            "llm_corrected": "Awaiting input...", # Reset visual
                            "committed": True,
                            "char": ""
                        }))
                        if state["current_llm_task"] and not state["current_llm_task"].done():
                            state["current_llm_task"].cancel()
                        state["current_llm_task"] = asyncio.create_task(run_llm_task(state["raw_sentence"], websocket))

                elif data['type'] == 'accept_suggestion':
                    suggestion = data.get('suggestion', '')
                    if suggestion:
                        state["raw_sentence"] = state["raw_sentence"].strip() + " " + suggestion + " "
                        await websocket.send_text(json.dumps({
                            "type": "update",
                            "full_sentence": state["raw_sentence"],
                            "llm_suggestion": "", # Clear visual
                            "committed": True,
                            "char": ""
                        }))
                        if state["current_llm_task"] and not state["current_llm_task"].done():
                            state["current_llm_task"].cancel()
                        state["current_llm_task"] = asyncio.create_task(run_llm_task(state["raw_sentence"], websocket))

                elif data['type'] == 'force_char':
                    char = data.get('char', ' ')
                    state["raw_sentence"] += char
                    await websocket.send_text(json.dumps({
                        "type": "update",
                        "full_sentence": state["raw_sentence"],
                        "committed": True,
                        "char": char
                    }))
                    if state["current_llm_task"] and not state["current_llm_task"].done():
                        state["current_llm_task"].cancel()
                    state["current_llm_task"] = asyncio.create_task(run_llm_task(state["raw_sentence"], websocket))

                elif data['type'] == 'reset':
                    state["raw_sentence"] = ""
                    state["corrected_sentence"] = ""
                    processor.reset_stability()
                    await websocket.send_text(json.dumps({
                        "type": "update",
                        "full_sentence": "",
                        "committed": True,
                        "char": ""
                    }))
                elif data['type'] == 'save':
                    if state["raw_sentence"]:
                        state["history"].append(state["raw_sentence"])
                        state["raw_sentence"] = ""
                        state["corrected_sentence"] = ""
                    await websocket.send_text(json.dumps({"type": "save_success"}))

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = "127.0.0.1"
    print("\n" + "="*60)
    print(f"🚀 ASL Pipeline Server Starting...")
    print(f"📱 Open your browser and go to:")
    print(f"👉 http://{host}:{port}")
    print("="*60 + "\n")
    uvicorn.run(app, host=host, port=port, log_level="info")
