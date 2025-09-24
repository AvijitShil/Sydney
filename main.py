# main.py

import os
import json
import random
import threading
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import gradio as gr
import soundfile as sf  # for saving mic audio to wav

from TTS.api import TTS
from faster_whisper import WhisperModel
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from rag_system import MedicalRAGSystem, is_medical_emergency, get_emergency_response

from rag_system import MedicalRAGSystem, is_medical_emergency, get_emergency_response

CONFIG = {
    "memory_file": "memory.json",
    "max_memory_turns": 10,
    "audio_sample_rate": 16000,
    "medical_disclaimer": (
        "⚕️ MEDICAL DISCLAIMER: I am an AI assistant and cannot provide medical "
        "diagnosis or treatment advice. For medical emergencies, contact emergency services immediately."
    ),
    "medical_docs": [
        "Diabetes is a chronic disease affecting blood sugar levels. Symptoms include increased thirst, frequent urination, and fatigue.",
        "Hypertension (high blood pressure) can cause headaches, shortness of breath, and chest pain. Regular monitoring is essential.",
        "Common cold symptoms include runny nose, sore throat, and mild fever. Rest and hydration are important.",
        "Proper nutrition is essential for health. A balanced diet includes proteins, carbohydrates, fats, vitamins, and minerals.",
        "Regular exercise helps maintain cardiovascular health, manage weight, and improve mental wellbeing.",
        "Adequate sleep (7-9 hours for adults) is crucial for physical and mental health.",
        "Stress management techniques include deep breathing, meditation, and regular physical activity.",
        "Vaccinations help prevent serious diseases by building immunity before exposure.",
        "Mental health is as important as physical health. Common conditions include anxiety and depression.",
        "Proper handwashing is crucial for preventing the spread of infections and diseases."
    ]
}

class MedicalMemoryManager:
    def __init__(self, memory_file: str = CONFIG["memory_file"]):
        self.memory_file = memory_file
        self.conversation_history = self._load_memory()
        self.memory = ConversationBufferMemory(
            k=CONFIG["max_memory_turns"],
            return_messages=True,
        )
        self._prime_buffer()

    def _load_memory(self) -> List[Dict]:
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠ Error loading memory: {e}")
        return []

    def _save_memory(self):
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠ Error saving memory: {e}")

    def _prime_buffer(self):
        for turn in self.conversation_history[-2 * CONFIG["max_memory_turns"] :]:
            if turn["role"] == "user":
                self.memory.chat_memory.add_user_message(turn["content"])
            elif turn["role"] == "assistant":
                self.memory.chat_memory.add_ai_message(turn["content"])

    def add(self, user_text: str, ai_text: str):
        now = datetime.now().isoformat()
        self.conversation_history.append({"role": "user", "content": user_text, "timestamp": now})
        self.conversation_history.append({"role": "assistant", "content": ai_text, "timestamp": now})
        self.memory.chat_memory.add_user_message(user_text)
        self.memory.chat_memory.add_ai_message(ai_text)
        self._save_memory()

    def context_prompt(self, current_input: str) -> str:
        ctx = ""
        if self.conversation_history:
            ctx = "Recent context:\n"
            for turn in self.conversation_history[-2:]:
                role = "User" if turn["role"] == "user" else "Assistant"
                ctx += f"{role}: {turn['content']}\n"
            ctx += "\n"
        return (
            f"Current question: {current_input}\n\n"
            "You are a helpful, concise medical assistant. Be clear and friendly. "
            "Do not provide diagnoses. Encourage consulting professionals for serious concerns.\n\n"
            f"{ctx}Please respond directly to: {current_input}"
        )

# Initialize thread pool for async operations
import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# Initialize subsystems
print("🔄 Initializing medical knowledge base...")
rag_system = MedicalRAGSystem()

# Build the medical knowledge index
print("📚 Building medical knowledge index...")
try:
    rag_system.build_index(CONFIG["medical_docs"])
    print("✅ Knowledge base initialized successfully")
except Exception as e:
    print(f"⚠ Error building knowledge index: {e}")
    print("⚠ System will continue with basic functionality")

print("🔄 Loading Whisper model (small)...")  # Using small with async transcription
whisper_model = WhisperModel(
    "small",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16" if torch.cuda.is_available() else "int8",
)

print("🔄 Loading GlowTTS model...")
tts = TTS(
    model_name="tts_models/en/ljspeech/glow-tts",
    progress_bar=False,
    gpu=torch.cuda.is_available(),
)

print("🔄 Loading LLM (Gemma3:1b via Ollama)...")
try:
    llm = OllamaLLM(model="gemma3:1b", temperature=0.7)
except Exception as e:
    print(f"⚠ Could not connect to Ollama. Is ollama running? Error: {e}")
    raise

memory_manager = MedicalMemoryManager()
_last_llm_response: Optional[str] = None

# Helpers
def unique_filename(prefix="file", ext=".wav") -> str:
    return f"{prefix}_{random.randint(100000, 999999)}{ext}"

def transcribe_wav(path: str) -> str:
    segments, _ = whisper_model.transcribe(
        path, beam_size=1, language="en", condition_on_previous_text=True
    )
    return "".join(seg.text for seg in segments).strip()

def transcribe_async(path: str) -> str:
    """Run transcription in thread pool executor"""
    future = executor.submit(transcribe_wav, path)
    return future.result()

def async_transcribe(file_path: str, callback) -> None:
    """Run transcription in a background thread to prevent UI freezing"""
    def worker():
        text = transcribe_async(file_path)
        callback(text)
    threading.Thread(target=worker, daemon=True).start()

def ask_llm(user_text: str) -> str:
    global _last_llm_response

    # Emergency check
    if is_medical_emergency(user_text):
        return get_emergency_response()

    # Build context + RAG
    context_prompt = memory_manager.context_prompt(user_text)
    enhanced = rag_system.enhance_prompt_with_rag(context_prompt, user_text)

    # ---------------- Prompt Engineering ----------------
    # Instructions for LLM to return concise, clean, factual response
    prompt_engineered = (
        "You are an expert medical assistant. "
        "Provide **concise**, factual, and friendly advice. "
        "Do NOT include greetings, meta-comments, or any prompt engineering text. "
        "Do NOT add repetitive summaries. "
        "Use bullet points only if multiple points are necessary.\n\n"
        "Rules:\n"
        "1. Start directly with the key information\n"
        "2. Use **bold** for important warnings or key points\n"
        "3. No diagnoses - only general information\n"
        "4. Keep responses under 3-4 paragraphs\n"
        "5. Skip pleasantries and meta-text\n\n"
        f"{enhanced}"  # Append RAG-enhanced prompt
    )
    # -----------------------------------------------------

    try:
        resp = llm.invoke(prompt_engineered)

        # Reinvoke if response repeats previous answer
        if _last_llm_response and resp.strip() == _last_llm_response.strip():
            resp = llm.invoke(prompt_engineered + "\n\n(Please rephrase concisely.)")

        # ---------------- Post-processing ----------------
        # Remove unwanted greetings or extra instructions automatically
        lines = resp.splitlines()
        filtered_lines = []
        skip_line = False
        
        for line in lines:
            line = line.strip()
            if not line:  # skip empty lines
                continue
                
            # Skip common meta-text and greetings
            skip_patterns = [
                "okay", "here's", "hi there", "let's explore", "aiming for",
                "i understand", "i hope this helps", "hope this information",
                "let me know", "please note", "to summarize"
            ]
            
            if any(pattern in line.lower() for pattern in skip_patterns):
                skip_line = True
                continue
                
            # Skip lines that look like prompt engineering
            if any(line.lower().startswith(start) for start in ["you asked", "regarding your", "concerning the"]):
                skip_line = True
                continue
                
            if not skip_line:
                filtered_lines.append(line)
            skip_line = False
            
        resp = "\n".join(filtered_lines)

        # Clean up multiple newlines
        resp = "\n".join(line for line in resp.splitlines() if line.strip())
        # --------------------------------------------------

    except Exception as e:
        return f"⚠ LLM error: {e}\n\n{CONFIG['medical_disclaimer']}"

    # Append disclaimer if needed
    health_terms = ["health", "medical", "doctor", "medication", "symptoms", 
                   "pain", "treatment", "condition", "disease", "allergy"]
    if any(t in user_text.lower() for t in health_terms) and "medical disclaimer" not in resp.lower():
        resp += f"\n\n{CONFIG['medical_disclaimer']}"

    _last_llm_response = resp
    return resp

    health_terms = ["health", "medical", "doctor", "medication", "symptoms", "pain", "treatment"]
    if any(t in user_text.lower() for t in health_terms) and "medical disclaimer" not in resp.lower():
        resp += f"\n\n{CONFIG['medical_disclaimer']}"

    _last_llm_response = resp
    return resp

def clean_text_for_tts(text: str) -> str:
    """Clean markdown and format text for TTS"""
    # Remove markdown formatting
    text = text.replace('*', '')
    text = text.replace('**', '')
    text = text.replace('[', '')
    text = text.replace(']', '')
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('#', '')
    
    # Remove URLs
    import re
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Clean up extra whitespace
    text = ' '.join(text.split())
    return text

def chunk_text(text: str, max_chunk_size: int = 500) -> List[str]:
    """Split text into smaller chunks for TTS"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
            
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def tts_to_file(text: str, max_retries: int = 3) -> str:
    """Convert text to speech, with retries on failure"""
    # Clean the text first
    text = clean_text_for_tts(text)
    
    # Split into chunks if needed
    chunks = chunk_text(text)
    final_wav = unique_filename("response", ".wav")
    
    import wave
    import numpy as np
    
    # Process each chunk
    all_audio_data = []
    sample_width = None
    framerate = None
    
    for chunk in chunks:
        for attempt in range(max_retries):
            try:
                temp_wav = unique_filename("temp", ".wav")
                tts.tts_to_file(text=chunk, file_path=temp_wav, speed=0.9)
                
                # Read the audio data
                with wave.open(temp_wav, 'rb') as wf:
                    if sample_width is None:
                        sample_width = wf.getsampwidth()
                        framerate = wf.getframerate()
                    
                    audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                    all_audio_data.append(audio_data)
                
                # Clean up temp file
                os.remove(temp_wav)
                break
            except Exception as e:
                print(f"⚠ TTS attempt {attempt + 1} failed for chunk: {e}")
                if attempt < max_retries - 1:
                    print("Retrying TTS...")
                    continue
                raise
    
    # Combine all chunks
    combined_audio = np.concatenate(all_audio_data)
    
    # Write the final wav file
    with wave.open(final_wav, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(framerate)
        wf.writeframes(combined_audio.tobytes())
    
    return final_wav

# Gradio handlers
def handle_text(user_text: str) -> Tuple[str, Optional[str]]:
    if not user_text or not user_text.strip():
        return "⚠ Please type something.", None
    answer = ask_llm(user_text)
    memory_manager.add(user_text, answer)
    audio_path = tts_to_file(answer)
    return answer, audio_path

def handle_mic(audio_path: str) -> Tuple[str, Optional[str]]:
    """
    Handle microphone input using file path from Gradio audio component.
    Uses async transcription to prevent UI freezing.
    """
    if not audio_path:
        return "⚠ No audio received.", None
    
    try:
        # Use the recorded WAV file directly
        temp_path = audio_path

        # Set up the progress function
        progress_fn = gr.Progress()
        
        progress_fn(0, desc="Starting transcription...")
        # Transcribe audio
        try:
            result = transcribe_async(temp_path)
            if not result.strip():
                return "⚠ Couldn't understand. Please try again.", None
        except Exception as e:
            print(f"Transcription error: {e}")
            return f"⚠ Transcription failed: {str(e)}", None
        
        progress_fn(0.4, desc="Getting AI response...")
        # Get AI response
        try:
            answer = ask_llm(result)
            memory_manager.add(result, answer)
        except Exception as e:
            print(f"LLM error: {e}")
            return f"⚠ Failed to get AI response: {str(e)}", None

        progress_fn(0.7, desc="Generating speech...")
        # Convert to speech - will retry on failure
        try:
            audio_path = tts_to_file(answer)
        except Exception as e:
            print(f"TTS error after all retries: {e}")
            return "⚠ Failed to generate speech response. Please try again.", None

        progress_fn(0.9, desc="Cleaning up...")
        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception as e:
            print(f"Cleanup error: {e}")  # Non-critical error, continue
            
        progress_fn(1.0, desc="Done!")
        return answer, audio_path

    except Exception as e:
        print(f"Unexpected error: {e}")
        return f"⚠ An unexpected error occurred: {str(e)}", None

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🩺 Sydney – Your Medical Assistant")
    gr.Markdown(CONFIG["medical_disclaimer"])

    with gr.Tab("💬 Chat"):
        with gr.Row():
            user_box = gr.Textbox(
                label="Ask a health question",
                placeholder="e.g., I have a headache, what should I do?",
                lines=2,
            )
            send_btn = gr.Button("Ask")
        with gr.Row():
            with gr.Column(scale=2):
                reply_box = gr.Textbox(
                    label="Assistant reply",
                    lines=15,  # Make text box taller
                    max_lines=30,  # Allow expanding up to 30 lines
                    show_copy_button=True,  # Add copy button
                    container=True,  # Better container styling
                )
            with gr.Column(scale=1):
                audio_out = gr.Audio(label="Spoken response", type="filepath")

        send_btn.click(fn=handle_text, inputs=user_box, outputs=[reply_box, audio_out])

    with gr.Tab("🎤 Speak"):
        with gr.Row():
            mic_in = gr.Audio(
                type="filepath",
                label="Record your question (max 10 seconds)",
                format="wav",
                sources=["microphone"],
                min_length=1,
                max_length=10,
            )
            mic_btn = gr.Button("Transcribe & Answer", variant="primary")
        with gr.Row():
            with gr.Column(scale=2):
                reply_box2 = gr.Textbox(
                    label="Assistant reply",
                    lines=15,  # Make text box taller
                    max_lines=30,  # Allow expanding up to 30 lines
                    show_copy_button=True,  # Add copy button
                    container=True,  # Better container styling
                )
            with gr.Column(scale=1):
                audio_out2 = gr.Audio(label="Spoken response", type="filepath")

        mic_btn.click(
            fn=handle_mic,
            inputs=mic_in,
            outputs=[reply_box2, audio_out2],
            api_name="transcribe",
            show_progress="full"
        )

if __name__ == "__main__":
    print("🚀 Gradio starting...")
    try:
        demo.launch(debug=True)  # Added debug mode for troubleshooting
    finally:
        # Clean up thread pool
        executor.shutdown(wait=True)