# 🩺 Sydney – The Ultimate Local (Offline) AI Medical Assistant

![Sydney](https://img.shields.io/badge/status-Alpha-yellow)

**Sydney** is an **advanced multimodal Local (offline) AI assistant** capable of **understanding text and voice**, delivering **highly accurate, context-aware medical guidance**, and working **fully offline with CPU optimization**. Powered by **Gemma 3 !B, Glow-TTS, Whisper, LangChain LLMs, and the Granite 47M R2 embedding model**, Sydney merges **retrieval-augmented generation (RAG)**, **conversation memory**, and **multimodal interaction** to provide **rapid, reliable responses**.

⚠️ **Medical Disclaimer:** Sydney is **not a substitute for professional medical advice**. For emergencies, always contact certified healthcare providers.

---

## **🚀 Key Highlights**

* 🧠 **Multimodal**: Supports both **text** and **voice input/output** in the same version.
* 📚 **Offline-first**: Entirely functional without internet once models and embeddings are downloaded.
* ⚡ **CPU-optimized**: Fast inference even on standard CPUs without compromising response quality.
* 🛠 **Context-aware memory**: Stores user queries and AI responses, allowing nuanced conversations.
* 🔍 **Multiple RAG pipelines**: Integrates **local medical documents** and **Granite 47M R2 embeddings** for factual, precise, and comprehensive answers.
* 💡 **Problem-solving powerhouse**: Capable of **complex multi-topic medical reasoning**, combining retrieval and generative capabilities.
* 🎯 **Clean & structured outputs**: Markdown removal, concise formatting, and speech-ready text.
* 🚨 **Emergency detection**: Flags urgent situations and provides immediate guidance to contact professionals.

---

## **🧩 Multimodal & Offline Architecture**

Sydney is designed as a **unified multimodal AI assistant**:

```
                +--------------------+
                |   User Input       |
                |  Text / Voice      |
                +---------+----------+
                          |
                  [Whisper Speech-to-Text]
                          |
        +-----------------+-----------------+
        |                                   |
[Memory Manager]                     [RAG Pipelines]
- Tracks last N turns               - Local medical docs
- Maintains context                 - Granite 47M R2 embeddings
- Preserves conversation            - Combines results for reasoning
        |                                   |
        +-----------------+-----------------+
                          |
                   [LLM Processing]
                  - Concise, context-aware
                  - Multi-topic reasoning
                          |
                 [Post-processing & Cleanup]
                  - Markdown removal
                  - Bullet point formatting
                          |
                  [Glow-TTS Speech Synthesis]
                  - Natural, expressive audio
                          |
                 +---------------------+
                 | Gradio UI / Output  |
                 | Text + Voice        |
                 +---------------------+
```

---

## **💾 Memory & Context Management**

Sydney’s **conversation memory** is **persistent, intelligent, and context-aware**:

* Stores **queries and AI responses** in `memory.json`.
* Limits memory to **configurable recent turns** (default: 10) for performance.
* Each query is augmented with **recent conversation context** for:

  * Coherent multi-turn dialogue
  * Avoiding repetition
  * Tailored answers based on user history

**Example Context-Aware Query:**

```
Recent context:
User: I have diabetes and high blood pressure. Can I exercise daily?
Assistant: Light cardio 3-5 times/week, strength training, and regular monitoring of blood sugar levels.

Current query: What dietary changes should I implement alongside exercise?
Assistant: - Low glycemic index foods, high fiber intake...
           - Reduce sodium and processed foods...
           - Maintain protein balance for muscle health...
```

---

## **📚 Multiple RAG Pipelines for Robust Knowledge Retrieval**

Sydney leverages **multiple RAG systems** for precise answers:

1. **Local Medical Knowledge Base RAG**

   * Curated offline documents on diseases, symptoms, nutrition, and lifestyle.
2. **Granite 47M R2 Embeddings RAG**

   * Embedding-based vector search for **fast, semantic retrieval** of relevant information.
   * Handles **rare, multi-faceted medical queries** with high accuracy.

**Benefits:**

* Multi-topic problem-solving
* Context-enhanced recommendations
* Factually grounded AI responses

---

## **💨 Offline & CPU-Optimized Performance**

Sydney is engineered for **offline, CPU-friendly operation**:

* **Glow-TTS + Multi-band MelGAN** ensures smooth text-to-speech without GPU.
* **Whisper small/int8** allows fast audio transcription on CPU.
* **Precomputed Granite embeddings** for instant retrieval.
* **Chunking long responses** prevents CPU overload.
* **Async threading** ensures non-blocking, fast UI.
* **Persistent memory caching** reduces repeated computation.

> Result: **High-speed inference and audio response**, even on standard CPUs, with **zero reliance on cloud services**.

---

## **💡 Problem-Solving Capabilities**

Sydney is more than a chatbot—it’s a **problem-solving AI assistant**:

* **Multi-step reasoning**: Can combine multiple symptoms, conditions, and treatment options.
* **Cross-topic retrieval**: Uses **RAG + memory + LLM** to synthesize complex guidance.
* **Structured outputs**: Provides bullet points, warnings, and key takeaways for clarity.

---

## **🌟 Why Sydney Stands Out**

* **Multimodal**: Single version supports **text and voice** seamlessly.
* **Offline & CPU-Optimized**: Works anywhere, no internet needed, fast even on standard CPUs.
* **Memory + Multi-RAG**: Maintains **context**, retrieves **accurate information**, solves **complex problems**.
* **Granite 47M R2 Embeddings**: State-of-the-art retrieval model for **high-fidelity, semantic medical reasoning**.
* **Problem-Solving Ready**: Handles **multi-topic queries, stepwise reasoning, and actionable suggestions**.
**Example Use Case:**

```
User: I have hypertension and mild kidney issues. Can I exercise, and what should I eat?
Assistant: 
- Exercise: Low-impact cardio, yoga, 3-5 times/week
- Diet: Low sodium, moderate protein, avoid processed foods
- Monitor: Blood pressure and kidney function regularly
- Warning: Avoid strenuous exercises that elevate blood pressure rapidly
```

---

## **⚙ Installation**

### **1. Clone Repository**

```bash
git clone https://github.com/AvijitShil/Sydney.git
cd Sydney
```

### **2. Setup Python Environment**

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

> Or manually if `requirements.txt` is missing:

```bash
pip install torch numpy gradio soundfile coqui-tts faster-whisper langchain_ollama
```

---

## **🚀 Usage**

### **Launch Sydney**

```bash
python main.py
```

### **Gradio Interface**

* **💬 Chat Tab**: Type questions → get **text + audio responses**.
* **🎤 Speak Tab**: Record questions → get **transcribed text + audio response**.

---

## **🔧 Customization**

* **Memory Size**: Adjust `CONFIG["max_memory_turns"]`.
* **TTS Speed / Style**: Change `speed` parameter in `tts_to_file()`.
* **Knowledge Base Expansion**: Add new documents to `CONFIG["medical_docs"]`.
* **Emergency Handling**: Customize `is_medical_emergency()` and `get_emergency_response()`.

---

## **📂 Folder Structure**

```
.
├─ main.py                # Core application
├─ rag_system.py          # RAG & emergency handling
├─ requirements.txt       # Dependencies
├─ memory.json            # Persistent conversation memory
└─ README.md
```

---

## **💡 Contributing**

* Fork & PR for improvements:

  * Fine-tune TTS or LLM for voice/naturalness
  * Add medical knowledge or RAG sources
  * Enhance offline performance
* Ensure **no real medical advice is hard-coded**.

---

## **📜 License**

MIT License – see [LICENSE](LICENSE) for details.


### ⚕️ **IMPORTANT MEDICAL DISCLAIMER**

This AI assistant is for **educational and informational purposes only**.
It cannot and should not replace professional medical advice, diagnosis, or treatment.
**Always consult qualified healthcare professionals for medical concerns.**

---

### 🚀 Features

* **🎤 Speech-to-Text (STT)**: Uses **Faster-Whisper** for accurate local transcription
* **🧠 LLM Processing**: Runs **Gemma 1B** locally via Ollama
* **🔊 Text-to-Speech (TTS)**: **GlowTTS** for natural, expressive voice output
* **📚 Retrieval-Augmented Generation (RAG)**: Knowledge base with vector embeddings
* **💭 Memory Management**: Maintains conversation context across turns
* **🚨 Emergency Detection**: Detects critical medical keywords automatically
* **⚡ Optimized Performance**: Async processing + model caching

---

### 📋 Requirements

* **Python**: 3.8 or higher
* **RAM**: 8GB minimum (16GB recommended)
* **Storage**: \~5GB free space for models
* **Dependencies**:

  * PyTorch (CPU or CUDA if GPU available)
  * Ollama (for hosting LLMs)
  * Audio drivers (microphone & speaker support)

---

### 📁 Project Structure

```
sydney/
├── main.py                 # Core application
├── rag_system.py           # RAG implementation
├── config.py               # Configuration management
├── requirements.txt        # Dependencies
├── memory.json             # Conversation memory
├── response.json           # Response cache
└── medical_knowledge_db/   # Vector database (auto-created)
```

---

### 🔒 Privacy & Safety

* **100% Local Processing**: No internet or cloud dependency
* **Conversation Memory**: Stored locally in JSON (can be encrypted)
* **Emergency Safety**: Detects medical emergencies and prompts urgent actions
* **Medical Disclaimers**: Auto-injected into sensitive health responses

---

### 🚀 Performance Optimizations

* **Async Processing**: Non-blocking STT, LLM, and TTS pipeline
* **Model Caching**: Preloads models for faster response
* **Memory Management**: Maintains rolling context window
* **Vector Search**: Efficient retrieval with embeddings

---

### 🛡️ Safety Features

1. **Emergency Detection**: Auto flagging of life-threatening terms
2. **Medical Disclaimers**: Always included in health-related answers
3. **Graceful Fallbacks**: Handles low-resource environments (CPU-only, int8 quantization)
4. **Local-Only**: Keeps sensitive queries private

---






Sydney is **not just an assistant—it’s an offline, multimodal AI medical companion**, capable of **massive-scale problem solving** while preserving **privacy, speed, and accuracy**.

---



