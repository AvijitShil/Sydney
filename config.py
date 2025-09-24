# --- Medical AI Configuration ---

import os
from typing import Dict, Any

# Environment-based configuration
ENV = os.getenv("ENVIRONMENT", "development")  # development, production

# Base configuration
BASE_CONFIG = {
    # Memory and Context
    "memory_file": "memory.json",
    "max_memory_turns": 10,
    "response_cache_size": 100,
    
    # Audio Configuration
    "audio_sample_rate": 16000,
    "audio_duration": 5,
    "audio_chunk_duration": 0.1,
    "silence_threshold": 0.01,
    "silence_timeout": 1.5,
    
    # Model Configuration
    "whisper_model_size": "small",  # tiny, base, small, medium, large
    "tts_model": "tts_models/en/ljspeech/glow-tts",
    "llm_model": "gemma:1b",
    "embedding_model": "all-MiniLM-L6-v2",
    
    # RAG Configuration
    "vector_db_path": "./medical_knowledge_db",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "retrieval_k": 3,

    # Dataset Configuration (NEW)
    "dataset_path": "./bioasq_dataset.json",  # Path to your dataset file
    "dataset_cache_file": "./bioasq_dataset_cache.pkl",  # Optional: cache embeddings
    
    # Performance
    "max_workers": 3,
    "use_gpu": True,
    "model_compute_type": "float16",  # float16, int8, float32
    
    # Safety and Medical
    "medical_disclaimer": "⚕️ MEDICAL DISCLAIMER: I am an AI assistant and cannot provide medical diagnosis or treatment advice. For medical emergencies, contact emergency services immediately.",
    "emergency_phone_numbers": {
        "US": "911",
        "UK": "999", 
        "EU": "112",
        "Canada": "911",
        "Australia": "000"
    },
    
    # Logging
    "log_level": "INFO",
    "log_file": "medical_ai.log"
}

# Development configuration
DEVELOPMENT_CONFIG = {
    **BASE_CONFIG,
    "whisper_model_size": "tiny",  # Faster for development
    "model_compute_type": "int8",
    "max_workers": 2,
    "log_level": "DEBUG"
}

# Production configuration  
PRODUCTION_CONFIG = {
    **BASE_CONFIG,
    "whisper_model_size": "small",
    "model_compute_type": "float16",
    "max_workers": 4,
    "log_level": "WARNING",
    "use_response_caching": True
}

# Medical Emergency Configuration
EMERGENCY_CONFIG = {
    "keywords": [
        "chest pain", "heart attack", "stroke", "difficulty breathing", "severe pain",
        "unconscious", "bleeding", "overdose", "suicide", "emergency", "911", "ambulance",
        "severe headache", "high fever", "allergic reaction", "anaphylaxis", "seizure",
        "can't breathe", "choking", "severe bleeding", "overdose", "poisoning"
    ],
    "priority_response": """🚨 MEDICAL EMERGENCY DETECTED 🚨

If this is a life-threatening emergency:
• Call 911 (US), 999 (UK), 112 (EU), or your local emergency number IMMEDIATELY
• If unconscious/unresponsive: Call emergency services and start CPR if trained
• If having chest pain: Call 100, chew aspirin if not allergic, stay calm
• If stroke symptoms: Call 911 immediately - every minute counts

I am an AI and cannot provide emergency medical care. Professional medical help is essential for emergencies."""
}

# Medical Dataset Configuration
MEDICAL_DATASET = {
    # Core medical knowledge
    "base_knowledge": [
        "Diabetes is a chronic disease affecting blood sugar levels. Symptoms include increased thirst, frequent urination, and fatigue.",
        "Hypertension (high blood pressure) can cause headaches, shortness of breath, and chest pain. Regular monitoring is essential.",
        "Common cold symptoms include runny nose, sore throat, and mild fever. Rest and hydration are important.",
        "Proper nutrition is essential for health. A balanced diet includes proteins, carbohydrates, fats, vitamins, and minerals.",
    ],
    
    # File paths for external datasets
    "dataset_paths": {
        "primary": "C:/Users/aviji/OneDrive/Desktop/whisper/llm/Datasets/bioasq_dataset.json",  # Your main dataset path
        "supplementary": "c:/Users/aviji/OneDrive/Desktop/whisper/llm/medical_knowledge_db/additional_data.json",  # Additional data if needed
        "custom": "c:/Users/aviji/OneDrive/Desktop/whisper/llm/medical_knowledge_db/custom_knowledge.json"  # Custom knowledge if needed
    },
    
    # Dataset processing settings
    "processing": {
        "chunk_size": 500,
        "overlap": 50,
        "min_chunk_length": 100,
        "include_metadata": True
    }
}

# Medical Knowledge Categories
MEDICAL_CATEGORIES = {
    "cardiovascular": ["heart", "chest pain", "blood pressure", "angina", "arrhythmia"],
    "respiratory": ["breathing", "asthma", "cough", "pneumonia", "lung"],
    "endocrine": ["diabetes", "thyroid", "hormone", "insulin", "blood sugar"],
    "neurological": ["headache", "seizure", "stroke", "migraine", "dizziness"],
    "gastrointestinal": ["stomach", "nausea", "diarrhea", "constipation", "heartburn"],
    "mental_health": ["depression", "anxiety", "stress", "panic", "mood"],
    "general": ["fever", "fatigue", "pain", "medication", "treatment"]
}

def get_config() -> Dict[str, Any]:
    """Get configuration based on environment."""
    if ENV == "production":
        return PRODUCTION_CONFIG
    else:
        return DEVELOPMENT_CONFIG

# Export CONFIG for other modules to import
CONFIG = get_config()

# Hardware optimization detection
def get_hardware_config() -> Dict[str, Any]:
    """Detect hardware capabilities and optimize accordingly."""
    import torch
    import platform
    
    config = {}
    
    # GPU Detection
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        config["gpu_available"] = True
        config["gpu_name"] = gpu_name
        config["gpu_memory_gb"] = gpu_memory
        config["use_gpu"] = True
        config["compute_type"] = "float16" if gpu_memory > 4 else "int8"
    else:
        config["gpu_available"] = False
        config["use_gpu"] = False
        config["compute_type"] = "int8"
    
    # CPU Detection
    import psutil
    config["cpu_cores"] = psutil.cpu_count(logical=False)
    config["cpu_threads"] = psutil.cpu_count(logical=True)
    config["memory_gb"] = psutil.virtual_memory().total / 1024**3
    
    # Optimize workers based on hardware
    if config["cpu_cores"] >= 8:
        config["max_workers"] = 4
    elif config["cpu_cores"] >= 4:
        config["max_workers"] = 3
    else:
        config["max_workers"] = 2
    
    # Whisper model size based on available memory
    if config["memory_gb"] >= 16:
        config["whisper_model_size"] = "small"
    elif config["memory_gb"] >= 8:
        config["whisper_model_size"] = "base"
    else:
        config["whisper_model_size"] = "tiny"
    
    return config
