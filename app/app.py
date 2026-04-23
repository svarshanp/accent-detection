"""
Accent Detection using Speech Audio — Streamlit Application
============================================================
Premium dark-themed web application with glassmorphism UI.
Features: Upload audio → Predict accent → View analysis → Compare models
"""

import os
import sys
import io
import tempfile
import numpy as np
import librosa
import streamlit as st
import plotly.graph_objects as go

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import ACCENT_MAP, ACCENT_CLASSES, SAMPLE_RATE, DURATION, load_metrics
from src.predict import AccentPredictor, get_available_models
# Import components from same directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
from components import (
    inject_custom_css,
    render_header,
    render_prediction_card,
    render_confidence_bars,
    render_audio_visualizations,
    render_metric_card,
    render_model_comparison_chart,
    render_confusion_matrix_plotly,
    render_gradient_divider,
    render_transcription_card,
    render_recording_status,
    render_speech_analysis_card,
)

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Accent Detection AI — Speech Analysis",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()


# ============================================================
# Sidebar
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="background: linear-gradient(135deg, #667eea, #764ba2);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       font-size: 1.5rem; font-weight: 800;">🎙️ Accent AI</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🎙️ Accent Detector", "🗣️ Speech-to-Text", "📊 Model Comparison", "🔬 Audio Analysis", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Model selection
        available_models = get_available_models()
        if available_models:
            selected_model = st.selectbox(
                "🤖 Select Model",
                available_models,
                index=0,
                help="Choose which trained model to use for prediction"
            )
        else:
            selected_model = None
            st.warning("No trained models found!")
        
        st.markdown("---")
        
        # Info
        st.markdown("""
        <div style="padding: 1rem; background: rgba(255,255,255,0.03);
                    border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
            <p style="color: #718096; font-size: 0.8rem; margin: 0;">
                <strong style="color: #a0aec0;">Supported Accents:</strong><br>
                🇺🇸 US • 🇬🇧 England • 🇮🇳 India<br>
                🇦🇺 Australia • 🇨🇦 Canada<br>
                🏴 Scotland • 🇮🇪 Ireland • 🌍 Africa
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <p style="color: #4a5568; font-size: 0.75rem; text-align: center;">
            Built with PyTorch, Transformers,<br>
            Whisper, Scikit-learn & Streamlit<br>
            Accent Detection Project © 2026
        </p>
        """, unsafe_allow_html=True)
    
    return page, selected_model


# ============================================================
# Page 1: Accent Detector
# ============================================================
def page_accent_detector(selected_model):
    render_header()
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #e2e8f0; margin-bottom: 1rem;">📤 Upload Audio</h3>
            <p style="color: #718096; font-size: 0.9rem;">
                Upload an audio file (.wav, .mp3, .ogg, .flac) to detect the speaker's accent.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'ogg', 'flac', 'webm'],
            key="audio_uploader",
            label_visibility="collapsed"
        )
        
        # Sample audio buttons
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #e2e8f0; margin-bottom: 0.8rem;">🎵 Or Try a Sample</h4>
            <p style="color: #718096; font-size: 0.85rem;">
                Generate a sample audio with a specific accent to test the model.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        sample_cols = st.columns(4)
        sample_accent = None
        
        accent_list = list(ACCENT_MAP.items())
        for i, col in enumerate(sample_cols):
            with col:
                if i < len(accent_list):
                    key, info = accent_list[i]
                    if st.button(f"{info['flag']} {key.upper()}", key=f"sample_{key}",
                                use_container_width=True):
                        sample_accent = key
        
        sample_cols2 = st.columns(4)
        for i, col in enumerate(sample_cols2):
            with col:
                idx = i + 4
                if idx < len(accent_list):
                    key, info = accent_list[idx]
                    if st.button(f"{info['flag']} {key.upper()}", key=f"sample_{key}",
                                use_container_width=True):
                        sample_accent = key
    
    with col2:
        audio_data = None
        audio_sr = SAMPLE_RATE
        
        # Handle uploaded file
        if uploaded_file is not None:
            try:
                # Read file bytes once, then reuse
                file_bytes = uploaded_file.getvalue()
                if not file_bytes or len(file_bytes) == 0:
                    st.error("⚠️ Uploaded file is empty. Please upload a valid audio file.")
                else:
                    # Save to temp file for librosa
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name
                    
                    audio_data, audio_sr = librosa.load(tmp_path, sr=SAMPLE_RATE, duration=5)
                    try:
                        os.unlink(tmp_path)
                    except (PermissionError, OSError):
                        pass  # Windows file lock; temp file cleaned up later
                    
                    # Play audio from the raw bytes (not the consumed file object)
                    st.audio(file_bytes, format='audio/wav')
                    
                    # Validate audio content
                    rms = np.sqrt(np.mean(audio_data ** 2))
                    if rms < 1e-6:
                        st.warning("⚠️ The uploaded audio appears to be silent. Results may be inaccurate.")
            except Exception as e:
                st.error(f"❌ Error loading audio: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        # Handle sample generation
        elif sample_accent is not None:
            try:
                from src.data_loader import generate_accent_audio
                
                with st.spinner(f"Generating {ACCENT_MAP[sample_accent]['flag']} {sample_accent.upper()} sample..."):
                    audio_data = generate_accent_audio(
                        sample_accent, duration=DURATION, sr=SAMPLE_RATE,
                        speaker_id=np.random.randint(0, 100)
                    )
                    
                    # Play audio using in-memory buffer (avoids Windows temp file lock)
                    import soundfile as sf
                    audio_buffer = io.BytesIO()
                    sf.write(audio_buffer, audio_data, SAMPLE_RATE, format='WAV')
                    audio_buffer.seek(0)
                    st.audio(audio_buffer, format='audio/wav')
            except Exception as e:
                st.error(f"❌ Error generating sample audio: {e}")
        
        # Make prediction
        if audio_data is not None and selected_model is not None:
            with st.spinner("🔮 Analyzing accent..."):
                try:
                    predictor = AccentPredictor(model_type=selected_model)
                    result = predictor.predict(audio_array=audio_data, sr=audio_sr)
                    
                    render_prediction_card(result)
                    
                    # Top 3
                    st.markdown("""
                    <div class="glass-card">
                        <h3 style="color: #e2e8f0; margin-bottom: 0.8rem;">🏆 Top 3 Predictions</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for i, pred in enumerate(result['top_3']):
                        medal = ['🥇', '🥈', '🥉'][i]
                        st.markdown(f"""
                        <div class="confidence-bar-container">
                            <span style="font-size: 1.3rem;">{medal}</span>
                            <span class="confidence-flag">{pred['flag']}</span>
                            <span class="confidence-label">{pred['accent']}</span>
                            <div class="confidence-bar-bg">
                                <div class="confidence-bar-fill" style="width: {max(pred['confidence']*100, 1)}%"></div>
                            </div>
                            <span class="confidence-value">{pred['confidence']:.1%}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                except FileNotFoundError as e:
                    st.error(f"🔍 Model file not found: {e}")
                    st.info("Please train the models first using `python run_pipeline.py`")
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        elif audio_data is None:
            st.markdown("""
            <div class="prediction-card" style="opacity: 0.5;">
                <div class="prediction-flag">🎤</div>
                <div class="prediction-accent">Waiting for Audio</div>
                <div class="prediction-country">Upload a file or click a sample button</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Full predictions bar chart (below)
    if audio_data is not None and selected_model is not None:
        render_gradient_divider()
        try:
            predictor = AccentPredictor(model_type=selected_model)
            result = predictor.predict(audio_array=audio_data, sr=audio_sr)
            render_confidence_bars(result['all_predictions'])
        except Exception as e:
            st.error(f"❌ Could not generate confidence chart: {e}")


# ============================================================
# Page 2: Model Comparison
# ============================================================
def page_model_comparison():
    st.markdown("""
    <div class="main-header">
        <h1>📊 Model Comparison</h1>
        <p>Compare performance across all trained models</p>
    </div>
    """, unsafe_allow_html=True)
    
    metrics = load_metrics()
    
    if not metrics:
        st.warning("No model metrics found. Train models first!")
        return
    
    # Summary metrics
    cols = st.columns(len(metrics))
    for i, (model_name, data) in enumerate(metrics.items()):
        render_metric_card(model_name, f"{data['accuracy']:.1%}", cols[i])
    
    render_gradient_divider()
    
    # Interactive comparison chart
    render_model_comparison_chart(metrics)
    
    render_gradient_divider()
    
    # Confusion matrices
    st.markdown("""
    <h2 style="color: #e2e8f0; text-align: center; margin-bottom: 1.5rem;">
        Confusion Matrices
    </h2>
    """, unsafe_allow_html=True)
    
    model_names = list(metrics.keys())
    selected_cm_model = st.selectbox("Select model", model_names, key="cm_select")
    
    if selected_cm_model and 'confusion_matrix' in metrics[selected_cm_model]:
        render_confusion_matrix_plotly(
            metrics[selected_cm_model]['confusion_matrix'],
            selected_cm_model
        )
    
    render_gradient_divider()
    
    # Per-class F1 scores
    st.markdown("""
    <h2 style="color: #e2e8f0; text-align: center; margin-bottom: 1.5rem;">
        Per-Class Performance
    </h2>
    """, unsafe_allow_html=True)
    
    labels = [ACCENT_MAP[a]['label'] for a in ACCENT_CLASSES]
    
    fig = go.Figure()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i, (model_name, data) in enumerate(metrics.items()):
        report = data.get('classification_report', {})
        f1_scores = [report.get(label, {}).get('f1-score', 0) * 100 for label in labels]
        
        fig.add_trace(go.Scatterpolar(
            r=f1_scores + [f1_scores[0]],
            theta=labels + [labels[0]],
            name=model_name,
            line=dict(color=colors[i % len(colors)], width=2),
            fill='toself',
            fillcolor=f'rgba({int(colors[i % len(colors)][1:3], 16)}, '
                       f'{int(colors[i % len(colors)][3:5], 16)}, '
                       f'{int(colors[i % len(colors)][5:7], 16)}, 0.1)',
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                          gridcolor='rgba(255,255,255,0.1)'),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            bgcolor='rgba(0,0,0,0)',
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a0aec0'),
        legend=dict(bgcolor='rgba(255,255,255,0.05)',
                   font=dict(color='#e2e8f0')),
        height=500,
        title=dict(text='Radar Chart — Per-Class F1 Score',
                   font=dict(color='#e2e8f0', size=16)),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Training curves for DL models
    dl_models = {k: v for k, v in metrics.items() if 'history' in v}
    if dl_models:
        render_gradient_divider()
        st.markdown("""
        <h2 style="color: #e2e8f0; text-align: center; margin-bottom: 1.5rem;">
            Training Curves
        </h2>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_loss = go.Figure()
            dl_colors = {'CNN': '#FF6B6B', 'LSTM': '#4ECDC4', 'Transformer': '#764ba2'}
            for name, data in dl_models.items():
                h = data['history']
                color = dl_colors.get(name, '#45B7D1')
                epochs = list(range(1, len(h['train_loss']) + 1))
                fig_loss.add_trace(go.Scatter(x=epochs, y=h['train_loss'],
                                             name=f'{name} Train', line=dict(color=color, width=2)))
                fig_loss.add_trace(go.Scatter(x=epochs, y=h['val_loss'],
                                             name=f'{name} Val', line=dict(color=color, width=2, dash='dash')))
            
            fig_loss.update_layout(
                title=dict(text='Loss Curves', font=dict(color='#e2e8f0', size=16)),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.02)',
                font=dict(color='#a0aec0'),
                legend=dict(bgcolor='rgba(255,255,255,0.05)', font=dict(color='#e2e8f0')),
                height=350,
                xaxis=dict(title='Epoch', gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(title='Loss', gridcolor='rgba(255,255,255,0.05)'),
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        
        with col2:
            fig_acc = go.Figure()
            for name, data in dl_models.items():
                h = data['history']
                color = dl_colors.get(name, '#45B7D1')
                epochs = list(range(1, len(h['train_acc']) + 1))
                fig_acc.add_trace(go.Scatter(x=epochs, y=[a*100 for a in h['train_acc']],
                                            name=f'{name} Train', line=dict(color=color, width=2)))
                fig_acc.add_trace(go.Scatter(x=epochs, y=[a*100 for a in h['val_acc']],
                                            name=f'{name} Val', line=dict(color=color, width=2, dash='dash')))
            
            fig_acc.update_layout(
                title=dict(text='Accuracy Curves', font=dict(color='#e2e8f0', size=16)),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.02)',
                font=dict(color='#a0aec0'),
                legend=dict(bgcolor='rgba(255,255,255,0.05)', font=dict(color='#e2e8f0')),
                height=350,
                xaxis=dict(title='Epoch', gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(title='Accuracy (%)', gridcolor='rgba(255,255,255,0.05)'),
            )
            st.plotly_chart(fig_acc, use_container_width=True)


# ============================================================
# Page 3: Audio Analysis
# ============================================================
def page_audio_analysis():
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Audio Analysis</h1>
        <p>Visualize waveform, spectrogram, and MFCC features of your audio</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #e2e8f0;">Upload Audio</h3>
            <p style="color: #718096; font-size: 0.9rem;">
                Upload any audio file to visualize its acoustic features.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded = st.file_uploader("Upload audio", type=['wav', 'mp3', 'ogg', 'flac'],
                                    key="analysis_upload", label_visibility="collapsed")
        
        # Or generate sample
        st.markdown("**Or generate a sample:**")
        sample_accent = st.selectbox(
            "Accent",
            [''] + list(ACCENT_MAP.keys()),
            format_func=lambda x: f"{ACCENT_MAP[x]['flag']} {ACCENT_MAP[x]['label']}" if x else "Select...",
            key="analysis_accent"
        )
        
        if sample_accent and st.button("Generate Sample", key="gen_analysis"):
            try:
                from src.data_loader import generate_accent_audio
                audio_data = generate_accent_audio(sample_accent, DURATION, SAMPLE_RATE)
                st.session_state['analysis_audio'] = audio_data
                st.session_state['analysis_sr'] = SAMPLE_RATE
            except Exception as e:
                st.error(f"❌ Error generating sample: {e}")
    
    with col2:
        audio_data = None
        sr = SAMPLE_RATE
        
        if uploaded is not None:
            try:
                file_bytes = uploaded.getvalue()
                if not file_bytes or len(file_bytes) == 0:
                    st.error("⚠️ Uploaded file is empty.")
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name
                    audio_data, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, duration=5)
                    try:
                        os.unlink(tmp_path)
                    except (PermissionError, OSError):
                        pass  # Windows file lock; temp file cleaned up later
                    st.audio(file_bytes, format='audio/wav')
            except Exception as e:
                st.error(f"❌ Error loading audio: {e}")
                import traceback
                st.code(traceback.format_exc())
        elif 'analysis_audio' in st.session_state:
            audio_data = st.session_state['analysis_audio']
            sr = st.session_state['analysis_sr']
        
        if audio_data is not None:
            # Audio stats
            stat_cols = st.columns(4)
            render_metric_card("Duration", f"{len(audio_data)/sr:.2f}s", stat_cols[0])
            render_metric_card("Sample Rate", f"{sr}Hz", stat_cols[1])
            render_metric_card("Samples", f"{len(audio_data):,}", stat_cols[2])
            render_metric_card("RMS Energy", f"{np.sqrt(np.mean(audio_data**2)):.4f}", stat_cols[3])
            
            render_gradient_divider()
            render_audio_visualizations(audio_data, sr)
        else:
            st.markdown("""
            <div class="prediction-card" style="opacity: 0.5;">
                <div class="prediction-flag">🔬</div>
                <div class="prediction-accent">No Audio Loaded</div>
                <div class="prediction-country">Upload a file or generate a sample</div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# Page 4: Speech-to-Text
# ============================================================
def page_speech_to_text(selected_model):
    st.markdown("""
    <div class="main-header">
        <h1>🗣️ Speech-to-Text + Accent Detection</h1>
        <p>Record your voice or upload audio — get accent detection and speech transcription powered by Whisper AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #e2e8f0; margin-bottom: 0.8rem;">🎙️ Record Your Voice</h3>
            <p style="color: #718096; font-size: 0.9rem;">
                Click the microphone button below to record a short audio clip.
                Speak naturally in English for best results.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Microphone recording via Streamlit's built-in audio input
        recorded_audio = st.audio_input(
            "🎙️ Click to record your voice",
            key="mic_recorder"
        )
        
        render_gradient_divider()
        
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #e2e8f0; margin-bottom: 0.8rem;">📤 Or Upload Audio</h3>
            <p style="color: #718096; font-size: 0.9rem;">
                Upload a pre-recorded audio file for analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload audio file",
            type=['wav', 'mp3', 'ogg', 'flac', 'webm'],
            key="stt_uploader",
            label_visibility="collapsed"
        )
        
        # Whisper model size selector
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #e2e8f0; margin-bottom: 0.8rem;">⚙️ Whisper Model</h3>
        </div>
        """, unsafe_allow_html=True)
        
        whisper_size = st.selectbox(
            "Whisper model size",
            ['tiny', 'base', 'small'],
            index=0,
            help="tiny = fastest (~75MB), base = balanced (~150MB), small = most accurate (~500MB)",
            key="whisper_size"
        )
    
    with col2:
        audio_data = None
        audio_sr = SAMPLE_RATE
        audio_source = None  # Temp file path for Whisper transcription
        
        # Handle recorded audio
        if recorded_audio is not None:
            try:
                # Read bytes once, reuse for both playback and processing
                rec_bytes = recorded_audio.getvalue()
                if not rec_bytes or len(rec_bytes) == 0:
                    st.error("⚠️ Recording is empty. Please try recording again.")
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                        tmp.write(rec_bytes)
                        tmp_path = tmp.name
                    
                    audio_data, audio_sr = librosa.load(tmp_path, sr=SAMPLE_RATE, duration=10)
                    audio_source = tmp_path  # Keep for Whisper file-based transcription
                    st.audio(rec_bytes, format='audio/wav')
                    render_recording_status(False)
            except Exception as e:
                st.error(f"❌ Error processing recording: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        # Handle uploaded file
        elif uploaded_file is not None:
            try:
                # Read bytes once, reuse for both playback and processing
                file_bytes = uploaded_file.getvalue()
                if not file_bytes or len(file_bytes) == 0:
                    st.error("⚠️ Uploaded file is empty. Please upload a valid audio file.")
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name
                    
                    audio_data, audio_sr = librosa.load(tmp_path, sr=SAMPLE_RATE, duration=10)
                    audio_source = tmp_path  # Keep for Whisper file-based transcription
                    st.audio(file_bytes, format='audio/wav')
            except Exception as e:
                st.error(f"❌ Error loading audio: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        # Process audio: Accent Detection + Speech-to-Text
        if audio_data is not None:
            # Note: Even quiet audio may contain speech — we try transcription regardless
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms < 1e-7:
                st.info("ℹ️ The audio appears very quiet. Transcription will still be attempted.")
            
            # Initialize result holders
            accent_result = {'accent': 'Unknown', 'flag': '🌐', 'confidence': 0}
            stt_result = {'text': '', 'language': 'unknown', 'confidence': 0, 'error': ''}
            
            # 1. ACCENT DETECTION
            if selected_model is not None:
                with st.spinner("🔮 Analyzing accent..."):
                    try:
                        predictor = AccentPredictor(model_type=selected_model)
                        accent_result = predictor.predict(audio_array=audio_data, sr=audio_sr)
                        render_prediction_card(accent_result)
                    except FileNotFoundError as e:
                        st.error(f"🔍 Model file not found: {e}")
                        st.info("Please train the models first using `python run_pipeline.py`")
                    except Exception as e:
                        st.error(f"❌ Accent detection error: {e}")
            else:
                st.warning("⚠️ No trained models available for accent detection.")
            
            render_gradient_divider()
            
            # 2. SPEECH-TO-TEXT
            with st.spinner(f"📝 Transcribing speech with Whisper ({whisper_size})..."):
                try:
                    from src.speech_to_text import SpeechToText
                    
                    # Cache Whisper model in session state
                    cache_key = f'stt_model_{whisper_size}'
                    if cache_key not in st.session_state:
                        st.session_state[cache_key] = SpeechToText(model_size=whisper_size)
                    
                    stt = st.session_state[cache_key]
                    
                    # Check if Whisper model loaded successfully
                    if not stt.is_available:
                        st.warning(
                            f"⚠️ **Whisper model could not be loaded.**\n\n"
                            f"Error: {stt.load_error or 'Unknown'}\n\n"
                            "Install OpenAI Whisper with:\n\n"
                            "```\npip install openai-whisper\n```\n\n"
                            "Also ensure **ffmpeg** is installed on your system."
                        )
                    else:
                        # Prefer array-based transcription (doesn't require ffmpeg)
                        # Fall back to file-based only if array is not available
                        if audio_data is not None:
                            stt_result = stt.transcribe_array(audio_data, sr=audio_sr, language='en')
                        elif audio_source and os.path.exists(audio_source):
                            stt_result = stt.transcribe(audio_source, language='en')
                        else:
                            stt_result = {'text': '', 'language': 'unknown', 'confidence': 0, 'error': 'No audio data available'}
                        
                        # Check for errors from STT module
                        error = stt_result.get('error', '')
                        text = stt_result.get('text', '')
                        
                        if error:
                            st.warning(
                                f"⚠️ **Transcription issue:** {error}\n\n"
                                "This can happen if:\n"
                                "- The audio is too short or too quiet\n"
                                "- The recording contains only noise/music\n"
                                "- Try speaking louder and closer to the microphone"
                            )
                        elif text:
                            render_transcription_card(
                                text,
                                stt_result.get('language', 'en'),
                                stt_result.get('confidence', 0.0)
                            )
                        else:
                            st.info(
                                "📭 No speech detected in the audio. This can happen if:\n"
                                "- The audio is too short or too quiet\n"
                                "- The recording contains only noise/music\n"
                                "- Try speaking louder and closer to the microphone"
                            )
                    
                except ImportError as ie:
                    st.warning(
                        f"⚠️ **Whisper dependency missing:** `{ie}`\n\n"
                        "Install with:\n\n"
                        "```\npip install openai-whisper\n```"
                    )
                except Exception as e:
                    st.error(f"❌ Transcription error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # 3. COMBINED ANALYSIS CARD
            if accent_result.get('confidence', 0) > 0 and stt_result.get('text', ''):
                render_gradient_divider()
                render_speech_analysis_card(accent_result, stt_result)
            
            # Cleanup temp file
            if audio_source:
                try:
                    os.unlink(audio_source)
                except (PermissionError, OSError):
                    pass  # Windows file lock; temp file cleaned up later
        
        else:
            # Placeholder when no audio is loaded
            st.markdown("""
            <div class="prediction-card" style="opacity: 0.5;">
                <div class="prediction-flag">🗣️</div>
                <div class="prediction-accent">Waiting for Audio</div>
                <div class="prediction-country">Record your voice or upload a file to begin analysis</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature description
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: #e2e8f0; margin-bottom: 0.8rem;">✨ What This Page Does</h4>
                <ul style="color: #a0aec0; line-height: 2;">
                    <li><strong style="color: #667eea;">Accent Detection</strong> — Identifies your English accent and maps it to a country/region</li>
                    <li><strong style="color: #4ECDC4;">Speech-to-Text</strong> — Transcribes your spoken words using OpenAI Whisper</li>
                    <li><strong style="color: #764ba2;">Combined Analysis</strong> — Shows accent + transcription results side by side</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# Page 5: About
# ============================================================
def page_about():
    st.markdown("""
    <div class="main-header">
        <h1>ℹ️ About This Project</h1>
        <p>Accent Detection using Speech Audio — End-to-End ML/DL/Transformer Pipeline with Speech-to-Text</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #e2e8f0;">🎯 Project Overview</h3>
        <p style="color: #a0aec0; line-height: 1.8;">
            This project implements an end-to-end pipeline for detecting English accents from speech audio.
            Given a short audio clip of someone speaking English, the system identifies their accent type
            and maps it to an approximate country or region of origin. It also includes Speech-to-Text
            transcription powered by OpenAI Whisper.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #e2e8f0;">🏗️ Architecture</h3>
            <ul style="color: #a0aec0; line-height: 2;">
                <li><strong>Audio Processing:</strong> librosa, soundfile</li>
                <li><strong>Feature Extraction:</strong> MFCC, Mel-Spectrogram, Chroma</li>
                <li><strong>ML Models:</strong> SVM, Random Forest, XGBoost</li>
                <li><strong>DL Models:</strong> 1D-CNN, Bidirectional LSTM</li>
                <li><strong>Transformer:</strong> Audio Spectrogram Transformer (AST)</li>
                <li><strong>Speech-to-Text:</strong> OpenAI Whisper</li>
                <li><strong>Framework:</strong> PyTorch, Scikit-learn</li>
                <li><strong>Deployment:</strong> Streamlit</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #e2e8f0;">🌍 Supported Accents</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for key, info in ACCENT_MAP.items():
            st.markdown(f"""
            <div class="accent-info-card">
                <h4>{info['flag']} {info['label']}</h4>
                <p>Country/Region: {info['country']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    render_gradient_divider()
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #e2e8f0;">📁 Project Structure</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.code("""
    Accent Detection Project/
    ├── data/
    │   ├── raw/                  # Audio files by accent
    │   ├── processed/            # Extracted features
    │   └── metadata.csv          # Dataset metadata
    ├── src/
    │   ├── data_loader.py        # Dataset generation
    │   ├── feature_extractor.py  # Audio feature extraction
    │   ├── train_ml.py           # ML model training
    │   ├── train_dl.py           # DL model training
    │   ├── train_transformer.py  # Transformer model training
    │   ├── speech_to_text.py     # Whisper STT module
    │   ├── evaluate.py           # Evaluation & plots
    │   ├── predict.py            # Inference pipeline
    │   └── utils.py              # Utilities & constants
    ├── models/
    │   ├── ml/                   # Saved sklearn models
    │   └── dl/                   # Saved PyTorch + Transformer models
    ├── app/
    │   ├── app.py                # This Streamlit app
    │   └── components.py         # UI components
    └── results/
        ├── figures/              # Generated plots
        └── metrics.json          # Model metrics
    """, language='text')
    
    render_gradient_divider()
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #e2e8f0;">🔧 Pipeline Steps</h3>
        <ol style="color: #a0aec0; line-height: 2.2;">
            <li><strong>Data Generation:</strong> Synthetic accent-specific audio with realistic acoustic profiles</li>
            <li><strong>Feature Extraction:</strong> MFCC (13 coefficients + deltas), Mel-Spectrogram (128 bands), Spectral features, Chroma</li>
            <li><strong>ML Training:</strong> SVM (RBF kernel), Random Forest, XGBoost with GridSearchCV</li>
            <li><strong>DL Training:</strong> 1D-CNN on spectrograms, Bi-LSTM with attention on MFCC sequences</li>
            <li><strong>Transformer Training:</strong> Audio Spectrogram Transformer with patch embeddings and multi-head self-attention</li>
            <li><strong>Speech-to-Text:</strong> OpenAI Whisper integration for real-time transcription</li>
            <li><strong>Evaluation:</strong> Confusion matrices, per-class F1 scores, model comparison</li>
            <li><strong>Deployment:</strong> Interactive Streamlit app with live recording and prediction</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Main App Router
# ============================================================
def main():
    page, selected_model = render_sidebar()
    
    if page == "🎙️ Accent Detector":
        page_accent_detector(selected_model)
    elif page == "🗣️ Speech-to-Text":
        page_speech_to_text(selected_model)
    elif page == "📊 Model Comparison":
        page_model_comparison()
    elif page == "🔬 Audio Analysis":
        page_audio_analysis()
    elif page == "ℹ️ About":
        page_about()


if __name__ == '__main__':
    main()
