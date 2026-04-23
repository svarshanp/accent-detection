"""
Streamlit UI Components for Accent Detection
=============================================
Reusable styled components for the premium UI.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def inject_custom_css():
    """Inject premium dark-themed CSS."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(20px);
        text-align: center;
    }
    
    .main-header h1 {
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Glass card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);
    }
    
    /* Prediction result card */
    .prediction-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
        backdrop-filter: blur(15px);
        animation: fadeInUp 0.6s ease-out;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .prediction-flag {
        font-size: 4rem;
        margin-bottom: 0.5rem;
    }
    
    .prediction-accent {
        color: #e2e8f0;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    
    .prediction-country {
        color: #a0aec0;
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    .prediction-confidence {
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-top: 0.5rem;
    }
    
    /* Confidence bar */
    .confidence-bar-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
        transition: all 0.3s ease;
    }
    
    .confidence-bar-container:hover {
        background: rgba(255, 255, 255, 0.08);
    }
    
    .confidence-flag { font-size: 1.5rem; }
    
    .confidence-label {
        color: #e2e8f0;
        font-weight: 500;
        min-width: 140px;
        font-size: 0.95rem;
    }
    
    .confidence-bar-bg {
        flex: 1;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        height: 12px;
        overflow: hidden;
    }
    
    .confidence-bar-fill {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transition: width 1s ease-out;
    }
    
    .confidence-value {
        color: #a0aec0;
        font-weight: 600;
        min-width: 55px;
        text-align: right;
        font-size: 0.95rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: #a0aec0;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.3rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%) !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #667eea !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #a0aec0;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2)) !important;
        color: #e2e8f0 !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(102, 126, 234, 0.3);
        border-radius: 16px;
        padding: 1rem;
    }
    
    /* Accent info card */
    .accent-info-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    
    .accent-info-card h4 {
        color: #e2e8f0;
        margin: 0;
    }
    
    .accent-info-card p {
        color: #718096;
        font-size: 0.9rem;
        margin: 0.3rem 0 0 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Divider */
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
        border: none;
        margin: 2rem 0;
        border-radius: 1px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0a1a; }
    ::-webkit-scrollbar-thumb { background: #667eea; border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the main application header."""
    st.markdown("""
    <div class="main-header">
        <h1>🎙️ Accent Detection AI</h1>
        <p>Identify English accents and approximate country of origin using advanced speech analysis</p>
    </div>
    """, unsafe_allow_html=True)


def render_prediction_card(result: dict):
    """Render the main prediction result card."""
    st.markdown(f"""
    <div class="prediction-card">
        <div class="prediction-flag">{result['flag']}</div>
        <div class="prediction-accent">{result['accent']}</div>
        <div class="prediction-country">{result['country']}</div>
        <div class="prediction-confidence">{result['confidence']:.1%}</div>
        <p style="color: #718096; margin-top: 0.3rem; font-size: 0.9rem;">
            Confidence Score • Model: {result.get('model_used', 'N/A')}
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_confidence_bars(predictions: dict):
    """Render confidence bars for all accent predictions."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils import ACCENT_MAP, ACCENT_CLASSES
    
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #e2e8f0; margin-bottom: 1rem;">📊 All Predictions</h3>',
                unsafe_allow_html=True)
    
    for accent_label, confidence in sorted_preds:
        # Find flag
        flag = '🌐'
        for key, info in ACCENT_MAP.items():
            if info['label'] == accent_label:
                flag = info['flag']
                break
        
        width = max(confidence * 100, 1)
        st.markdown(f"""
        <div class="confidence-bar-container">
            <span class="confidence-flag">{flag}</span>
            <span class="confidence-label">{accent_label}</span>
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill" style="width: {width}%"></div>
            </div>
            <span class="confidence-value">{confidence:.1%}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_audio_visualizations(audio: np.ndarray, sr: int):
    """Render waveform and spectrogram visualizations using Plotly."""
    
    # 1. Waveform
    time = np.linspace(0, len(audio) / sr, num=len(audio))
    
    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(
        x=time, y=audio,
        mode='lines',
        line=dict(color='#667eea', width=1),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)',
        name='Waveform'
    ))
    fig_wave.update_layout(
        title=dict(text='Waveform', font=dict(color='#e2e8f0', size=16)),
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.02)',
        font=dict(color='#a0aec0'),
        height=250,
        margin=dict(l=50, r=20, t=40, b=40),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
    )
    st.plotly_chart(fig_wave, use_container_width=True)
    
    # 2. Mel-Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    fig_mel = go.Figure(data=go.Heatmap(
        z=mel_spec_db,
        colorscale=[
            [0, '#0a0a1a'],
            [0.25, '#1a1a4e'],
            [0.5, '#667eea'],
            [0.75, '#764ba2'],
            [1, '#f093fb']
        ],
        showscale=True,
        colorbar=dict(title='dB', tickfont=dict(color='#a0aec0')),
    ))
    fig_mel.update_layout(
        title=dict(text='Mel-Spectrogram', font=dict(color='#e2e8f0', size=16)),
        xaxis_title='Time Frame',
        yaxis_title='Mel Band',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.02)',
        font=dict(color='#a0aec0'),
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_mel, use_container_width=True)
    
    # 3. MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    fig_mfcc = go.Figure(data=go.Heatmap(
        z=mfcc,
        colorscale=[
            [0, '#16213e'],
            [0.3, '#0f3460'],
            [0.5, '#533483'],
            [0.7, '#e94560'],
            [1, '#f093fb']
        ],
        showscale=True,
        colorbar=dict(title='Value', tickfont=dict(color='#a0aec0')),
    ))
    fig_mfcc.update_layout(
        title=dict(text='MFCC Features', font=dict(color='#e2e8f0', size=16)),
        xaxis_title='Time Frame',
        yaxis_title='MFCC Coefficient',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.02)',
        font=dict(color='#a0aec0'),
        height=280,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_mfcc, use_container_width=True)


def render_metric_card(label: str, value: str, col):
    """Render a single metric card."""
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)


def render_model_comparison_chart(metrics: dict):
    """Render interactive model comparison chart."""
    models = list(metrics.keys())
    accuracies = [metrics[m]['accuracy'] * 100 for m in models]
    f1_scores = [metrics[m]['f1_score'] * 100 for m in models]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Accuracy',
        x=models,
        y=accuracies,
        marker_color='#667eea',
        text=[f'{a:.1f}%' for a in accuracies],
        textposition='outside',
        textfont=dict(color='#e2e8f0'),
    ))
    fig.add_trace(go.Bar(
        name='F1-Score',
        x=models,
        y=f1_scores,
        marker_color='#764ba2',
        text=[f'{f:.1f}%' for f in f1_scores],
        textposition='outside',
        textfont=dict(color='#e2e8f0'),
    ))
    
    fig.update_layout(
        barmode='group',
        title=dict(text='Model Performance Comparison',
                   font=dict(color='#e2e8f0', size=20)),
        xaxis_title='Model',
        yaxis_title='Score (%)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.02)',
        font=dict(color='#a0aec0', family='Inter'),
        legend=dict(
            bgcolor='rgba(255,255,255,0.05)',
            bordercolor='rgba(255,255,255,0.1)',
            font=dict(color='#e2e8f0'),
        ),
        height=450,
        yaxis=dict(range=[0, 110], gridcolor='rgba(255,255,255,0.05)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_confusion_matrix_plotly(cm: list, model_name: str):
    """Render interactive confusion matrix with Plotly."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils import ACCENT_MAP, ACCENT_CLASSES
    
    labels = [ACCENT_MAP[a]['label'] for a in ACCENT_CLASSES]
    cm_array = np.array(cm)
    
    # Normalize
    cm_norm = cm_array.astype(float) / cm_array.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    
    text = [[f'{cm_array[i][j]}<br>({cm_norm[i][j]:.1%})'
             for j in range(len(labels))] for i in range(len(labels))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=labels,
        y=labels,
        text=text,
        texttemplate='%{text}',
        textfont=dict(size=10, color='white'),
        colorscale=[
            [0, '#0a0a1a'],
            [0.5, '#667eea'],
            [1, '#764ba2']
        ],
        showscale=True,
        colorbar=dict(title='Ratio'),
    ))
    
    fig.update_layout(
        title=dict(text=f'{model_name} — Confusion Matrix',
                   font=dict(color='#e2e8f0', size=16)),
        xaxis_title='Predicted',
        yaxis_title='Actual',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.02)',
        font=dict(color='#a0aec0', size=11),
        height=500,
        xaxis=dict(tickangle=45),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_gradient_divider():
    """Render a gradient divider line."""
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


def render_transcription_card(text: str, language: str = 'en', confidence: float = 0.0):
    """Render a styled card displaying transcribed speech text."""
    lang_display = {
        'en': '🇬🇧 English', 'es': '🇪🇸 Spanish', 'fr': '🇫🇷 French',
        'de': '🇩🇪 German', 'it': '🇮🇹 Italian', 'pt': '🇵🇹 Portuguese',
        'hi': '🇮🇳 Hindi', 'zh': '🇨🇳 Chinese', 'ja': '🇯🇵 Japanese',
        'ko': '🇰🇷 Korean', 'ar': '🇸🇦 Arabic', 'ru': '🇷🇺 Russian',
    }.get(language, f'🌐 {language.upper()}')
    
    conf_color = '#4ECDC4' if confidence > 0.5 else '#FFEAA7' if confidence > 0.3 else '#FF6B6B'
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(78, 205, 196, 0.08) 0%, rgba(69, 183, 209, 0.08) 100%);
        border: 1px solid rgba(78, 205, 196, 0.25);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(15px);
        animation: fadeInUp 0.6s ease-out;
    ">
        <div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 1rem;">
            <span style="font-size: 1.8rem;">📝</span>
            <h3 style="color: #e2e8f0; margin: 0; font-size: 1.3rem;">Transcribed Text</h3>
        </div>
        <div style="
            background: rgba(0, 0, 0, 0.2);
            border-radius: 14px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.05);
        ">
            <p style="
                color: #e2e8f0;
                font-size: 1.15rem;
                line-height: 1.8;
                margin: 0;
                font-style: italic;
                font-weight: 400;
            ">"{text}"</p>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
            <span style="color: #a0aec0; font-size: 0.9rem;">
                Language: <strong style="color: #e2e8f0;">{lang_display}</strong>
            </span>
            <span style="color: {conf_color}; font-size: 0.9rem; font-weight: 600;">
                Confidence: {confidence:.0%}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_recording_status(is_recording: bool = False):
    """Render an animated recording status indicator."""
    if is_recording:
        st.markdown("""
        <div style="
            display: flex; align-items: center; gap: 0.8rem;
            padding: 0.8rem 1.2rem;
            background: rgba(255, 107, 107, 0.1);
            border: 1px solid rgba(255, 107, 107, 0.3);
            border-radius: 12px;
            margin: 0.5rem 0;
        ">
            <div style="
                width: 12px; height: 12px;
                background: #FF6B6B;
                border-radius: 50%;
                animation: pulse 1.5s ease-in-out infinite;
            "></div>
            <span style="color: #FF6B6B; font-weight: 600;">Recording...</span>
        </div>
        <style>
            @keyframes pulse {
                0%, 100% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.5; transform: scale(1.2); }
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            display: flex; align-items: center; gap: 0.8rem;
            padding: 0.8rem 1.2rem;
            background: rgba(78, 205, 196, 0.08);
            border: 1px solid rgba(78, 205, 196, 0.2);
            border-radius: 12px;
            margin: 0.5rem 0;
        ">
            <span style="font-size: 1.2rem;">🎙️</span>
            <span style="color: #4ECDC4; font-weight: 500;">Ready to record</span>
        </div>
        """, unsafe_allow_html=True)


def render_speech_analysis_card(accent_result: dict, stt_result: dict):
    """Render a combined speech analysis results card."""
    st.markdown(f"""
    <div class="glass-card" style="border-color: rgba(102, 126, 234, 0.3);">
        <h3 style="color: #e2e8f0; margin-bottom: 1.2rem;">
            🔬 Speech Analysis Summary
        </h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
            <div style="
                background: rgba(102, 126, 234, 0.08);
                border-radius: 14px;
                padding: 1.2rem;
                border: 1px solid rgba(102, 126, 234, 0.15);
            ">
                <p style="color: #a0aec0; font-size: 0.8rem; text-transform: uppercase;
                           letter-spacing: 0.05em; margin: 0 0 0.5rem 0;">Detected Accent</p>
                <p style="color: #e2e8f0; font-size: 1.3rem; font-weight: 700; margin: 0;">
                    {accent_result.get('flag', '🌐')} {accent_result.get('accent', 'Unknown')}
                </p>
                <p style="color: #a0aec0; font-size: 0.9rem; margin: 0.3rem 0 0 0;">
                    Confidence: {accent_result.get('confidence', 0):.1%}
                </p>
            </div>
            <div style="
                background: rgba(78, 205, 196, 0.08);
                border-radius: 14px;
                padding: 1.2rem;
                border: 1px solid rgba(78, 205, 196, 0.15);
            ">
                <p style="color: #a0aec0; font-size: 0.8rem; text-transform: uppercase;
                           letter-spacing: 0.05em; margin: 0 0 0.5rem 0;">Speech Language</p>
                <p style="color: #e2e8f0; font-size: 1.3rem; font-weight: 700; margin: 0;">
                    {stt_result.get('language', 'unknown').upper()}
                </p>
                <p style="color: #a0aec0; font-size: 0.9rem; margin: 0.3rem 0 0 0;">
                    Words: {len(stt_result.get('text', '').split())}
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
