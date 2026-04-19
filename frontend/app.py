"""
Interactive Dashboard — Audio Compression in Real Time
Built with Plotly Dash.

Features:
  • Upload audio file OR record via microphone (WebRTC bridge)
  • Select codec (MP3 / AAC / Opus / OGG) and bitrate slider
  • Live bitrate / latency line plots (updates every second)
  • Waveform comparison: original vs reconstructed
  • Spectrogram comparison panel
  • Metrics table: SNR, PSNR, compression ratio, space saving
  • Playback: original vs compressed buttons
"""

import base64
import io
import json
import os
import time
import wave
import asyncio
import threading
import logging

import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import websockets

from backend.audio_codec import AudioCodec
from backend.metrics import MetricsCalculator

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = dash.Dash(
    __name__,
    title="Audio Compression Dashboard",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

SAMPLE_RATE = 44100
CODECS = ["mp3", "aac", "opus", "ogg"]
BITRATES = [32, 64, 96, 128, 192, 256, 320]
COLORS = {"mp3": "#636EFA", "aac": "#EF553B", "opus": "#00CC96", "ogg": "#AB63FA"}

codec_obj = AudioCodec()
metrics_calc = MetricsCalculator(sample_rate=SAMPLE_RATE)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
CARD = {
    "background": "#1e1e2e",
    "border": "1px solid #383854",
    "borderRadius": "12px",
    "padding": "20px",
    "marginBottom": "18px",
}

app.layout = html.Div(
    style={"background": "#13131f", "minHeight": "100vh", "fontFamily": "Inter, sans-serif", "color": "#e0e0ef"},
    children=[
        # ── Header ──────────────────────────────────────────────────────
        html.Div(
            style={"background": "linear-gradient(135deg,#1a1a3e,#0d0d1a)", "padding": "24px 40px",
                   "borderBottom": "1px solid #383854"},
            children=[
                html.H1("🎙 Audio Compression Dashboard",
                        style={"margin": 0, "fontSize": "26px", "fontWeight": 700,
                               "background": "linear-gradient(90deg,#818cf8,#c084fc)",
                               "WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent"}),
                html.P("Real-time codec comparison · SNR · Bitrate · Latency · Spectrogram",
                       style={"margin": "4px 0 0", "color": "#8888aa", "fontSize": "13px"}),
            ]
        ),

        html.Div(
            style={"maxWidth": "1400px", "margin": "0 auto", "padding": "24px 20px"},
            children=[

                # ── Controls row ─────────────────────────────────────────
                html.Div(style={**CARD, "display": "flex", "flexWrap": "wrap", "gap": "24px",
                                "alignItems": "flex-end"},
                    children=[
                        # Upload
                        html.Div([
                            html.Label("Upload Audio (WAV)", style={"fontSize": "12px", "color": "#8888aa"}),
                            dcc.Upload(
                                id="upload-audio",
                                children=html.Div(["📂 Drag & Drop or ", html.B("Browse")]),
                                style={"border": "2px dashed #383854", "borderRadius": "8px",
                                       "padding": "12px 20px", "cursor": "pointer",
                                       "textAlign": "center", "fontSize": "13px",
                                       "background": "#0d0d1a", "color": "#aaaacc"},
                                accept=".wav,.mp3,.ogg,.flac",
                                max_size=50 * 1024 * 1024,
                            ),
                        ], style={"flex": "1", "minWidth": "200px"}),

                        # Codec selector
                        html.Div([
                            html.Label("Codec", style={"fontSize": "12px", "color": "#8888aa"}),
                            dcc.Checklist(
                                id="codec-select",
                                options=[{"label": f" {c.upper()}", "value": c} for c in CODECS],
                                value=["mp3", "opus"],
                                inline=True,
                                inputStyle={"marginRight": "4px"},
                                labelStyle={"marginRight": "16px", "fontSize": "13px"},
                            ),
                        ], style={"flex": "1", "minWidth": "220px"}),

                        # Bitrate slider
                        html.Div([
                            html.Label("Bitrate (kbps)", style={"fontSize": "12px", "color": "#8888aa"}),
                            dcc.Slider(
                                id="bitrate-slider",
                                min=32, max=320, step=None,
                                marks={b: {"label": str(b), "style": {"color": "#8888aa", "fontSize": "11px"}}
                                       for b in BITRATES},
                                value=128,
                                tooltip={"always_visible": True, "placement": "top"},
                            ),
                        ], style={"flex": "2", "minWidth": "280px"}),

                        # Analyze button
                        html.Button("▶ Analyze", id="analyze-btn",
                                    style={"background": "linear-gradient(135deg,#6366f1,#8b5cf6)",
                                           "color": "#fff", "border": "none", "borderRadius": "8px",
                                           "padding": "10px 28px", "fontSize": "14px",
                                           "fontWeight": 600, "cursor": "pointer"}),
                    ]
                ),

                # ── Status bar ───────────────────────────────────────────
                html.Div(id="status-bar",
                         style={"background": "#0d0d1a", "borderRadius": "8px",
                                "padding": "10px 16px", "marginBottom": "18px",
                                "fontSize": "13px", "color": "#8888aa", "border": "1px solid #383854"}),

                # ── Metric cards ─────────────────────────────────────────
                html.Div(id="metric-cards",
                         style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(180px,1fr))",
                                "gap": "12px", "marginBottom": "18px"}),

                # ── Charts ───────────────────────────────────────────────
                html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "18px"},
                    children=[
                        html.Div(style=CARD, children=[
                            html.H3("Waveform Comparison", style={"margin": "0 0 12px", "fontSize": "14px",
                                                                   "color": "#c084fc"}),
                            dcc.Graph(id="waveform-chart", style={"height": "280px"},
                                      config={"displayModeBar": False}),
                        ]),
                        html.Div(style=CARD, children=[
                            html.H3("SNR vs Bitrate", style={"margin": "0 0 12px", "fontSize": "14px",
                                                              "color": "#c084fc"}),
                            dcc.Graph(id="snr-bitrate-chart", style={"height": "280px"},
                                      config={"displayModeBar": False}),
                        ]),
                        html.Div(style=CARD, children=[
                            html.H3("Compression Ratio & Space Saving", style={"margin": "0 0 12px",
                                                                                "fontSize": "14px",
                                                                                "color": "#c084fc"}),
                            dcc.Graph(id="compression-chart", style={"height": "280px"},
                                      config={"displayModeBar": False}),
                        ]),
                        html.Div(style=CARD, children=[
                            html.H3("Encode / Decode Latency (ms)", style={"margin": "0 0 12px",
                                                                            "fontSize": "14px",
                                                                            "color": "#c084fc"}),
                            dcc.Graph(id="latency-chart", style={"height": "280px"},
                                      config={"displayModeBar": False}),
                        ]),
                    ]),

                # ── Spectrogram row ───────────────────────────────────────
                html.Div(style={**CARD, "marginTop": "18px"}, children=[
                    html.H3("Spectrogram Comparison", style={"margin": "0 0 12px", "fontSize": "14px",
                                                              "color": "#c084fc"}),
                    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
                        children=[
                            dcc.Graph(id="spectrogram-orig", style={"height": "260px"},
                                      config={"displayModeBar": False}),
                            dcc.Graph(id="spectrogram-comp", style={"height": "260px"},
                                      config={"displayModeBar": False}),
                        ]
                    ),
                    html.Div(style={"display": "flex", "gap": "12px", "marginTop": "12px"},
                        children=[
                            html.Div([
                                html.Label("Compare Codec", style={"fontSize": "12px", "color": "#8888aa"}),
                                dcc.Dropdown(id="spec-codec-select",
                                             options=[{"label": c.upper(), "value": c} for c in CODECS],
                                             value="mp3",
                                             style={"background": "#0d0d1a", "color": "#e0e0ef",
                                                    "minWidth": "120px"}),
                            ]),
                        ]
                    ),
                ]),

                # ── Playback row ─────────────────────────────────────────
                html.Div(style={**CARD, "marginTop": "18px"}, children=[
                    html.H3("Playback", style={"margin": "0 0 12px", "fontSize": "14px", "color": "#c084fc"}),
                    html.Div(style={"display": "flex", "gap": "24px", "flexWrap": "wrap"},
                        children=[
                            html.Div([
                                html.P("Original", style={"fontSize": "12px", "color": "#8888aa", "margin": "0 0 6px"}),
                                html.Audio(id="audio-original", controls=True,
                                           style={"width": "340px", "filter": "invert(1) hue-rotate(180deg)"}),
                            ]),
                            html.Div([
                                html.P("Compressed (selected codec + bitrate)", style={"fontSize": "12px",
                                                                                        "color": "#8888aa",
                                                                                        "margin": "0 0 6px"}),
                                html.Audio(id="audio-compressed", controls=True,
                                           style={"width": "340px", "filter": "invert(1) hue-rotate(180deg)"}),
                            ]),
                        ]
                    ),
                ]),

                # ── Hidden store ──────────────────────────────────────────
                dcc.Store(id="audio-store"),  
                dcc.Store(id="raw-store"),    

                # ── Live update interval ──────────────────────────────────
                dcc.Interval(id="live-interval", interval=1000, n_intervals=0, disabled=False),
                html.Div(id="live-metrics-container",
                         style={"marginTop": "18px"},
                         children=[
                             html.Div(style=CARD, children=[
                                 html.H3("Live Stream Metrics", style={"margin": "0 0 12px", "fontSize": "14px",
                                                                        "color": "#c084fc"}),
                                 dcc.Graph(id="live-chart", style={"height": "200px"},
                                           config={"displayModeBar": False}),
                             ])
                         ]),
                dcc.Store(id="live-store", data={"snr": [], "latency": [], "t": []}),
            ]
        )
    ]
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c0c0d8", size=11),
    xaxis=dict(gridcolor="#2a2a3e", zerolinecolor="#2a2a3e"),
    yaxis=dict(gridcolor="#2a2a3e", zerolinecolor="#2a2a3e"),
    margin=dict(l=50, r=20, t=30, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#383854"),
)


def decode_upload(contents: str) -> tuple[np.ndarray, bytes]:
    header, b64data = contents.split(",", 1)
    raw = base64.b64decode(b64data)

    buf = io.BytesIO(raw)
    try:
        with wave.open(buf, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw_pcm = wf.readframes(n_frames)
    except Exception:
        try:
            import soundfile as sf
            buf.seek(0)
            data, sr = sf.read(buf, dtype="float32")
            samples = data[:, 0] if data.ndim > 1 else data
        except Exception as e:
            raise ValueError(f"Cannot decode audio: {e}")
        
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((samples * 32767).astype(np.int16).tobytes())
        return samples, wav_buf.getvalue()

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth, np.int16)
    arr = np.frombuffer(raw_pcm, dtype=dtype).astype(np.float32)
    if sampwidth > 1:
        arr /= (2 ** (8 * sampwidth - 1))
    if n_channels > 1:
        arr = arr[::n_channels]

    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((arr * 32767).astype(np.int16).tobytes())
    return arr, wav_buf.getvalue()


def run_full_analysis(samples: np.ndarray, selected_codecs: list, bitrate: int) -> dict:
    duration = len(samples) / SAMPLE_RATE
    results = {"by_codec_bitrate": {}, "at_selected_bitrate": {}, "duration_s": duration}

    sweep_bitrates = [32, 64, 96, 128, 192, 256, 320]

    for codec in selected_codecs:
        results["by_codec_bitrate"][codec] = {}
        for br in sweep_bitrates:
            try:
                import time as _time
                t0 = _time.perf_counter()
                compressed = codec_obj.compress(samples, codec, bitrate=br)
                t1 = _time.perf_counter()
                reconstructed = codec_obj.decompress(compressed, codec)
                t2 = _time.perf_counter()

                min_len = min(len(samples), len(reconstructed))
                o, r = samples[:min_len], reconstructed[:min_len]

                results["by_codec_bitrate"][codec][br] = {
                    "snr_db": round(metrics_calc.snr(o, r), 2),
                    "psnr_db": round(metrics_calc.psnr(o, r), 2),
                    "compression_ratio": round(metrics_calc.compression_ratio(len(samples)*4, len(compressed)), 3),
                    "space_saving_pct": round(metrics_calc.space_saving(len(samples)*4, len(compressed)), 1),
                    "effective_bitrate_kbps": round(metrics_calc.bitrate_kbps(len(compressed), duration), 1),
                    "encode_latency_ms": round((t1 - t0) * 1000, 2),
                    "decode_latency_ms": round((t2 - t1) * 1000, 2),
                }
            except Exception as e:
                results["by_codec_bitrate"][codec][br] = {"error": str(e)}

        entry = results["by_codec_bitrate"][codec].get(bitrate, {})
        results["at_selected_bitrate"][codec] = entry

    return results


def samples_to_b64_wav(samples: np.ndarray) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((samples * 32767).astype(np.int16).tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def compressed_to_b64(data: bytes, codec: str) -> str:
    mime = {"mp3": "audio/mpeg", "aac": "audio/aac", "opus": "audio/ogg", "ogg": "audio/ogg"}.get(codec, "audio/octet-stream")
    return f"data:{mime};base64," + base64.b64encode(data).decode()


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("audio-store", "data"),
    Output("raw-store", "data"),
    Output("status-bar", "children"),
    Input("analyze-btn", "n_clicks"),
    State("upload-audio", "contents"),
    State("codec-select", "value"),
    State("bitrate-slider", "value"),
    prevent_initial_call=True,
)
def run_analysis(n_clicks, contents, selected_codecs, bitrate):
    if not contents:
        return no_update, no_update, "⚠ Please upload an audio file first."
    if not selected_codecs:
        return no_update, no_update, "⚠ Please select at least one codec."

    try:
        samples, wav_bytes = decode_upload(contents)
    except Exception as e:
        return no_update, no_update, f"Error reading file: {e}"

    max_samples = SAMPLE_RATE * 30
    if len(samples) > max_samples:
        samples = samples[:max_samples]

    try:
        analysis = run_full_analysis(samples, selected_codecs, bitrate)
    except Exception as e:
        return no_update, no_update, f"Analysis error: {e}"

    spec = metrics_calc.compute_spectrogram(samples)
    analysis["spectrogram_orig"] = spec

    compressed_b64_map = {}
    for codec in selected_codecs:
        try:
            comp = codec_obj.compress(samples, codec, bitrate=bitrate)
            compressed_b64_map[codec] = compressed_to_b64(comp, codec)
        except Exception:
            pass
    analysis["compressed_b64"] = compressed_b64_map
    analysis["original_b64"] = samples_to_b64_wav(samples)

    duration_s = len(samples) / SAMPLE_RATE
    status = (f"Analysis complete — {duration_s:.1f}s audio, "
              f"{len(selected_codecs)} codec(s), bitrate sweep 32–320 kbps")
    return analysis, base64.b64encode(wav_bytes).decode(), status


@app.callback(
    Output("metric-cards", "children"),
    Input("audio-store", "data"),
    State("bitrate-slider", "value"),
    prevent_initial_call=True,
)
def update_metric_cards(data, bitrate):
    if not data:
        return []
    at_sel = data.get("at_selected_bitrate", {})
    cards = []
    for codec, m in at_sel.items():
        if "error" in m:
            continue
        col = COLORS.get(codec, "#818cf8")
        cards.append(html.Div(
            style={"background": "#0d0d1a", "border": f"1px solid {col}44",
                   "borderRadius": "10px", "padding": "14px 16px",
                   "borderTop": f"3px solid {col}"},
            children=[
                html.Div(codec.upper(), style={"fontSize": "11px", "color": col, "fontWeight": 700,
                                               "letterSpacing": "1px", "marginBottom": "8px"}),
                html.Div([
                    html.Span("SNR ", style={"color": "#8888aa", "fontSize": "12px"}),
                    html.Span(f"{m.get('snr_db','—')} dB", style={"fontWeight": 600, "fontSize": "14px"}),
                ], style={"marginBottom": "4px"}),
                html.Div([
                    html.Span("PSNR ", style={"color": "#8888aa", "fontSize": "12px"}),
                    html.Span(f"{m.get('psnr_db','—')} dB", style={"fontWeight": 600, "fontSize": "14px"}),
                ], style={"marginBottom": "4px"}),
                html.Div([
                    html.Span("Ratio ", style={"color": "#8888aa", "fontSize": "12px"}),
                    html.Span(f"{m.get('compression_ratio','—')}×", style={"fontWeight": 600, "fontSize": "14px"}),
                ], style={"marginBottom": "4px"}),
                html.Div([
                    html.Span("Saving ", style={"color": "#8888aa", "fontSize": "12px"}),
                    html.Span(f"{m.get('space_saving_pct','—')}%", style={"fontWeight": 600, "fontSize": "14px"}),
                ]),
            ]
        ))
    return cards


@app.callback(
    Output("snr-bitrate-chart", "figure"),
    Output("compression-chart", "figure"),
    Output("latency-chart", "figure"),
    Output("waveform-chart", "figure"),
    Input("audio-store", "data"),
    Input("bitrate-slider", "value"),    
    State("upload-audio", "contents"),
    prevent_initial_call=True,
)
def update_charts(data, bitrate, contents):  
    empty = go.Figure(layout={**DARK_LAYOUT})

    if not data:
        return empty, empty, empty, empty

    by_cb = data.get("by_codec_bitrate", {})
    bitrates_x = [32, 64, 96, 128, 192, 256, 320]

    # ── SNR vs Bitrate ───────────────────────────────────────────────
    fig_snr = go.Figure(layout={**DARK_LAYOUT, "yaxis_title": "SNR (dB)", "xaxis_title": "Bitrate (kbps)"})
    for codec, br_map in by_cb.items():
        snrs = [br_map.get(str(br), {}).get("snr_db") for br in bitrates_x]
        snrs_clean = [v if v is not None else None for v in snrs]
        fig_snr.add_trace(go.Scatter(
            x=bitrates_x, y=snrs_clean, name=codec.upper(),
            mode="lines+markers", line=dict(color=COLORS.get(codec), width=2),
            marker=dict(size=6),
        ))

    # ── Compression Ratio ────────────────────────────────────────────
    fig_comp = go.Figure(layout={**DARK_LAYOUT, "barmode": "group",
                                 "yaxis_title": "Compression Ratio (×)",
                                 "xaxis_title": "Bitrate (kbps)"})
    for codec, br_map in by_cb.items():
        ratios = [br_map.get(str(br), {}).get("compression_ratio") for br in bitrates_x]
        fig_comp.add_trace(go.Bar(
            x=bitrates_x, y=ratios, name=codec.upper(),
            marker_color=COLORS.get(codec),
        ))

    # ── Latency ──────────────────────────────────────────────────────
    br_str = str(bitrate)
    fig_lat = go.Figure(layout={**DARK_LAYOUT, "barmode": "group",
                                "yaxis_title": "Latency (ms)", 
                                "xaxis_title": f"Codec @ {bitrate} kbps"}) 
    codecs_list = list(by_cb.keys())
    
    enc_lats = [by_cb[c].get(br_str, {}).get("encode_latency_ms", 0) for c in codecs_list]
    dec_lats = [by_cb[c].get(br_str, {}).get("decode_latency_ms", 0) for c in codecs_list]
    
    fig_lat.add_trace(go.Bar(x=[c.upper() for c in codecs_list], y=enc_lats,
                             name="Encode", marker_color="#818cf8"))
    fig_lat.add_trace(go.Bar(x=[c.upper() for c in codecs_list], y=dec_lats,
                             name="Decode", marker_color="#f472b6"))

    # ── Waveform ─────────────────────────────────────────────────────
    fig_wave = go.Figure(layout={**DARK_LAYOUT, "yaxis_title": "Amplitude",
                                 "xaxis_title": "Sample"})
    if contents:
        try:
            samples, _ = decode_upload(contents)
            n_plot = min(len(samples), 4000)
            t_ax = np.linspace(0, len(samples) / SAMPLE_RATE, n_plot)
            step = max(1, len(samples) // n_plot)
            orig_plot = samples[::step][:n_plot]
            fig_wave.add_trace(go.Scatter(x=t_ax[:len(orig_plot)], y=orig_plot.tolist(),
                                          name="Original", line=dict(color="#818cf8", width=1)))

            if codecs_list:
                codec = codecs_list[0]
                try:
                    comp = codec_obj.compress(samples, codec, bitrate=bitrate)
                    recon = codec_obj.decompress(comp, codec)
                    recon_plot = recon[::step][:n_plot]
                    fig_wave.add_trace(go.Scatter(
                        x=t_ax[:len(recon_plot)], y=recon_plot.tolist(),
                        name=f"{codec.upper()} (recon)", line=dict(color="#f472b6", width=1)))
                except Exception:
                    pass
        except Exception:
            pass

    return fig_snr, fig_comp, fig_lat, fig_wave


@app.callback(
    Output("spectrogram-orig", "figure"),
    Output("spectrogram-comp", "figure"),
    Input("audio-store", "data"),
    Input("spec-codec-select", "value"), 
    Input("bitrate-slider", "value"),    
    State("upload-audio", "contents"),
    prevent_initial_call=True,
)
def update_spectrograms(data, spec_codec, bitrate, contents):
    empty = go.Figure(layout={**DARK_LAYOUT})
    if not data or not contents:
        return empty, empty

    try:
        samples, _ = decode_upload(contents)
        n_max = SAMPLE_RATE * 10  # 10s for speed
        samples = samples[:n_max]
    except Exception:
        return empty, empty

    def make_spectrogram_fig(spec: dict, title: str):
        if not spec.get("magnitude_db"):
            return empty
        z = spec["magnitude_db"]
        x = spec["time_frames"]
        y = spec["freq_bins"]
        fig = go.Figure(
            go.Heatmap(z=z, x=x, y=y, colorscale="Magma",
                       zmin=-80, zmax=0,
                       colorbar=dict(title="dB", tickfont=dict(size=10))),
            layout={**DARK_LAYOUT,
                    "title": dict(text=title, font=dict(size=12, color="#c084fc")),
                    "xaxis_title": "Time (s)", "yaxis_title": "Freq (Hz)"},
        )
        return fig

    orig_spec = data.get("spectrogram_orig", {})
    fig_orig = make_spectrogram_fig(orig_spec, "Original")

    try:
        comp = codec_obj.compress(samples, spec_codec, bitrate=bitrate)
        recon = codec_obj.decompress(comp, spec_codec)
        comp_spec = metrics_calc.compute_spectrogram(recon)
        fig_comp = make_spectrogram_fig(comp_spec, f"{spec_codec.upper()} @ {bitrate} kbps")
    except Exception as e:
        fig_comp = empty

    return fig_orig, fig_comp


@app.callback(
    Output("audio-original", "src"),
    Output("audio-compressed", "src"),
    Input("audio-store", "data"),
    Input("spec-codec-select", "value"),  
    prevent_initial_call=True
)
def update_audio_playback(data, selected_codec):
    if not data:
        return dash.no_update, dash.no_update
    
    orig_src = data.get("original_b64")
    comp_src = data.get("compressed_b64", {}).get(selected_codec)
    
    return orig_src, comp_src


@app.callback(
    Output("live-chart", "figure"),
    Input("live-interval", "n_intervals")
)
def update_live_chart(n_intervals):
    fig = go.Figure(layout={
        **DARK_LAYOUT, 
        "yaxis_title": "Giá trị (dB / ms)", 
        "xaxis_title": "Time (Chunks)",
        "margin": dict(l=50, r=20, t=20, b=40)
    })

    filepath = "results/client_metrics.json"
    
    if not os.path.exists(filepath):
        return fig

    try:
        with open(filepath, "r") as f:
            history = json.load(f)
        
        if history:
            history = history[-40:]
            x_vals = list(range(len(history)))
            
            snr_mp3 = [item.get("codecs", {}).get("mp3", {}).get("snr_db", None) for item in history]
            snr_opus = [item.get("codecs", {}).get("opus", {}).get("snr_db", None) for item in history]
            latencies = [item.get("total_latency_ms", 0) for item in history]

            fig.add_trace(go.Scatter(x=x_vals, y=snr_mp3, mode="lines+markers", 
                                     name="MP3 SNR (dB)", line=dict(color="#636EFA")))
            fig.add_trace(go.Scatter(x=x_vals, y=snr_opus, mode="lines+markers", 
                                     name="Opus SNR (dB)", line=dict(color="#00CC96")))
            fig.add_trace(go.Scatter(x=x_vals, y=latencies, mode="lines", 
                                     name="Latency (ms)", line=dict(color="#EF553B", dash="dot")))
            
    except Exception:
        pass

    return fig

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)