import io
import time
from pathlib import Path
from textwrap import dedent

import keras
import numpy as np
import streamlit as st
from PIL import Image
from keras.applications.resnet50 import preprocess_input


CLASS_NAMES = ["Immature", "Mature", "Normal"]

CLASS_META = {
    "Normal": {
        "bar_color": "#22C55E",
        "css_class": "normal",
        "description": "Tidak terlihat tanda kekeruhan pada gambar.",
    },
    "Immature": {
        "bar_color": "#F59E0B",
        "css_class": "immature",
        "description": (
            "Terlihat kekeruhan sebagian pada lensa. "
            "Sebaiknya lakukan pemeriksaan lebih lanjut."
        ),
    },
    "Mature": {
        "bar_color": "#EF4444",
        "css_class": "mature",
        "description": (
            "Terlihat kekeruhan yang cukup jelas pada lensa. "
            "Disarankan berkonsultasi dengan dokter mata."
        ),
    },
}

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.keras"


def render_page_config() -> None:
    st.set_page_config(
        page_title="CataractAI",
        page_icon="👁️",
        layout="centered",
        initial_sidebar_state="collapsed",
    )


def render_styles() -> None:
    st.markdown(
        dedent(
            """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Syne:wght@700;800&display=swap');

            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
            }

            html,
            body,
            [data-testid="stAppViewContainer"],
            [data-testid="stMain"] {
                background: #F8FAFC;
            }

            #MainMenu,
            footer,
            header,
            [data-testid="stHeader"],
            [data-testid="stHeaderActionElements"],
            [data-testid="stToolbar"] {
                display: none !important;
            }

            [data-testid="stAppViewBlockContainer"],
            .block-container {
                width: 100% !important;
                max-width: 620px !important;
                padding: 0.55rem 0.75rem 1.15rem !important;
            }

            .hero-card {
                margin: 0 0 0.55rem !important;
                padding: 0.78rem 1rem 0.72rem !important;
                border: 1px solid #E2E8F0;
                border-radius: 13px;
                background: #FFFFFF;
                box-shadow: 0 4px 14px rgba(15, 23, 42, 0.045);
                text-align: center;
            }

            .hero-eyebrow {
                margin: 0 0 0.2rem !important;
                color: #7183A1 !important;
                font-size: 0.54rem !important;
                font-weight: 600;
                line-height: 1.2;
                letter-spacing: 0.13em;
                text-transform: uppercase;
            }

            .hero-title {
                margin: 0 !important;
                color: #172033 !important;
                font-family: 'Syne', sans-serif !important;
                font-size: 1.65rem !important;
                font-weight: 800 !important;
                line-height: 1 !important;
            }

            .hero-title span {
                color: #2563EB !important;
            }

            .hero-subtitle {
                max-width: 455px;
                margin: 0.35rem auto 0 !important;
                color: #64748B !important;
                font-size: 0.72rem !important;
                line-height: 1.35 !important;
            }

            [data-testid="stFileUploader"] {
                width: 100% !important;
                margin: 0 auto 0.1rem !important;
            }

            [data-testid="stFileUploader"] section,
            [data-testid="stFileUploaderDropzone"] {
                min-height: 58px !important;
                padding: 0.35rem 0.5rem !important;
                border: 1.2px dashed #B8C6D9 !important;
                border-radius: 10px !important;
                background: #FFFFFF !important;
                box-shadow: 0 3px 10px rgba(15, 23, 42, 0.03) !important;
            }

            [data-testid="stFileUploader"] section:hover,
            [data-testid="stFileUploaderDropzone"]:hover {
                border-color: #2563EB !important;
                background: #F7FAFF !important;
            }

            [data-testid="stFileUploader"] section > div,
            [data-testid="stFileUploaderDropzone"] > div {
                min-height: 0 !important;
                gap: 0.35rem !important;
                padding: 0 !important;
            }

            [data-testid="stFileUploaderDropzoneInstructions"] {
                gap: 0.05rem !important;
            }

            [data-testid="stFileUploader"] svg {
                width: 1.35rem !important;
                height: 1.35rem !important;
            }

            [data-testid="stFileUploader"] p,
            [data-testid="stFileUploader"] small,
            [data-testid="stFileUploader"] span,
            [data-testid="stFileUploaderDropzoneInstructions"] * {
                margin: 0 !important;
                color: #64748B !important;
                font-size: 0.66rem !important;
                line-height: 1.15 !important;
            }

            [data-testid="stFileUploader"] button,
            [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {
                width: auto !important;
                min-width: 74px !important;
                min-height: 26px !important;
                height: 26px !important;
                padding: 0 0.48rem !important;
                border: 1px solid #CBD5E1 !important;
                border-radius: 7px !important;
                background: #FFFFFF !important;
                color: #334155 !important;
                font-size: 0.65rem !important;
                font-weight: 500 !important;
                line-height: 1 !important;
                white-space: nowrap !important;
            }

            [data-testid="stFileUploader"] button *,
            [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] * {
                color: inherit !important;
                font-size: 0.65rem !important;
                line-height: 1 !important;
            }

            [data-testid="stFileUploader"] button:hover,
            [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]:hover {
                border-color: #2563EB !important;
                color: #2563EB !important;
            }

            [data-testid="stImage"] img {
                display: block;
                width: 100%;
                max-height: 300px;
                margin: 0.05rem auto 0;
                object-fit: contain;
                border: 1px solid #E2E8F0;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
            }

            .result-card {
                margin-top: 0.05rem;
                padding: 0.78rem 0.8rem 0.72rem;
                border: 1px solid #E2E8F0;
                border-radius: 10px;
                background: #FFFFFF;
                box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
            }

            .result-label {
                margin-bottom: 0.14rem;
                color: #7C8AA0;
                font-size: 0.53rem;
                font-weight: 600;
                letter-spacing: 0.11em;
                text-transform: uppercase;
            }

            .result-class {
                margin-bottom: 0.36rem;
                font-family: 'Syne', sans-serif;
                font-size: 1.48rem;
                font-weight: 800;
                line-height: 1;
            }

            .result-class.normal { color: #16A34A; }
            .result-class.immature { color: #D97706; }
            .result-class.mature { color: #DC2626; }

            .confidence-row,
            .probability-row {
                display: flex;
                align-items: center;
                gap: 0.35rem;
            }

            .confidence-row { margin-bottom: 0.58rem; }
            .probability-row { margin-bottom: 0.28rem; }

            .confidence-value {
                min-width: 67px;
                color: #334155;
                font-size: 0.59rem;
                font-weight: 600;
            }

            .probability-name {
                width: 56px;
                flex-shrink: 0;
                color: #334155;
                font-size: 0.59rem;
            }

            .probability-value {
                width: 34px;
                flex-shrink: 0;
                color: #64748B;
                font-size: 0.57rem;
                text-align: right;
            }

            .progress-track {
                flex: 1;
                height: 4px;
                overflow: hidden;
                border-radius: 999px;
                background: #E9EEF5;
            }

            .progress-fill {
                height: 100%;
                border-radius: 999px;
            }

            .breakdown-title {
                margin-bottom: 0.35rem;
                color: #7C8AA0;
                font-size: 0.52rem;
                font-weight: 600;
                letter-spacing: 0.09em;
                text-transform: uppercase;
            }

            .description-box {
                margin-top: 0.48rem;
                padding: 0.46rem 0.55rem;
                border-radius: 7px;
                color: #334155;
                font-size: 0.61rem;
                line-height: 1.35;
            }

            .description-box.normal {
                border: 1px solid #BBF7D0;
                background: #F0FDF4;
            }

            .description-box.immature {
                border: 1px solid #FDE68A;
                background: #FFFBEB;
            }

            .description-box.mature {
                border: 1px solid #FECACA;
                background: #FEF2F2;
            }

            .disclaimer {
                margin: 0.5rem auto 0 !important;
                color: #94A3B8 !important;
                font-size: 0.56rem !important;
                line-height: 1.35 !important;
                text-align: center;
            }

            [data-testid="stSpinner"] {
                font-size: 0.68rem !important;
            }

            div[data-testid="stHorizontalBlock"] {
                gap: 0.65rem !important;
            }

            @media (max-width: 640px) {
                [data-testid="stAppViewBlockContainer"],
                .block-container {
                    max-width: 100% !important;
                    padding: 0.4rem 0.5rem 0.85rem !important;
                }

                .hero-card {
                    padding: 0.68rem 0.72rem 0.62rem !important;
                    border-radius: 11px;
                }

                .hero-title { font-size: 1.45rem !important; }
                .hero-subtitle { font-size: 0.66rem !important; }

                [data-testid="stFileUploader"] section,
                [data-testid="stFileUploaderDropzone"] {
                    min-height: 54px !important;
                    padding: 0.3rem 0.42rem !important;
                }

                [data-testid="stFileUploader"] button,
                [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {
                    min-width: 68px !important;
                    height: 24px !important;
                    min-height: 24px !important;
                    font-size: 0.61rem !important;
                }

                [data-testid="stImage"] img { max-height: 250px; }
                .result-card { padding: 0.68rem; }
                .result-class { font-size: 1.32rem; }
            }
            </style>
            """
        ),
        unsafe_allow_html=True,
    )


def render_polish_styles() -> None:
    """Visual overrides kept separate from the inference-related code."""
    st.markdown(
        dedent(
            """
            <style>
            :root {
                --ink: #10233f;
                --muted: #667892;
                --line: #dce5f0;
                --primary: #2563eb;
            }

            html, body, [class*="css"] {
                font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
                    "Segoe UI", sans-serif;
            }

            html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
                background: #f7f8fa;
            }

            [data-testid="stDecoration"] { display: none !important; }

            [data-testid="stAppViewBlockContainer"], .block-container {
                max-width: 720px !important;
                padding: 1.25rem 1rem 1.5rem !important;
            }

            .hero-card {
                position: relative;
                margin: 0 0 .9rem !important;
                padding: 1.35rem 1.5rem 1.3rem !important;
                border: 1px solid #e1e6ed;
                border-radius: 14px;
                background: #fff;
                box-shadow: 0 5px 18px rgba(43, 58, 78, .045);
            }

            .hero-card::before { display: none; }

            .brand-mark {
                display: inline-grid;
                place-items: center;
                display: none;
            }

            .brand-mark svg { width: 23px; height: 23px; }
            .hero-eyebrow {
                margin: 0 0 .35rem !important;
                color: #7b8798 !important;
                font-size: .63rem !important;
                font-weight: 650;
                letter-spacing: .11em;
            }

            .hero-title {
                font-family: inherit !important;
                font-size: 2rem !important;
                font-weight: 750 !important;
                letter-spacing: -.045em;
            }

            .hero-subtitle {
                max-width: 500px;
                margin: .55rem auto 0 !important;
                color: var(--muted) !important;
                font-size: .82rem !important;
                line-height: 1.5 !important;
            }

            .upload-heading {
                display: flex;
                align-items: flex-end;
                justify-content: space-between;
                gap: 1rem;
                max-width: 460px;
                margin: 1rem auto .5rem;
            }

            .upload-title { color: var(--ink); font-size: .86rem; font-weight: 700; }
            .upload-note { color: #8493a8; font-size: .7rem; white-space: nowrap; }

            [data-testid="stFileUploader"] {
                width: 100% !important;
                max-width: 460px !important;
                margin: 0 auto !important;
            }
            [data-testid="stFileUploader"] section,
            [data-testid="stFileUploaderDropzone"] {
                min-height: 96px !important;
                padding: .75rem 1rem !important;
                border: 1px dashed #bdc8d6 !important;
                border-radius: 12px !important;
                background: #fff !important;
                box-shadow: 0 4px 14px rgba(43, 58, 78, .035) !important;
                transition: .2s ease !important;
            }

            [data-testid="stFileUploader"] section:hover,
            [data-testid="stFileUploaderDropzone"]:hover {
                border-color: var(--primary) !important;
                background: #f8fbff !important;
                transform: translateY(-1px);
            }

            [data-testid="stFileUploader"] section > div,
            [data-testid="stFileUploaderDropzone"] > div {
                min-height: 66px !important;
                gap: .75rem !important;
            }

            [data-testid="stFileUploaderDropzoneInstructions"] { gap: .3rem !important; }
            [data-testid="stFileUploader"] svg {
                width: 1.5rem !important;
                height: 1.5rem !important;
                color: #667892 !important;
            }

            [data-testid="stFileUploader"] p {
                color: #334a68 !important;
                font-size: .8rem !important;
                font-weight: 600 !important;
            }

            [data-testid="stFileUploader"] small {
                color: #8998ab !important;
                font-size: .69rem !important;
            }

            [data-testid="stFileUploader"] button,
            [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {
                min-width: auto !important;
                min-height: 34px !important;
                height: 34px !important;
                padding: 0 .8rem !important;
                border: 1px solid #c9d3df !important;
                border-radius: 8px !important;
                color: #2563eb !important;
                font-size: .73rem !important;
                font-weight: 650 !important;
                box-shadow: 0 3px 8px rgba(32, 57, 87, .06) !important;
            }

            [data-testid="stFileUploader"] button * { font-size: .73rem !important; }
            [data-testid="stFileUploaderFile"] {
                padding: .8rem 1rem !important;
                border-radius: 12px !important;
                background: #fff !important;
            }

            [data-testid="stImage"] img {
                height: 300px;
                max-height: none;
                object-fit: contain;
                padding: .6rem;
                border: 1px solid var(--line);
                border-radius: 12px;
                background: #fff;
                box-shadow: 0 5px 18px rgba(43, 58, 78, .045);
            }

            .result-card {
                min-height: 300px;
                margin-top: 0;
                padding: 1.15rem;
                border: 1px solid var(--line);
                border-radius: 12px;
                box-shadow: 0 5px 18px rgba(43, 58, 78, .045);
            }

            .result-label, .breakdown-title {
                color: #8493a8;
                font-size: .68rem;
                font-weight: 750;
                letter-spacing: .12em;
            }

            .result-label { margin-bottom: .5rem; }
            .result-class {
                margin-bottom: 1rem;
                font-family: inherit;
                font-size: 1.75rem;
                font-weight: 760;
                letter-spacing: -.04em;
            }

            .confidence-row { margin-bottom: 1rem; }
            .probability-row { margin-bottom: .45rem; }
            .confidence-value { min-width: 92px; font-size: .7rem; font-weight: 700; }
            .probability-name { width: 62px; font-size: .69rem; }
            .probability-value { width: 38px; font-size: .67rem; }
            .progress-track { height: 5px; }
            .breakdown-title { margin-bottom: .55rem; }
            .description-box {
                margin-top: .75rem;
                padding: .65rem .75rem;
                border-radius: 8px;
                font-size: .69rem;
                line-height: 1.45;
            }

            .disclaimer {
                margin: .85rem auto 0 !important;
                padding-top: .75rem;
                border-top: 1px solid #e2e8f0;
                font-size: .67rem !important;
                line-height: 1.55 !important;
            }

            div[data-testid="stHorizontalBlock"] { gap: 1rem !important; }

            @media (max-width: 640px) {
                [data-testid="stAppViewBlockContainer"], .block-container {
                    padding: .75rem .65rem 1rem !important;
                }
                .hero-card { padding: 1.15rem .9rem !important; border-radius: 12px; }
                .hero-title { font-size: 1.75rem !important; }
                .hero-subtitle { font-size: .76rem !important; line-height: 1.45 !important; }
                .upload-heading { align-items: flex-start; flex-direction: column; gap: .2rem; }
                .upload-note { white-space: normal; }
                [data-testid="stFileUploader"] section,
                [data-testid="stFileUploaderDropzone"] { min-height: 90px !important; padding: .65rem !important; }
                [data-testid="stFileUploader"] section > div,
                [data-testid="stFileUploaderDropzone"] > div { gap: .75rem !important; }
                [data-testid="stImage"] img { height: 250px; }
                .result-card { min-height: 0; padding: 1rem; }
                .result-class { font-size: 1.65rem; }
            }
            </style>
            """
        ),
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_cataract_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan di: {MODEL_PATH}"
        )

    try:
        return keras.models.load_model(
            str(MODEL_PATH),
            compile=False,
            safe_mode=False,
        )
    except TypeError:
        return keras.models.load_model(
            str(MODEL_PATH),
            compile=False,
        )


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(
        (224, 224),
        Image.Resampling.LANCZOS,
    )

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    image_array = preprocess_input(image_array)

    return np.expand_dims(
        image_array,
        axis=0,
    )


def predict_image(model, image: Image.Image):
    image_tensor = preprocess_image(image)

    predictions = model.predict(
        image_tensor,
        verbose=0,
    )

    predictions = np.asarray(predictions).squeeze()

    if predictions.ndim != 1:
        raise ValueError(
            f"Format output model tidak sesuai: {predictions.shape}"
        )

    if len(predictions) != len(CLASS_NAMES):
        raise ValueError(
            "Jumlah output model tidak sesuai dengan jumlah kelas. "
            f"Output model: {len(predictions)}, "
            f"jumlah kelas: {len(CLASS_NAMES)}."
        )

    predicted_index = int(np.argmax(predictions))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    return predicted_label, confidence, predictions


def render_hero() -> None:
    html = (
        '<div class="hero-card">'
        '<div class="hero-eyebrow">Klasifikasi foto mata</div>'
        '<h1 class="hero-title">Cataract<span>AI</span></h1>'
        '<div class="hero-subtitle">Unggah foto mata untuk melihat hasil klasifikasi '
        'Normal, Immature, atau Mature.</div>'
        '</div>'
    )

    st.markdown(html, unsafe_allow_html=True)



def render_result(
    label: str,
    confidence: float,
    probabilities: np.ndarray,
) -> None:
    metadata = CLASS_META[label]
    confidence_percent = confidence * 100
    probability_rows = []

    for class_name, probability in zip(CLASS_NAMES, probabilities):
        class_metadata = CLASS_META[class_name]
        probability_percent = float(probability) * 100
        font_weight = "600" if class_name == label else "500"

        probability_rows.append(
            dedent(
                f"""
                <div class="probability-row">
                    <span class="probability-name" style="font-weight: {font_weight};">{class_name}</span>
                    <div class="progress-track">
                        <div class="progress-fill" style="width: {probability_percent:.2f}%; background: {class_metadata['bar_color']};"></div>
                    </div>
                    <span class="probability-value">{probability_percent:.1f}%</span>
                </div>
                """
            ).strip()
        )

    result_html = dedent(
        f"""
        <div class="result-card">
            <div class="result-label">HASIL</div>
            <div class="result-class {metadata['css_class']}">{label}</div>
            <div class="confidence-row">
                <span class="confidence-value">Keyakinan {confidence_percent:.1f}%</span>
                <div class="progress-track">
                    <div class="progress-fill" style="width: {confidence_percent:.2f}%; background: {metadata['bar_color']};"></div>
                </div>
            </div>
            <div class="breakdown-title">RINCIAN HASIL</div>
            {''.join(probability_rows)}
            <div class="description-box {metadata['css_class']}">{metadata['description']}</div>
        </div>
        """
    )

    st.markdown(result_html, unsafe_allow_html=True)


def render_disclaimer() -> None:
    st.markdown(
        dedent(
            """
            <div class="disclaimer">
                Untuk keperluan penelitian dan edukasi. Bukan pengganti diagnosis medis.
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def main() -> None:
    render_page_config()
    render_styles()
    render_polish_styles()
    render_hero()

    try:
        with st.spinner("Memuat model..."):
            model = load_cataract_model()

    except FileNotFoundError as error:
        st.error(
            f"Model tidak ditemukan. {error}",
            icon="⚠️",
        )
        return

    except Exception as error:
        st.error(
            "Model ditemukan, tetapi gagal dimuat. "
            "Kemungkinan versi Keras pada deployment berbeda "
            "dengan versi Keras saat model disimpan.",
            icon="⚠️",
        )

        with st.expander("Lihat detail error"):
            st.write("Versi Keras:", keras.__version__)
            st.write("Lokasi model:", str(MODEL_PATH))
            st.code(str(error))

        return

    st.markdown(
        '<div class="upload-heading">'
        '<div class="upload-title">Pilih foto mata</div>'
        '<div class="upload-note">JPG, JPEG, atau PNG</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        label="Foto Mata",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="Format gambar yang didukung: JPG, JPEG, dan PNG.",
    )

    if uploaded_file is None:
        render_disclaimer()
        return

    try:
        image = Image.open(
            io.BytesIO(uploaded_file.getvalue())
        )

    except Exception:
        st.error(
            "File gambar tidak dapat dibaca. "
            "Silakan unggah gambar JPG, JPEG, atau PNG yang valid."
        )
        return

    image_col, result_col = st.columns([1, 1], gap="small")

    with image_col:
        st.image(
            image,
            use_container_width=True,
        )

    try:
        with st.spinner("Menganalisis gambar..."):
            time.sleep(0.4)

            label, confidence, probabilities = predict_image(
                model,
                image,
            )

    except Exception as error:
        st.error(
            "Terjadi kesalahan saat melakukan prediksi."
        )

        with st.expander("Lihat detail error"):
            st.code(str(error))

        return

    with result_col:
        render_result(
            label,
            confidence,
            probabilities,
        )

    render_disclaimer()


if __name__ == "__main__":
    main()
