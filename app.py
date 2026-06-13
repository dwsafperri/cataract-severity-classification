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
        "description": (
            "Lensa mata tampak jernih tanpa tanda-tanda kekeruhan. "
            "Tidak terdeteksi katarak pada gambar ini."
        ),
    },
    "Immature": {
        "bar_color": "#F59E0B",
        "css_class": "immature",
        "description": (
            "Terdeteksi kekeruhan sebagian pada lensa mata. "
            "Katarak masih dalam tahap awal, sehingga pemeriksaan lebih lanjut disarankan."
        ),
    },
    "Mature": {
        "bar_color": "#EF4444",
        "css_class": "mature",
        "description": (
            "Lensa mata menunjukkan kekeruhan yang signifikan. "
            "Konsultasi dengan dokter mata segera dianjurkan."
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
            [data-testid="stAppViewContainer"] {
                background: #F8FAFC;
            }

            #MainMenu,
            footer,
            header,
            [data-testid="stHeaderActionElements"],
            [data-testid="stToolbar"] {
                display: none !important;
            }

            .block-container {
                max-width: 760px;
                padding: 1.25rem 1rem 2.25rem;
            }

            .hero-card {
                margin-bottom: 1.25rem;
                padding: 2rem 2rem 1.8rem;
                border: 1px solid #E5EAF1;
                border-radius: 22px;
                background: #FFFFFF;
                box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06);
                text-align: center;
            }

            .hero-eyebrow {
                margin-bottom: 0.65rem;
                color: #7183A1;
                font-size: 0.72rem;
                font-weight: 600;
                letter-spacing: 0.18em;
                text-transform: uppercase;
            }

            .hero-title {
                margin: 0;
                color: #172033;
                font-family: 'Syne', sans-serif;
                font-size: 2.75rem;
                font-weight: 800;
                line-height: 1.05;
            }

            .hero-title span {
                color: #2563EB;
            }

            .hero-subtitle {
                max-width: 540px;
                margin: 0.95rem auto 0;
                color: #64748B;
                font-size: 0.96rem;
                line-height: 1.65;
            }

            .upload-title {
                margin: 0 0 0.6rem 0.15rem;
                color: #334155;
                font-size: 0.78rem;
                font-weight: 600;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            [data-testid="stFileUploader"] {
                width: 100%;
                margin-bottom: 0.4rem;
            }

            [data-testid="stFileUploader"] section {
                min-height: 120px;
                padding: 1rem 1.2rem;
                border: 1.5px dashed #B8C6D9;
                border-radius: 16px;
                background: #FFFFFF;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
                transition:
                    border-color 0.2s ease,
                    background-color 0.2s ease,
                    transform 0.2s ease;
            }

            [data-testid="stFileUploader"] section:hover {
                border-color: #2563EB;
                background: #F7FAFF;
                transform: translateY(-1px);
            }

            [data-testid="stFileUploader"] button {
                border: 1px solid #CBD5E1;
                border-radius: 10px;
                background: #FFFFFF;
                color: #334155;
                font-weight: 500;
            }

            [data-testid="stFileUploader"] button:hover {
                border-color: #2563EB;
                color: #2563EB;
            }

            [data-testid="stImage"] img {
                margin-top: 0.85rem;
                border: 1px solid #E2E8F0;
                border-radius: 16px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            }

            .result-card {
                margin-top: 1rem;
                padding: 1.5rem;
                border: 1px solid #E2E8F0;
                border-radius: 16px;
                background: #FFFFFF;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            }

            .result-label {
                margin-bottom: 0.35rem;
                color: #7C8AA0;
                font-size: 0.7rem;
                font-weight: 600;
                letter-spacing: 0.14em;
                text-transform: uppercase;
            }

            .result-class {
                margin-bottom: 0.8rem;
                font-family: 'Syne', sans-serif;
                font-size: 2rem;
                font-weight: 800;
                line-height: 1.1;
            }

            .result-class.normal {
                color: #16A34A;
            }

            .result-class.immature {
                color: #D97706;
            }

            .result-class.mature {
                color: #DC2626;
            }

            .confidence-row,
            .probability-row {
                display: flex;
                align-items: center;
                gap: 0.7rem;
            }

            .confidence-row {
                margin-bottom: 1.35rem;
            }

            .probability-row {
                margin-bottom: 0.6rem;
            }

            .confidence-value {
                min-width: 92px;
                color: #334155;
                font-size: 0.83rem;
                font-weight: 600;
            }

            .probability-name {
                width: 78px;
                flex-shrink: 0;
                color: #334155;
                font-size: 0.82rem;
            }

            .probability-value {
                width: 44px;
                flex-shrink: 0;
                color: #64748B;
                font-size: 0.78rem;
                text-align: right;
            }

            .progress-track {
                flex: 1;
                height: 7px;
                overflow: hidden;
                border-radius: 999px;
                background: #E9EEF5;
            }

            .progress-fill {
                height: 100%;
                border-radius: 999px;
            }

            .breakdown-title {
                margin-bottom: 0.8rem;
                color: #7C8AA0;
                font-size: 0.7rem;
                font-weight: 600;
                letter-spacing: 0.12em;
                text-transform: uppercase;
            }

            .description-box {
                margin-top: 1.15rem;
                padding: 0.85rem 1rem;
                border-radius: 12px;
                color: #334155;
                font-size: 0.86rem;
                line-height: 1.55;
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
                margin: 1.15rem auto 0;
                color: #94A3B8;
                font-size: 0.73rem;
                line-height: 1.6;
                text-align: center;
            }

            @media (max-width: 640px) {
                .block-container {
                    padding: 0.7rem 0.7rem 1.5rem;
                }

                .hero-card {
                    padding: 1.5rem 1rem 1.35rem;
                    border-radius: 18px;
                }

                .hero-title {
                    font-size: 2.15rem;
                }

                .hero-subtitle {
                    font-size: 0.9rem;
                }

                [data-testid="stFileUploader"] section {
                    min-height: 105px;
                    padding: 0.9rem;
                }

                .result-card {
                    padding: 1.15rem;
                }
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
    st.markdown(
        dedent(
            """
            <div class="hero-card">
                <div class="hero-eyebrow">
                    ResNet50 · Transfer Learning
                </div>

                <div class="hero-title">
                    Cataract<span>AI</span>
                </div>

                <div class="hero-subtitle">
                    Unggah foto mata untuk mendeteksi tingkat keparahan
                    katarak secara otomatis: Normal, Immature, atau Mature.
                </div>
            </div>

            <div class="upload-title">
                Unggah foto mata
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_result(
    label: str,
    confidence: float,
    probabilities: np.ndarray,
) -> None:
    metadata = CLASS_META[label]
    confidence_percent = confidence * 100
    probability_rows = []

    for class_name, probability in zip(
        CLASS_NAMES,
        probabilities,
    ):
        class_metadata = CLASS_META[class_name]
        probability_percent = float(probability) * 100
        font_weight = "600" if class_name == label else "500"

        probability_rows.append(
            dedent(
                f"""
                <div class="probability-row">
                    <span
                        class="probability-name"
                        style="font-weight: {font_weight};"
                    >
                        {class_name}
                    </span>

                    <div class="progress-track">
                        <div
                            class="progress-fill"
                            style="
                                width: {probability_percent:.2f}%;
                                background: {class_metadata['bar_color']};
                            "
                        ></div>
                    </div>

                    <span class="probability-value">
                        {probability_percent:.1f}%
                    </span>
                </div>
                """
            ).strip()
        )

    result_html = dedent(
        f"""
        <div class="result-card">
            <div class="result-label">
                Hasil deteksi
            </div>

            <div class="result-class {metadata['css_class']}">
                {label}
            </div>

            <div class="confidence-row">
                <span class="confidence-value">
                    {confidence_percent:.1f}% yakin
                </span>

                <div class="progress-track">
                    <div
                        class="progress-fill"
                        style="
                            width: {confidence_percent:.2f}%;
                            background: {metadata['bar_color']};
                        "
                    ></div>
                </div>
            </div>

            <div class="breakdown-title">
                Distribusi probabilitas
            </div>

            {''.join(probability_rows)}

            <div class="description-box {metadata['css_class']}">
                {metadata['description']}
            </div>
        </div>
        """
    )

    st.markdown(
        result_html,
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    st.markdown(
        dedent(
            """
            <div class="disclaimer">
                Hasil ini hanya digunakan untuk keperluan edukasi dan penelitian.<br>
                Hasil prediksi bukan pengganti diagnosis medis profesional.
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def main() -> None:
    render_page_config()
    render_styles()
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

    render_result(
        label,
        confidence,
        probabilities,
    )

    render_disclaimer()


if __name__ == "__main__":
    main()
