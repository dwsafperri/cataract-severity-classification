import io
import time
from pathlib import Path

import keras
import numpy as np
import streamlit as st
from PIL import Image
from keras.applications.resnet50 import preprocess_input


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CataractAI",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');

    /* Reset & base */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header {
        visibility: hidden;
    }

    .block-container {
        padding: 2.5rem 1.5rem 4rem;
        max-width: 720px;
    }

    /* ── Hero ── */
    .hero {
        text-align: center;
        padding: 3rem 0 2rem;
    }

    .hero-eyebrow {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #6B7D99;
        margin-bottom: 0.75rem;
    }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1.1;
        color: #0F1923;
        margin: 0 0 1rem;
    }

    .hero-title span {
        color: #2563EB;
    }

    .hero-sub {
        font-size: 0.97rem;
        color: #556070;
        line-height: 1.65;
        max-width: 480px;
        margin: 0 auto;
    }

    /* ── Upload zone ── */
    .upload-label {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #6B7D99;
        margin-bottom: 0.4rem;
        display: block;
    }

    [data-testid="stFileUploader"] {
        border: 1.5px dashed #CBD5E1;
        border-radius: 12px;
        padding: 0.5rem;
        background: #F8FAFC;
        transition: border-color 0.2s;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #2563EB;
    }

    /* ── Result card ── */
    .result-card {
        border-radius: 16px;
        padding: 2rem 2rem 1.75rem;
        margin-top: 1.5rem;
        border: 1px solid #E2E8F0;
        background: #FFFFFF;
        box-shadow: 0 4px 24px rgba(15, 25, 35, 0.06);
    }

    .result-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #6B7D99;
        margin-bottom: 0.4rem;
    }

    .result-class {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        line-height: 1.15;
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

    .confidence-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-top: 0.75rem;
        margin-bottom: 1.5rem;
    }

    .conf-value {
        font-size: 0.88rem;
        font-weight: 500;
        color: #374151;
    }

    .conf-bar-bg {
        flex: 1;
        height: 6px;
        border-radius: 999px;
        background: #E2E8F0;
        overflow: hidden;
    }

    .conf-bar-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.8s ease;
    }

    /* ── Per-class breakdown ── */
    .breakdown-title {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #94A3B8;
        margin-bottom: 0.85rem;
    }

    .bar-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.55rem;
    }

    .bar-class-name {
        width: 80px;
        font-size: 0.82rem;
        font-weight: 500;
        color: #374151;
        flex-shrink: 0;
    }

    .bar-bg {
        flex: 1;
        height: 8px;
        border-radius: 999px;
        background: #F1F5F9;
        overflow: hidden;
    }

    .bar-fill {
        height: 100%;
        border-radius: 999px;
    }

    .bar-pct {
        width: 40px;
        text-align: right;
        font-size: 0.78rem;
        color: #6B7D99;
        flex-shrink: 0;
    }

    /* ── Description pill ── */
    .desc-pill {
        display: inline-block;
        padding: 0.65rem 1rem;
        border-radius: 10px;
        font-size: 0.87rem;
        line-height: 1.55;
        color: #374151;
        margin-top: 1.25rem;
        width: 100%;
        box-sizing: border-box;
    }

    .desc-pill.normal {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
    }

    .desc-pill.immature {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
    }

    .desc-pill.mature {
        background: #FEF2F2;
        border: 1px solid #FECACA;
    }

    /* ── Disclaimer ── */
    .disclaimer {
        font-size: 0.75rem;
        color: #94A3B8;
        text-align: center;
        margin-top: 2rem;
        line-height: 1.6;
    }

    /* ── Divider ── */
    .thin-divider {
        border: none;
        border-top: 1px solid #E2E8F0;
        margin: 2rem 0 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Immature", "Mature", "Normal"]

CLASS_META = {
    "Normal": {
        "color": "#16A34A",
        "bar_color": "#22C55E",
        "css_class": "normal",
        "description": (
            "Lensa mata tampak jernih tanpa tanda-tanda kekeruhan. "
            "Tidak terdeteksi katarak pada gambar ini."
        ),
    },
    "Immature": {
        "color": "#D97706",
        "bar_color": "#F59E0B",
        "css_class": "immature",
        "description": (
            "Terdeteksi kekeruhan sebagian pada lensa mata. "
            "Katarak masih dalam tahap awal — penanganan dini disarankan."
        ),
    },
    "Mature": {
        "color": "#DC2626",
        "bar_color": "#EF4444",
        "css_class": "mature",
        "description": (
            "Lensa mata menunjukkan kekeruhan yang signifikan. "
            "Katarak sudah berkembang — konsultasi dokter mata segera dianjurkan."
        ),
    },
}


# ── Model path ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.keras"


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_cataract_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan di lokasi: {MODEL_PATH}"
        )

    model = keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    return model


# ── Image preprocessing ───────────────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(image, dtype=np.float32)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_image(model, image: Image.Image):
    image_tensor = preprocess_image(image)

    predictions = model.predict(
        image_tensor,
        verbose=0,
    )[0]

    predicted_index = int(np.argmax(predictions))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    return predicted_label, confidence, predictions


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">
            ResNet50 · Transfer Learning
        </div>

        <h1 class="hero-title">
            Cataract<span>AI</span>
        </h1>

        <p class="hero-sub">
            Unggah foto mata untuk mendeteksi tingkat keparahan katarak
            secara otomatis — Normal, Immature, atau Mature.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Load model ────────────────────────────────────────────────────────────────
try:
    with st.spinner("Memuat model..."):
        model = load_cataract_model()

except FileNotFoundError as error:
    st.error(
        f"Model tidak ditemukan. {error}",
        icon="⚠️",
    )
    st.stop()

except Exception as error:
    st.error(
        "Model ditemukan, tetapi gagal dimuat. "
        "Kemungkinan terdapat perbedaan versi TensorFlow atau Keras "
        "antara proses training dan deployment.",
        icon="⚠️",
    )

    with st.expander("Lihat detail error"):
        st.code(str(error))

    st.stop()


st.markdown(
    '<hr class="thin-divider">',
    unsafe_allow_html=True,
)


# ── Upload image ──────────────────────────────────────────────────────────────
st.markdown(
    '<span class="upload-label">Foto Mata</span>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    label="Foto Mata",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    help="Format gambar yang didukung: JPG, JPEG, dan PNG.",
)


# ── Inference ─────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        image = Image.open(
            io.BytesIO(uploaded_file.read())
        )

    except Exception:
        st.error(
            "File gambar tidak dapat dibaca. "
            "Silakan unggah gambar JPG, JPEG, atau PNG yang valid."
        )
        st.stop()

    image_column, result_column = st.columns(
        [1, 1],
        gap="large",
    )

    with image_column:
        st.image(
            image,
            use_container_width=True,
        )

    with result_column:
        try:
            with st.spinner("Menganalisis gambar..."):
                time.sleep(0.4)

                label, confidence, all_probabilities = predict_image(
                    model,
                    image,
                )

        except Exception as error:
            st.error(
                "Terjadi kesalahan saat melakukan prediksi."
            )

            with st.expander("Lihat detail error"):
                st.code(str(error))

            st.stop()

        metadata = CLASS_META[label]
        css_class = metadata["css_class"]
        bar_color = metadata["bar_color"]
        confidence_percent = confidence * 100

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">
                    Hasil Deteksi
                </div>

                <div class="result-class {css_class}">
                    {label}
                </div>

                <div class="confidence-row">
                    <span class="conf-value">
                        {confidence_percent:.1f}% confidence
                    </span>

                    <div class="conf-bar-bg">
                        <div
                            class="conf-bar-fill"
                            style="
                                width: {confidence_percent}%;
                                background: {bar_color};
                            "
                        >
                        </div>
                    </div>
                </div>

                <div class="breakdown-title">
                    Distribusi Probabilitas
                </div>
            """,
            unsafe_allow_html=True,
        )

        for class_name, probability in zip(
            CLASS_NAMES,
            all_probabilities,
        ):
            class_metadata = CLASS_META[class_name]
            probability_percent = float(probability) * 100

            font_style = (
                "font-weight: 600;"
                if class_name == label
                else ""
            )

            st.markdown(
                f"""
                <div class="bar-row">
                    <span
                        class="bar-class-name"
                        style="{font_style}"
                    >
                        {class_name}
                    </span>

                    <div class="bar-bg">
                        <div
                            class="bar-fill"
                            style="
                                width: {probability_percent}%;
                                background: {class_metadata['bar_color']};
                            "
                        >
                        </div>
                    </div>

                    <span class="bar-pct">
                        {probability_percent:.1f}%
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
                <div class="desc-pill {css_class}">
                    {metadata['description']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <p class="disclaimer">
        Hasil ini hanya untuk keperluan edukasi dan penelitian.<br>
        Bukan pengganti diagnosis medis profesional.
    </p>
    """,
    unsafe_allow_html=True,
)
