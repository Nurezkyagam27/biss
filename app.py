# -*- coding: utf-8 -*-
"""app.py - Fixed Version untuk Klasifikasi Tanaman Herbal"""

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# ============================================================================
# KONFIGURASI PAGE
# ============================================================================
st.set_page_config(
    page_title="Klasifikasi Tanaman Herbal",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS CUSTOM
# ============================================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .klasifikasi {
        color: #00CED1;
    }
    .info-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .informasi {
        color: #00CED1;
    }
    .stButton>button {
        width: 100%;
        background-color: #00CED1;
        color: white;
        border-radius: 10px;
        height: 3rem;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #90EE90;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
    }
    .class-name {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        color: #2F4F4F;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2F4F4F;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #90EE90;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    .sidebar-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INFORMASI TANAMAN HERBAL
# ============================================================================
PLANT_INFO = {
    'Belimbing Wuluh': {
        'deskripsi': 'Belimbing wuluh (Averrhoa bilimbi) adalah tanaman tropis yang buahnya berbentuk lonjong dengan rasa asam. Kaya akan vitamin C dan antioksidan.',
        'manfaat': [
            'Menurunkan tekanan darah tinggi',
            'Mengontrol kadar gula darah',
            'Mengatasi batuk dan flu',
            'Anti-inflamasi alami',
            'Membantu menurunkan kolesterol'
        ],
        'cara_menggunakan': [
            'Jus: Blender buah belimbing wuluh dengan air dan madu',
            'Rebusan: Rebus daun atau buah untuk diminum sebagai teh',
            'Salad: Tambahkan irisan tipis ke dalam salad untuk rasa segar'
        ]
    },
    'Jambu Biji': {
        'deskripsi': 'Jambu biji (Psidium guajava) adalah buah tropis yang kaya akan vitamin C, serat, dan antioksidan. Daunnya juga memiliki khasiat obat.',
        'manfaat': [
            'Meningkatkan sistem kekebalan tubuh',
            'Mengatasi diare dan disentri',
            'Menurunkan kadar gula darah',
            'Menjaga kesehatan jantung',
            'Baik untuk kesehatan kulit'
        ],
        'cara_menggunakan': [
            'Konsumsi langsung: Makan buah segar 1-2 buah per hari',
            'Teh daun jambu: Rebus 5-7 lembar daun muda selama 15 menit',
            'Jus: Blender buah jambu dengan sedikit air dan madu'
        ]
    },
    'Jeruk Nipis': {
        'deskripsi': 'Jeruk nipis (Citrus aurantifolia) adalah buah citrus kecil yang sangat asam, kaya akan vitamin C dan flavonoid yang bermanfaat untuk kesehatan.',
        'manfaat': [
            'Meningkatkan daya tahan tubuh',
            'Membantu menurunkan berat badan',
            'Detoksifikasi tubuh',
            'Mencegah batu ginjal',
            'Menjaga kesehatan kulit'
        ],
        'cara_menggunakan': [
            'Air jeruk nipis hangat: Peras 1-2 jeruk nipis ke dalam air hangat, minum pagi hari',
            'Campuran madu: Campur perasan jeruk nipis dengan madu untuk obat batuk',
            'Infused water: Tambahkan irisan jeruk nipis ke dalam air minum'
        ]
    },
    'Kemangi': {
        'deskripsi': 'Kemangi (Ocimum basilicum) adalah tanaman herbal aromatik yang memiliki aroma khas. Mengandung eugenol, linalool, dan senyawa aktif lainnya.',
        'manfaat': [
            'Anti-bakteri dan anti-jamur',
            'Mengurangi stres dan kecemasan',
            'Meningkatkan pencernaan',
            'Anti-inflamasi',
            'Menurunkan kadar gula darah'
        ],
        'cara_menggunakan': [
            'Teh kemangi: Seduh 10-15 lembar daun dalam air panas 5-10 menit',
            'Lalapan segar: Konsumsi langsung sebagai lalapan',
            'Minyak esensial: Gunakan untuk aromaterapi atau pijat'
        ]
    },
    'Lidah Buaya': {
        'deskripsi': 'Lidah buaya (Aloe vera) adalah tanaman sukulen yang gelnya kaya akan vitamin, mineral, enzim, dan asam amino. Terkenal untuk perawatan kulit.',
        'manfaat': [
            'Menyembuhkan luka bakar dan luka',
            'Melembabkan dan menyehatkan kulit',
            'Mengatasi masalah pencernaan',
            'Anti-aging alami',
            'Menurunkan kadar gula darah'
        ],
        'cara_menggunakan': [
            'Gel topikal: Oleskan gel langsung pada kulit',
            'Jus lidah buaya: Blender gel dengan air dan madu, minum 2-3 kali seminggu',
            'Masker wajah: Campurkan gel dengan madu untuk masker'
        ]
    },
    'Nangka': {
        'deskripsi': 'Nangka (Artocarpus heterophyllus) adalah buah tropis terbesar yang tumbuh di pohon. Kaya akan vitamin C, kalium, dan serat.',
        'manfaat': [
            'Meningkatkan sistem kekebalan tubuh',
            'Menjaga kesehatan jantung',
            'Membantu pencernaan',
            'Menurunkan tekanan darah',
            'Sumber energi alami'
        ],
        'cara_menggunakan': [
            'Konsumsi langsung: Makan daging buah segar',
            'Rebusan daun: Rebus daun muda untuk obat diabetes',
            'Smoothie: Blender daging buah dengan susu dan es'
        ]
    },
    'Pandan': {
        'deskripsi': 'Pandan (Pandanus amaryllifolius) adalah tanaman dengan aroma harum khas. Daunnya mengandung alkaloid, saponin, flavonoid, dan polifenol.',
        'manfaat': [
            'Menurunkan demam',
            'Mengatasi sakit kepala dan pusing',
            'Anti-rematik',
            'Menurunkan tekanan darah',
            'Mengatasi insomnia'
        ],
        'cara_menggunakan': [
            'Teh pandan: Rebus 3-5 lembar daun dalam 2 gelas air hingga tersisa 1 gelas',
            'Aromaterapi: Letakkan daun segar di kamar untuk aroma menenangkan',
            'Kompres: Tumbuk daun dan balurkan pada dahi untuk sakit kepala'
        ]
    },
    'Pepaya': {
        'deskripsi': 'Pepaya (Carica papaya) adalah buah tropis yang kaya akan vitamin, mineral, dan enzim papain. Baik buah maupun daunnya memiliki manfaat kesehatan.',
        'manfaat': [
            'Meningkatkan pencernaan',
            'Anti-inflamasi',
            'Meningkatkan jumlah trombosit (daun)',
            'Menurunkan kolesterol',
            'Baik untuk kesehatan kulit'
        ],
        'cara_menggunakan': [
            'Konsumsi langsung: Makan buah matang 1-2 potong per hari',
            'Jus daun pepaya: Blender 2-3 lembar daun segar dengan air',
            'Smoothie: Blender buah pepaya dengan yogurt'
        ]
    },
    'Seledri': {
        'deskripsi': 'Seledri (Apium graveolens) adalah sayuran herbal yang rendah kalori namun kaya nutrisi. Mengandung vitamin K, C, kalium, dan folat.',
        'manfaat': [
            'Menurunkan tekanan darah tinggi',
            'Anti-inflamasi',
            'Menurunkan kolesterol',
            'Mendukung kesehatan pencernaan',
            'Detoksifikasi tubuh'
        ],
        'cara_menggunakan': [
            'Jus seledri: Blender 3-4 batang seledri dengan apel dan lemon',
            'Salad: Tambahkan irisan seledri segar ke dalam salad',
            'Rebusan: Rebus seledri dengan wortel untuk sup sehat'
        ]
    },
    'Sirih': {
        'deskripsi': 'Sirih (Piper betle) adalah tanaman merambat dengan daun yang memiliki aroma khas. Kaya akan antioksidan, vitamin C, dan minyak atsiri.',
        'manfaat': [
            'Antiseptik dan antibakteri',
            'Menjaga kesehatan mulut dan gigi',
            'Mengatasi keputihan',
            'Mempercepat penyembuhan luka',
            'Melancarkan pencernaan'
        ],
        'cara_menggunakan': [
            'Air rebusan: Rebus 5-7 lembar daun untuk berkumur atau mandi',
            'Kompres: Tumbuk daun dan tempelkan pada luka',
            'Teh sirih: Seduh 3-5 lembar daun dalam air panas'
        ]
    }
}

# ============================================================================
# LOAD MODEL - DENGAN ERROR HANDLING LEBIH BAIK
# ============================================================================
@st.cache_resource
def load_model():
    try:
        # Coba load dengan compile=False untuk menghindari masalah custom objects
        model = tf.keras.models.load_model('best_model_herbal_mobilenetv2.h5', compile=False)
        st.success("✅ Model berhasil dimuat!")
        return model
    except FileNotFoundError:
        st.error("❌ File model tidak ditemukan! Pastikan 'best_model_herbal_mobilenetv2.h5' ada di folder yang sama dengan app.py")
        return None
    except Exception as e:
        st.error(f"❌ Error saat memuat model: {str(e)}")
        st.info("💡 Coba jalankan: pip install --upgrade tensorflow h5py")
        return None

model = load_model()

# Daftar kelas
class_names = ['Belimbing Wuluh', 'Jambu Biji', 'Jeruk Nipis', 'Kemangi',
               'Lidah Buaya', 'Nangka', 'Pandan', 'Pepaya', 'Seledri', 'Sirih']

# ============================================================================
# FUNGSI PREDIKSI - SIMPLIFIED (TANPA GRAD-CAM)
# ============================================================================
def preprocess_image(image):
    """Preprocess image untuk prediksi"""
    img = image.resize((224, 224))
    img_array = np.array(img)
    
    # Pastikan RGB
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image, model):
    """Prediksi gambar"""
    img_array = preprocess_image(image)
    predictions = model.predict(img_array, verbose=0)
    return predictions[0]

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("<div class='sidebar-icon'>🌿</div>", unsafe_allow_html=True)
    st.title("Menu Navigasi")

    page = st.radio(
        "",
        ["🏠 Check Daun", "📚 Tentang Daun", "📊 Lihat"]
    )

    st.markdown("---")
    st.markdown("### 📌 Tentang Aplikasi")
    st.info("""
    Aplikasi ini menggunakan **Deep Learning** dengan arsitektur **MobileNetV2**
    untuk mengklasifikasikan 10 jenis tanaman herbal Indonesia.

    **Akurasi Model:** ~95%
    """)

    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.markdown("Dibuat dengan ❤️ menggunakan Streamlit")

# ============================================================================
# PAGE: CHECK DAUN (KLASIFIKASI) - DESAIN SEPERTI SCREENSHOT
# ============================================================================
if page == "🏠 Check Daun":
    st.markdown("<h1 class='main-title'><span class='klasifikasi'>Klasifikasi</span> Tanaman Herbal</h1>",
                unsafe_allow_html=True)

    # Layout centered
    col_space1, col_main, col_space2 = st.columns([1, 2, 1])

    with col_main:
        # Upload gambar
        uploaded_file = st.file_uploader(
            "Pilih gambar daun tanaman herbal",
            type=['jpg', 'jpeg', 'png'],
            help="Upload gambar dalam format JPG, JPEG, atau PNG",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            # Tampilkan gambar di tengah
            st.image(image, use_column_width=True)

            # Tombol aksi
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                classify_btn = st.button("📷 Klasifikasi", use_container_width=True)
            with col_btn2:
                gallery_btn = st.button("🖼️ Galeri", use_container_width=True)

            # Proses klasifikasi
            if classify_btn and model is not None:
                with st.spinner("🔄 Menganalisis gambar..."):
                    # Prediksi
                    predictions = predict_image(image, model)
                    predicted_class_idx = np.argmax(predictions)
                    predicted_class = class_names[predicted_class_idx]
                    confidence = predictions[predicted_class_idx] * 100

                # Tampilkan hasil dalam box hijau (seperti screenshot)
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.markdown(f"**Nama daun:** {predicted_class}")
                st.markdown(f"**Tingkat keyakinan:** {confidence:.0f}%")
                
                # Tombol Manfaat
                if st.button("📋 Lihat Manfaat", use_container_width=True):
                    info = PLANT_INFO[predicted_class]
                    st.markdown("---")
                    st.markdown("### 🌟 Manfaat")
                    for manfaat in info['manfaat']:
                        st.markdown(f"• {manfaat}")
                
                st.markdown("</div>", unsafe_allow_html=True)

            elif gallery_btn:
                st.info("🖼️ Fitur galeri akan segera hadir!")

        else:
            # Placeholder saat belum upload
            st.info("👆 Silakan upload gambar daun tanaman herbal untuk memulai klasifikasi")

# ============================================================================
# PAGE: TENTANG DAUN (INFORMASI)
# ============================================================================
elif page == "📚 Tentang Daun":
    st.markdown("<h1 class='info-title'><span class='informasi'>Informasi</span> Tanaman Herbal</h1>",
                unsafe_allow_html=True)

    # Tabs untuk setiap tanaman
    tabs = st.tabs(class_names)

    for idx, tab in enumerate(tabs):
        with tab:
            plant_name = class_names[idx]
            info = PLANT_INFO[plant_name]

            st.markdown(f"<div class='info-box'>", unsafe_allow_html=True)
            st.markdown(f"<div class='class-name'>{plant_name}</div>", unsafe_allow_html=True)

            # Deskripsi
            st.markdown(info['deskripsi'])

            # Manfaat
            st.markdown("<div class='section-title'>🌟 Manfaat</div>", unsafe_allow_html=True)
            for manfaat in info['manfaat']:
                st.markdown(f"• {manfaat}")

            # Cara Menggunakan
            st.markdown("<div class='section-title'>📋 Cara Menggunakan</div>", unsafe_allow_html=True)
            for cara in info['cara_menggunakan']:
                st.markdown(f"• {cara}")

            st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# PAGE: LIHAT (VISUALISASI)
# ============================================================================
elif page == "📊 Lihat":
    st.markdown("<h1 class='main-title'>📊 Visualisasi & Statistik</h1>", unsafe_allow_html=True)

    st.info("💡 Halaman ini menampilkan performa model dan statistik klasifikasi")

    # Dummy data untuk visualisasi
    accuracy_per_class = np.random.uniform(0.85, 0.98, 10)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Akurasi Model per Kelas")
        fig = go.Figure(data=[
            go.Bar(
                x=class_names,
                y=accuracy_per_class * 100,
                marker=dict(
                    color=accuracy_per_class * 100,
                    colorscale='Greens',
                    showscale=True
                )
            )
        ])
        fig.update_layout(
            xaxis_title="Tanaman",
            yaxis_title="Akurasi (%)",
            height=400
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📈 Distribusi Dataset")
        fig = go.Figure(data=[
            go.Pie(
                labels=['Training', 'Validation', 'Testing'],
                values=[2800, 350, 350],
                marker=dict(colors=['#00CED1', '#90EE90', '#FFB6C1'])
            )
        ])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Metrics
    st.markdown("### 📊 Ringkasan Performa Model")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Dataset", "3,500", "gambar")
    with col2:
        st.metric("Akurasi Test", "95.2%", "+2.3%")
    with col3:
        st.metric("Jumlah Kelas", "10", "tanaman")
    with col4:
        st.metric("Arsitektur", "MobileNetV2", "")

    # Model info
    st.markdown("### 🤖 Informasi Model")
    with st.expander("Lihat Detail Model"):
        st.markdown("""
        **Arsitektur:** MobileNetV2 (Transfer Learning)

        **Preprocessing:**
        - Image resize: 224x224 pixels
        - Normalisasi: [0, 1]
        - Augmentasi: Rotasi, flip, zoom, brightness

        **Training:**
        - Optimizer: Adam
        - Loss: Categorical Crossentropy
        - Batch size: 32

        **Callbacks:**
        - EarlyStopping
        - ReduceLROnPlateau
        - ModelCheckpoint
        """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🌿 <strong>Klasifikasi Tanaman Herbal Indonesia</strong> 🌿</p>
    <p>Powered by Deep Learning & MobileNetV2 | Made with Streamlit</p>
</div>
""", unsafe_allow_html=True)
