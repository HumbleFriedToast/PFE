import streamlit as st
from PIL import Image
import numpy as np
import Script.LSB as LB
import Script.DCT as DC
import Script.DWT as DW
from Script.DWT2 import preprocess_images_svd, embed_watermark_dwt_svd,extract_watermark_dwt_svd
import time
import cv2
from io import BytesIO
global result 
result = 0

def simulate_loading():
    progress_bar = st.progress(0, text="Starting...")
    for i in range(101):
        time.sleep(0.02)
        progress_bar.progress(i, text=f"Processing: {i}%")
    st.success("Embedding Complete!")
    return  1




# Custom CSS for text colors


# Apply the styles
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Roboto&display=swap" rel="stylesheet">
<style>
body {
    font-family: 'Roboto', sans-serif;
    color: #333333;
}
</style>
""", unsafe_allow_html=True)


import streamlit as st

# Hide Streamlit branding ribbon and other UI elements
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none !important;}
    .st-emotion-cache-13ln4jf {display: none;}  /* sometimes used for deploy ribbon */
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def watermark_embed_treatment(cover,watermark,mode):
    if mode =="DCT":
        pass
    if mode =="DWT":
        pass
    if mode =="LSB":
        pass
    return cover


def watermark_extract_treatment(cover,mode):
    if mode =="DCT":
        pass
    if mode =="DWT":
        pass
    if mode =="LSB":
        pass
    return cover





# Custom styling
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 5rem;
    }

    .stTabs [role="tab"] {
        padding: 0.5rem 1.5rem;
        font-size: 1.1rem;
        border-radius: 0.5rem;
    }

    .stButton>button {
        font-size: 1rem;
        padding: 0.5rem 1.5rem;
        border-radius: 0.5rem;
        border: none;
        margin: 0.5rem;
    }

    .selected-op {
        background-color: #1f77b4 !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Tab Setup
tab_labels = ["DCT", "DWT", "LSB"]
tabs = st.tabs(tab_labels)

for i, tab in enumerate(tabs):
    with tab:
        current_tab = tab_labels[i]
        key_op = f"operation_mode_{current_tab}"

        # Set default operation mode
        if key_op not in st.session_state:
            st.session_state[key_op] = "Embed"

        operation = st.session_state[key_op]

        # Operation buttons
        center_col1, center_col2, center_col3 = st.columns([1, 2, 1])
        with center_col2:
            col_embed, col_extract = st.columns(2)
            with col_embed:
                if st.button("Embed", key=f"embed_btn_{current_tab}"):
                    st.session_state[key_op] = "Embed"
                    operation = "Embed"
            with col_extract:
                if st.button("Extract", key=f"extract_btn_{current_tab}"):
                    st.session_state[key_op] = "Extract"
                    operation = "Extract"

        # Layout for file inputs and settings
        col_main, col_settings = st.columns([3, 1], gap="large")

        with col_main:
            if operation == "Embed":
                col1, col2 = st.columns(2)
                with col1:
                    cover = st.file_uploader("Upload Cover Image", type=["png", "jpg", "jpeg"], key=f"cover_{current_tab}")
                with col2:
                    watermark = st.file_uploader("Upload Watermark Image", type=["png", "jpg", "jpeg"], key=f"wm_{current_tab}")
            else:
                watermarked_image = st.file_uploader("Upload Watermarked Image", type=["png", "jpg", "jpeg"], key=f"extract_{current_tab}")

        with col_settings:
            if current_tab == "DCT":
                if operation == "Embed":
                    alpha = st.slider("DCT Quality", 1, 100, 10)
                    block8 = st.slider("Use 8x8 blocks",8,16,8,8)
                    freq = st.radio("Choose a frequency",("low","mid","high"))
                    robust = st.checkbox("Robust DCT")

                elif operation == "Extract":
                    alphaex = st.slider("DCT Quality", 1, 100, 10)
                    block8ex = st.slider("Use 8x8 blocks",8,16,8,8)
                    freqex = st.radio("Choose a frequency",("low","mid","high"))
                    robustex = st.checkbox("Robust DCT")

            elif current_tab == "DWT":
                if operation == "Embed":
                    dwt_level = st.slider("Decomposition Level", 1, 3, 1)
                    wm_method = st.selectbox("Watermarking Method", ["DWT", "DWT-SVD"], key=f"method_{operation}_{current_tab}")
                    embedding_strength = st.slider("Strength Level",1,100,60)
                elif operation == "Extract":
                    dwt_levelx = st.slider("Decomposition Level", 1, 3, 1)
                    embedding_strengthx = st.slider("Strength Level",1,100,60)
                    robustex = st.checkbox("Robust DWT")
                    wm_method = st.selectbox("Watermarking Method", ["DWT", "DWT-SVD"], key=f"method_{operation}_{current_tab}")
            elif current_tab == "LSB":
                if operation == "Embed":
                    bit = st.number_input("Bit Depth", 1, 2, 2)
                elif operation == "Extract":
                    bit = st.number_input("Bit Depth", 1, 2, 2)
                    st.selectbox("Recovery Mode", ["Basic", "Enhanced"])

        # Action Button (centered)
        st.markdown(" ")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:

            if st.button(f"{operation} Now", key=f"action_btn_{current_tab}"):
                if operation == "Embed":
                    if cover and watermark: # type: ignore
                        if current_tab == "DCT":
                            result = simulate_loading()
                            cover_dct = Image.open(cover)
                            cover_dct = np.array(cover_dct)
                            watermark_dct = Image.open(watermark)
                            watermark_dct = np.array(watermark_dct)
                            watermarked_dct,watermark_binary = DC.embed_watermark(cover_dct,watermark_dct,alpha= alpha,region=freq)
                            show_result = st.image(watermarked_dct, use_container_width=True)
                            image_rgb = cv2.cvtColor(watermarked_dct, cv2.COLOR_BGR2RGB)
                            image_pil = Image.fromarray(image_rgb)
                            buf = BytesIO()
                            image_pil.save(buf, format="jpeg")
                            byte_im = buf.getvalue()

                                # Streamlit download button
                            st.download_button(
                                    label="📥 Download Watermarked Image",
                                    data=byte_im,
                                    file_name="resultatwatermarking.jpg",
                                    mime="image/png"
                                )

                            pass
                        if current_tab == "LSB":
                            result = simulate_loading()
                            cover_lsb = Image.open(cover)
                            cover_lsb = np.array(cover_lsb)
                            watermark_lsb = Image.open(watermark)
                            watermark_lsb = np.array(watermark_lsb)
                            watermarked_lsb = LB.lsb_embed(cover_lsb,watermark_lsb)
                            show_result = st.image(watermarked_lsb, use_container_width=True)
                            image_rgb = cv2.cvtColor(watermarked_lsb, cv2.COLOR_BGR2RGB)
                            image_pil = Image.fromarray(image_rgb)
                            buf = BytesIO()
                            image_pil.save(buf, format="jpeg")
                            byte_im = buf.getvalue()

                                # Streamlit download button
                            st.download_button(
                                    label="📥 Download Watermarked Image",
                                    data=byte_im,
                                    file_name="resultatwatermarking.jpg",
                                    mime="image/png"
                                )

                            pass
                        if current_tab == "DWT":
                            result = simulate_loading()
                            cover_dwt = Image.open(cover).convert("L")
                            watermark_dwt = Image.open(watermark).convert("L")
                            cover_dwt_np = np.array(cover_dwt)
                            watermark_dwt_np = np.array(watermark_dwt)

                            if wm_method == "DWT":
                                watermarked_dwt = DW.embed_watermark(cover_dwt_np, watermark_dwt_np, level=dwt_level, strength=embedding_strength)

                            elif wm_method == "DWT-SVD":
                               
                                pre_cover, pre_wm = preprocess_images_svd(cover_dwt_np, watermark_dwt_np)
                                watermarked_dwt, embedding_info = embed_watermark_dwt_svd(pre_cover, pre_wm, level=dwt_level, strength=embedding_strength,subband="HH")

                                # 🔐 Save embedding_info in session state
                                st.session_state['embedding_info'] = embedding_info

                            st.image(watermarked_dwt, use_container_width=True)
                            image_rgb = cv2.cvtColor(watermarked_dwt, cv2.COLOR_GRAY2RGB)
                            image_pil = Image.fromarray(image_rgb)
                            buf = BytesIO()
                            image_pil.save(buf, format="jpeg")
                            byte_im = buf.getvalue()
                            st.download_button("📥 Download Watermarked Image", data=byte_im, file_name="resultatwatermarking.jpg", mime="image/jpeg")
                    else:
                        st.error("no cover or watermark uploaded")
                if operation =="Extract":
                    if watermarked_image:  # type: ignore
                         if current_tab == "DCT":
                            result = simulate_loading()
                            cover_dct = Image.open(watermarked_image)
                            cover_dct = np.array(cover_dct)
                            unwatermarked_dct,x = DC.extract_watermark(cover_dct,cover_dct.shape,alpha=alphaex,region =freqex)
                            show_result = st.image(x, use_container_width=True)

                            pass
                         if current_tab == "LSB":
                            result = simulate_loading()
                            cover_lsb = Image.open(watermarked_image)
                            cover_lsb = np.array(cover_lsb)
                            unwatermarked_lsb = LB.lsb_extract(cover_lsb,cover_lsb.shape)
                            show_result = st.image(unwatermarked_lsb, use_container_width=True) 


                            pass
                         if current_tab == "DWT":
                            result = simulate_loading()
                            watermarked_img = Image.open(watermarked_image).convert("L")
                            cover_dwt_np = np.array(watermarked_img)

                            if wm_method == "DWT":
                                unwatermarked_dwt = DW.extract_watermark(cover_dwt_np, cover_dwt_np.shape, level=dwt_levelx,  )
                                st.image(unwatermarked_dwt, use_container_width=True)

                            elif wm_method == "DWT-SVD":
                                embedding_info = st.session_state.get('embedding_info', None)
                                if "embedding_info" not in globals():
                                    st.error("DWT-SVD extraction requires the embedding info. Embed first in the same session.")
                                else:
                                    extracted_wm = extract_watermark_dwt_svd(cover_dwt_np, embedding_info, level=dwt_levelx, strength=embedding_strengthx,subband="HH")
                                    st.image(extracted_wm, use_container_width=True)



                            pass
                            

                    
