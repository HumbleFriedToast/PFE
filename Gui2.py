import streamlit as st
import Script.LSB as LB
import Script.DCT as DC
import Script.DWT as DW
import time

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
        color: #EAD8BF;
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
                    st.slider("DCT Quality", 1, 100, 50)
                    st.checkbox("Use 8x8 blocks")
                elif operation == "Extract":
                    st.checkbox("Auto-detect DCT blocks")

            elif current_tab == "DWT":
                if operation == "Embed":
                    st.selectbox("Wavelet Type", ["haar", "db1", "sym2"], key=f"wavelet_type_{current_tab}")
                    st.slider("Decomposition Level", 1, 5, 2)
                elif operation == "Extract":
                    st.checkbox("Preserve low-frequency band")

            elif current_tab == "LSB":
                if operation == "Embed":
                    st.number_input("Bit Depth", 1, 8, 2)
                    st.checkbox("Grayscale only")
                elif operation == "Extract":
                    st.selectbox("Recovery Mode", ["Basic", "Enhanced"])

        # Action Button (centered)
        st.markdown(" ")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:

            if st.button(f"{operation} Now", key=f"action_btn_{current_tab}"):
                if operation == "Embed":
                    if cover and watermark: # type: ignore
                        result = simulate_loading()
                        watermark_embed_treatment(cover,watermark,current_tab) # type: ignore
                    else:
                        st.error("no cover or watermark uploaded")
                if operation =="Extract":
                    if watermarked_image:  # type: ignore
                        watermark_extract_treatment(cover,current_tab) # type: ignore
                    
