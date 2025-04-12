pip install Ecg
import streamlit as st
from Ecg import ECG

# Initialize ecg object
ecg = ECG()

# Page title and description
st.title("ECG Analysis Tool")
st.write("Upload your ECG image and analyze the signals")

# Upload file
uploaded_file = st.file_uploader("Upload ECG Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.subheader("Uploaded Image")
    ecg_user_image_read = ecg.getImage(uploaded_file)
    st.image(ecg_user_image_read, caption="Uploaded ECG Image", use_column_width=True)

    # Gray scale image
    st.subheader("Gray Scale Image")
    ecg_user_gray_image_read = ecg.GrayImgae(ecg_user_image_read)
    st.image(ecg_user_gray_image_read, caption="Gray Scale ECG Image", use_column_width=True)

    # Dividing leads
    st.subheader("Dividing Leads")
    dividing_leads = ecg.DividingLeads(ecg_user_image_read)
    st.image('Leads_1-12_figure.png', caption="Dividing Leads (1-12)", use_column_width=True)
    st.image('Long_Lead_13_figure.png', caption="Long Lead (13)", use_column_width=True)

    # Preprocessed leads
    st.subheader("Preprocessed Leads")
    ecg_preprocessed_leads = ecg.PreprocessingLeads(dividing_leads)
    st.image('Preprossed_Leads_1-12_figure.png', caption="Preprocessed Leads (1-12)", use_column_width=True)
    st.image('Preprossed_Leads_13_figure.png', caption="Preprocessed Long Lead (13)", use_column_width=True)

    # Signal extraction
    st.subheader("Signal Extraction (1-12)")
    ec_signal_extraction = ecg.SignalExtraction_Scaling(dividing_leads)
    st.image('Contour_Leads_1-12_figure.png', caption="Contour Leads (1-12)", use_column_width=True)

    # Convert to 1D signal
    st.subheader("Convert to 1D Signal")
    ecg_1dsignal = ecg.CombineConvert1Dsignal()
    st.write("1D Signal:", ecg_1dsignal)

    # Dimensionality reduction
    st.subheader("Dimensionality Reduction")
    ecg_final = ecg.DimensionalReduciton(ecg_1dsignal)
    st.write("Reduced Dimensionality Signal:", ecg_final)

    # Pretrained ML model prediction
    st.subheader("Prediction using Pretrained Model")
    ecg_model = ecg.ModelLoad_predict(ecg_final)
    st.write("Predicted Output:", ecg_model)


