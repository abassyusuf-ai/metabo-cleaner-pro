import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="Metabo-Cleaner Pro", layout="wide")

st.title("🧪 Metabo-Cleaner Pro")
st.markdown("### Turn messy Mass Spec data into 'Ready-to-Analyze' Feature Tables")

uploaded_file = st.file_uploader("Upload your .csv file (Long Format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ **File Uploaded!** Here is a preview of your raw data:")
    st.dataframe(df.head())

    # --- THE PRODUCT LOGIC ---
    st.sidebar.header("Settings")
    mz_col = st.sidebar.selectbox("Select m/z column", df.columns, index=0)
    rt_col = st.sidebar.selectbox("Select RT column", df.columns, index=1)
    sample_col = st.sidebar.selectbox("Select Sample ID column", df.columns, index=4)
    intensity_col = st.sidebar.selectbox("Select Intensity column", df.columns, index=2)

    if st.button("Process & Pivot Data"):
        # 1. Create Unique ID
        df['Feature_ID'] = df[mz_col].astype(str) + "_" + df[rt_col].astype(str)
        
        # 2. Pivot
        pivot_df = df.pivot_table(
            index='Feature_ID', 
            columns=sample_col, 
            values=intensity_col, 
            aggfunc='mean'
        ).fillna(0)

        st.write("✨ **Cleaned Feature Table:**")
        st.dataframe(pivot_df.head())

        # 3. Download Button
        csv = pivot_df.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download Cleaned CSV",
            data=csv,
            file_name="metabo_cleaned_data.csv",
            mime="text/csv",
        )
        st.success("Success! You saved hours of manual Excel work.")

st.info("Goal: Simplify Metabolomics. Built by a Scientist, for Scientists.")
