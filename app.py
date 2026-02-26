import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="Metabo-Cleaner Pro", layout="wide", page_icon="🧪")

# --- SIDEBAR: MONETIZATION & INFO ---
st.sidebar.title("About")
st.sidebar.info("Developed by a Metabolomics Postdoc to simplify data preprocessing.")
st.sidebar.markdown("---")
st.sidebar.write("🙏 **Support this project**")
# Replace 'yourname' with your actual BuyMeACoffee username later
st.sidebar.markdown("[☕ Buy me a coffee](https://www.buymeacoffee.com/abassyusuf)") 

st.title("🧪 Metabo-Cleaner Pro")
st.markdown("### Pivot, Clean, and Normalize your Mass Spec Data")

uploaded_file = st.file_uploader("Step 1: Upload your Long-Format .csv file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File Uploaded!")
    
    with st.expander("Preview Raw Data"):
        st.dataframe(df.head())

    # --- SETTINGS ---
    st.markdown("### Step 2: Configure Columns")
    col1, col2, col3, col4 = st.columns(4)
    with col1: mz_col = st.selectbox("m/z Column", df.columns, index=0)
    with col2: rt_col = st.selectbox("RT Column", df.columns, index=1)
    with col3: sample_col = st.selectbox("Sample Name Column", df.columns, index=4 if len(df.columns)>4 else 0)
    with col4: intensity_col = st.selectbox("Intensity Column", df.columns, index=2)

    # --- PROCESSING ---
    if st.button("🚀 Process & Create Feature Table"):
        # 1. Create Unique Feature ID
        df['Feature_ID'] = df[mz_col].astype(str) + "_" + df[rt_col].astype(str)
        
        # 2. Pivot
        pivot_df = df.pivot_table(index='Feature_ID', columns=sample_col, values=intensity_col, aggfunc='mean').fillna(0)
        
        st.markdown("### Step 3: View & Normalize")
        st.write("Raw Feature Table (Pivoted):")
        st.dataframe(pivot_df.head())

        # 3. Normalization Feature
        tic_norm = pivot_df.div(pivot_df.sum(axis=0), axis=1) * 1000000 # Normalized to 10^6
        
        st.write("TIC Normalized Table (Counts per Million):")
        st.dataframe(tic_norm.head())

        # 4. Download Buttons
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📥 Download Pivoted CSV", pivot_df.to_csv().encode('utf-8'), "pivoted_data.csv", "text/csv")
        with c2:
            st.download_button("📥 Download TIC Normalized CSV", tic_norm.to_csv().encode('utf-8'), "normalized_data.csv", "text/csv")
        
        st.balloons()
