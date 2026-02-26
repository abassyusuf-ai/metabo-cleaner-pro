import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config(page_title="Metabo-Cleaner Pro", layout="wide", page_icon="🧪")

# --- 1. SIDEBAR & MONETIZATION ---
st.sidebar.title("Metabo-Cleaner Pro")
st.sidebar.info("A specialized tool for Pre-processing and Cleaning Untargeted Metabolomics Data.")
st.sidebar.markdown("---")
st.sidebar.subheader("🙏 Support Development")
st.sidebar.write("If this tool saved you hours of Excel work, consider supporting my research.")
# You can update this link with your actual Buy Me A Coffee URL
st.sidebar.markdown("[☕ Buy me a coffee](https://www.buymeacoffee.com/abassyusuf)") 

# --- 2. HEADER ---
st.title("🧪 Metabo-Cleaner Pro")
st.markdown("### Step 1: Upload your Long-Format .csv file")
uploaded_file = st.file_uploader("Drop your file here", type=["csv"])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.success("✅ File Uploaded successfully!")
    
    with st.expander("Preview Raw Data"):
        st.dataframe(df.head())

    # --- 3. CONFIGURATION ---
    st.markdown("### Step 2: Configure Data Columns")
    col1, col2, col3, col4 = st.columns(4)
    with col1: mz_col = st.selectbox("m/z Column", df.columns, index=0)
    with col2: rt_col = st.selectbox("RT Column", df.columns, index=1)
    with col3: sample_col = st.selectbox("Sample Name Column", df.columns, index=4 if len(df.columns)>4 else 0)
    with col4: intensity_col = st.selectbox("Intensity Column", df.columns, index=2)

    # --- 4. ADVANCED SETTINGS ---
    st.markdown("### Step 3: Pro Cleaning & Alignment Settings")
    c_a, c_b, c_c, c_d = st.columns(4)
    with c_a:
        mz_digits = st.slider("m/z Rounding", 1, 5, 3, help="Groups features with similar m/z.")
    with c_b:
        rt_digits = st.slider("RT Rounding", 0, 2, 1, help="Groups features with similar Retention Time.")
    with c_c:
        min_presence = st.slider("Min Presence (%)", 0, 100, 20, help="Removes rows that are 0 in too many samples.")
    with c_d:
        impute_choice = st.selectbox("Gap Filling (0s)", ["Keep as 0", "Half of Minimum Value"])

    # --- 5. THE ENGINE ---
    if st.button("🚀 Process, Clean & Normalize"):
        try:
            # A. Clean data types
            df[mz_col] = pd.to_numeric(df[mz_col], errors='coerce')
            df[rt_col] = pd.to_numeric(df[rt_col], errors='coerce')
            df[intensity_col] = pd.to_numeric(df[intensity_col], errors='coerce')
            df = df.dropna(subset=[mz_col, rt_col, intensity_col])

            # B. Alignment (Smart Feature ID)
            df['Feature_ID'] = df[mz_col].round(mz_digits).astype(str) + "_" + df[rt_col].round(rt_digits).astype(str)
            pivot_df = df.pivot_table(index='Feature_ID', columns=sample_col, values=intensity_col, aggfunc='mean').fillna(0)
            raw_count = len(pivot_df)

            # C. Filtering (80% Rule / Min Presence)
            threshold = (min_presence / 100) * len(pivot_df.columns)
            cleaned_df = pivot_df[(pivot_df != 0).sum(axis=1) >= threshold]
            filtered_count = len(cleaned_df)

            # D. Imputation (Gap Filling)
            if impute_choice == "Half of Minimum Value":
                min_val = cleaned_df[cleaned_df > 0].min().min()
                cleaned_df = cleaned_df.replace(0, min_val / 2)

            # E. TIC Normalization
            # We calculate normalization on the cleaned data
            tic_norm = cleaned_df.div(cleaned_df.sum(axis=0), axis=1) * 1000000 

            # F. Auto-Grouping (Metadata Row)
            # MetaboAnalyst needs a Label row as the second row
            groups = [str(col).split('_')[0] for col in cleaned_df.columns]
            metadata_row = pd.DataFrame([groups], columns=cleaned_df.columns, index=['Label (Group)'])
            
            final_pivoted = pd.concat([metadata_row, cleaned_df])
            final_normalized = pd.concat([metadata_row, tic_norm])

            # --- 6. RESULTS & DOWNLOADS ---
            st.markdown("---")
            st.markdown(f"### Results: Found {raw_count} features → Kept {filtered_count} high-quality features.")
            
            tab1, tab2 = st.tabs(["📊 Pivoted Data", "📈 Normalized Data"])
            
            with tab1:
                st.write("Ready for MetaboAnalyst (Pivoted & Filtered):")
                st.dataframe(final_pivoted.head(10))
                st.download_button("📥 Download Pivoted CSV", final_pivoted.to_csv().encode('utf-8'), "metabo_pivoted.csv", "text/csv")

            with tab2:
                st.write("TIC Normalized Data (Counts per Million):")
                st.dataframe(final_normalized.head(10))
                st.download_button("📥 Download Normalized CSV", final_normalized.to_csv().encode('utf-8'), "metabo_normalized.csv", "text/csv")

            st.balloons()
            st.success("Pre-processing Complete!")

        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption("Built by a Scientist for Scientists. Goal: Knowledge → Financial Freedom.")
