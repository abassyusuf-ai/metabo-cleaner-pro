import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="Metabo-Cleaner Pro", layout="wide", page_icon="🧪")

# --- SIDEBAR ---
st.sidebar.title("About")
st.sidebar.info("Developed by a Metabolomics Postdoc to simplify data preprocessing.")
st.sidebar.markdown("---")
st.sidebar.write("🙏 **Support this project**")
st.sidebar.markdown("[☕ Buy me a coffee](https://www.buymeacoffee.com/abassyusuf)") 

st.title("🧪 Metabo-Cleaner Pro")
st.markdown("### Pivot, Clean, and Normalize your Mass Spec Data")

uploaded_file = st.file_uploader("Step 1: Upload your Long-Format .csv file", type=["csv"])

if uploaded_file is not None:
    # 1. Load Data
    df = pd.read_csv(uploaded_file)
    st.success("File Uploaded Successfully!")
    
    with st.expander("Preview Raw Data"):
        st.dataframe(df.head())

    # 2. Configure Columns
    st.markdown("### Step 2: Configure Columns")
    col1, col2, col3, col4 = st.columns(4)
    with col1: mz_col = st.selectbox("m/z Column", df.columns, index=0)
    with col2: rt_col = st.selectbox("RT Column", df.columns, index=1)
    with col3: sample_col = st.selectbox("Sample Name Column", df.columns, index=4 if len(df.columns)>4 else 0)
    with col4: intensity_col = st.selectbox("Intensity Column", df.columns, index=2)

    # 3. Alignment Settings
    st.markdown("### Step 3: Alignment & Grouping Settings")
    col_a, col_b = st.columns(2)
    with col_a:
        mz_digits = st.slider("m/z Rounding (Decimals)", 1, 5, 3)
    with col_b:
        rt_digits = st.slider("RT Rounding (Decimals)", 0, 2, 1)

    # 4. Processing
    if st.button("🚀 Process & Create Feature Table"):
        try:
            # --- FIX: Ensure columns are numeric (This prevents your crash) ---
            df[mz_col] = pd.to_numeric(df[mz_col], errors='coerce')
            df[rt_col] = pd.to_numeric(df[rt_col], errors='coerce')
            df[intensity_col] = pd.to_numeric(df[intensity_col], errors='coerce')
            
            # Remove any rows that became 'NaN' after forcing to numeric
            df = df.dropna(subset=[mz_col, rt_col, intensity_col])

            # SMART ALIGNMENT
            df['mz_align'] = df[mz_col].round(mz_digits)
            df['rt_align'] = df[rt_col].round(rt_digits)
            df['Feature_ID'] = df['mz_align'].astype(str) + "_" + df['rt_align'].astype(str)
            
            # PIVOT
            pivot_df = df.pivot_table(
                index='Feature_ID', 
                columns=sample_col, 
                values=intensity_col, 
                aggfunc='mean'
            ).fillna(0)
            
            # AUTO-GROUPER (First part of sample name before the first underscore)
            groups = [str(col).split('_')[0] for col in pivot_df.columns]
            metadata_row = pd.DataFrame([groups], columns=pivot_df.columns, index=['Label (Group)'])
            
            final_output = pd.concat([metadata_row, pivot_df])

            st.markdown("### Step 4: View & Normalize")
            st.write("✨ **Ready-to-Analyze Table (with Group Labels):**")
            st.dataframe(final_output.head(10))

            # TIC NORMALIZATION
            tic_norm = pivot_df.div(pivot_df.sum(axis=0), axis=1) * 1000000 
            final_norm_output = pd.concat([metadata_row, tic_norm])
            
            st.write("✨ **TIC Normalized Table:**")
            st.dataframe(final_norm_output.head(10))

            # DOWNLOADS
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("📥 Download MetaboAnalyst CSV", final_output.to_csv().encode('utf-8'), "metabo_groups.csv", "text/csv")
            with c2:
                st.download_button("📥 Download TIC Normalized CSV", final_norm_output.to_csv().encode('utf-8'), "normalized_groups.csv", "text/csv")
            
            st.balloons()
            st.success(f"Successfully aligned {len(pivot_df)} features!")

        except Exception as e:
            st.error(f"Error processing data: {e}")
            st.warning("Make sure the columns you selected contain only numbers (no text).")
