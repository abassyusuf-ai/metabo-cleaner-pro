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

   # --- PRO SETTINGS FOR ALIGNMENT ---
    st.markdown("### Step 3: Alignment & Grouping Settings")
    col_a, col_b = st.columns(2)
    with col_a:
        mz_digits = st.slider("m/z Rounding (Decimals)", 1, 5, 3, help="Higher = stricter alignment. 3 is standard for Orbitrap/Q-TOF.")
    with col_b:
        rt_digits = st.slider("RT Rounding (Decimals)", 0, 2, 1, help="Rounding RT to 0.1 or 0.2 helps align retention time drift.")

    # --- THE BIG PROCESS BUTTON ---
    if st.button("🚀 Process & Create Feature Table"):
        # 1. SMART ALIGNMENT (Rounding m/z and RT to group similar features)
        # This fixes the "Too many zeros" problem
        df['mz_align'] = df[mz_col].round(mz_digits)
        df['rt_align'] = df[rt_col].round(rt_digits)
        df['Feature_ID'] = df['mz_align'].astype(str) + "_" + df['rt_align'].astype(str)
        
        # 2. PIVOT (Turn Long format to Wide format)
        pivot_df = df.pivot_table(
            index='Feature_ID', 
            columns=sample_col, 
            values=intensity_col, 
            aggfunc='mean'
        ).fillna(0)
        
        # 3. AUTO-GROUPER (Extracts the first part of your sample name as the 'Class')
        # Example: APL_R1_b1 becomes 'APL'
        groups = [str(col).split('_')[0] for col in pivot_df.columns]
        metadata_row = pd.DataFrame([groups], columns=pivot_df.columns, index=['Label (Group)'])
        
        # Combine Label row with Data (Requirement for MetaboAnalyst)
        final_output = pd.concat([metadata_row, pivot_df])

        st.markdown("### Step 4: View & Normalize")
        st.write("✨ **Ready-to-Analyze Table (with Group Labels):**")
        st.dataframe(final_output.head(10)) # Show first 10 rows

        # 4. NORMALIZATION (TIC)
        # Note: We only normalize the data rows, not the 'Label' row
        tic_norm = pivot_df.div(pivot_df.sum(axis=0), axis=1) * 1000000 
        final_norm_output = pd.concat([metadata_row, tic_norm])
        
        st.write("✨ **TIC Normalized Table (Counts per Million):**")
        st.dataframe(final_norm_output.head(10))

        # 5. DOWNLOAD BUTTONS
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                label="📥 Download MetaboAnalyst Ready CSV", 
                data=final_output.to_csv().encode('utf-8'), 
                file_name="metabo_cleaned_groups.csv", 
                mime="text/csv"
            )
        with c2:
            st.download_button(
                label="📥 Download TIC Normalized CSV", 
                data=final_norm_output.to_csv().encode('utf-8'), 
                file_name="metabo_normalized_groups.csv", 
                mime="text/csv"
            )
        
        st.balloons()
        st.success(f"Successfully aligned {len(pivot_df)} features from your data!")
