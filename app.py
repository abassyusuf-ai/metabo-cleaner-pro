import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
import plotly.express as px

st.set_page_config(page_title="Metabo-Cleaner Pro", layout="wide", page_icon="🧪")

# --- 1. SIDEBAR & MONETIZATION ---
st.sidebar.title("Metabo-Cleaner Pro")
st.sidebar.info("A specialized tool for Pre-processing and Visualizing Metabolomics Data.")
st.sidebar.markdown("---")
st.sidebar.subheader("🙏 Support Development")
st.sidebar.write("If this tool saved you hours of work, consider supporting my research.")
st.sidebar.markdown("[☕ Buy me a coffee](https://www.buymeacoffee.com/abassyusuf)") 

# --- 2. HEADER ---
st.title("🧪 Metabo-Cleaner Pro: Analytics Dashboard")
st.markdown("### Step 1: Upload your Long-Format .csv file")
uploaded_file = st.file_uploader("Drop your file here", type=["csv"])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.success("✅ File Uploaded successfully!")
    
    with st.expander("Preview Raw Data"):
        st.dataframe(df.head())

    # --- 3. CONFIGURATION ---
    st.markdown("### Step 2: Configure Data & Scaling")
    col1, col2, col3, col4 = st.columns(4)
    with col1: mz_col = st.selectbox("m/z Column", df.columns, index=0)
    with col2: rt_col = st.selectbox("RT Column", df.columns, index=1)
    with col3: sample_col = st.selectbox("Sample ID", df.columns, index=4 if len(df.columns)>4 else 0)
    with col4: intensity_col = st.selectbox("Intensity", df.columns, index=2)

    # --- 4. ADVANCED SETTINGS ---
    st.markdown("### Step 3: Pro Cleaning & Alignment Settings")
    c_a, c_b, c_c, c_d = st.columns(4)
    with c_a:
        mz_digits = st.slider("m/z Rounding", 1, 5, 3, help="Groups features with similar m/z.")
    with c_b:
        min_presence = st.slider("Min Presence (%)", 0, 100, 20, help="Removes noisy rows.")
    with c_c:
        impute_choice = st.selectbox("Gap Filling (0s)", ["Keep as 0", "Half of Minimum Value"])
    with c_d:
        scaling_method = st.selectbox("Scaling for PCA", ["None", "Auto-Scaling (UV)", "Pareto Scaling"])

    # --- 5. THE ENGINE ---
    if st.button("🚀 Process, Clean & Visualize"):
        try:
            # A. Clean data types
            df[mz_col] = pd.to_numeric(df[mz_col], errors='coerce')
            df[rt_col] = pd.to_numeric(df[rt_col], errors='coerce')
            df[intensity_col] = pd.to_numeric(df[intensity_col], errors='coerce')
            df = df.dropna(subset=[mz_col, rt_col, intensity_col])

            # B. Alignment (Smart Feature ID)
            df['Feature_ID'] = df[mz_col].round(mz_digits).astype(str) + "_" + df[rt_col].round(1).astype(str)
            pivot_df = df.pivot_table(index='Feature_ID', columns=sample_col, values=intensity_col, aggfunc='mean').fillna(0)
            raw_count = len(pivot_df)

            # C. Filtering (Presence threshold)
            threshold = (min_presence / 100) * len(pivot_df.columns)
            cleaned_df = pivot_df[(pivot_df != 0).sum(axis=1) >= threshold]
            filtered_count = len(cleaned_df)

            # D. Imputation (Gap Filling)
            data_to_viz = cleaned_df.copy()
            if impute_choice == "Half of Minimum Value":
                min_val = data_to_viz[data_to_viz > 0].min().min()
                data_to_viz = data_to_viz.replace(0, min_val / 2)

            # E. Normalization (TIC)
            tic_norm = data_to_viz.div(data_to_viz.sum(axis=0), axis=1) * 1000000 

            # F. Prepare Metadata labels
            groups = [str(col).split('_')[0] for col in cleaned_df.columns]
            metadata_row = pd.DataFrame([groups], columns=cleaned_df.columns, index=['Label (Group)'])
            
            final_pivoted = pd.concat([metadata_row, data_to_viz])
            final_normalized = pd.concat([metadata_row, tic_norm])

            # --- 6. VISUALIZATION LOGIC ---
            # Transpose for PCA (Samples must be rows)
            pca_input = tic_norm.T
            
            # SCALING
            if scaling_method == "Auto-Scaling (UV)":
                pca_input = (pca_input - pca_input.mean()) / pca_input.std()
            elif scaling_method == "Pareto Scaling":
                pca_input = (pca_input - pca_input.mean()) / np.sqrt(pca_input.std())
            
            # PCA Calculation
            pca = PCA(n_components=2)
            components = pca.fit_transform(pca_input)
            exp_var = pca.explained_variance_ratio_ * 100
            
            pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
            pca_df['Sample'] = pca_input.index
            pca_df['Group'] = [str(s).split('_')[0] for s in pca_df['Sample']]

            # --- 7. TABS FOR OUTPUT ---
            st.markdown("---")
            st.markdown(f"### Analysis Results: {raw_count} original features → {filtered_count} high-quality features.")
            
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Tables", "📈 Normalized", "🔵 PCA Plot", "🔥 Heatmap"])
            
            with tab1:
                st.write("Cleaned & Pivoted Table:")
                st.dataframe(final_pivoted.head(10))
                st.download_button("📥 Download Pivoted CSV", final_pivoted.to_csv().encode('utf-8'), "metabo_pivoted.csv")

            with tab2:
                st.write("TIC Normalized Table:")
                st.dataframe(final_normalized.head(10))
                st.download_button("📥 Download Normalized CSV", final_normalized.to_csv().encode('utf-8'), "metabo_normalized.csv")

            with tab3:
                fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Group', hover_name='Sample',
                                     title=f"PCA Analysis (PC1: {exp_var[0]:.1f}%, PC2: {exp_var[1]:.1f}%)",
                                     template="plotly_white", symbol='Group')
                st.plotly_chart(fig_pca, use_container_width=True)
                st.info("Pareto Scaling is recommended for Metabolomics to reduce the influence of massive peaks.")

            with tab4:
                # Log transform for heatmap visualization
                log_heat = np.log2(tic_norm + 1)
                fig_heat = px.imshow(log_heat.head(100), 
                                     aspect="auto",
                                     color_continuous_scale='RdBu_r',
                                     title="Top 100 Features Heatmap (Log2 Transformed)")
                st.plotly_chart(fig_heat, use_container_width=True)

            st.balloons()
            st.success("Analysis Complete!")

        except Exception as e:
            st.error(f"Error during processing: {e}")

st.markdown("---")
st.caption("Built for Scientists. Your path to knowledge and financial freedom.")
