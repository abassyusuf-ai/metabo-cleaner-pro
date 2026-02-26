import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
import plotly.express as px
from scipy.stats import ttest_ind

st.set_page_config(page_title="Metabo-Cleaner Pro", layout="wide", page_icon="🧪")

# --- 1. SIDEBAR & MONETIZATION (Your original sidebar) ---
st.sidebar.title("Metabo-Cleaner Pro")
st.sidebar.info("A specialized tool for Pre-processing, Visualizing and Discovery.")
st.sidebar.markdown("---")
st.sidebar.subheader("🙏 Support Development")
st.sidebar.write("If this tool saved you hours of work, consider supporting my research.")
st.sidebar.markdown("[☕ Buy me a coffee](https://www.buymeacoffee.com/abassyusuf)") 

# --- 2. HEADER ---
st.title("🧪 Metabo-Cleaner Pro: Discovery Edition")
st.markdown("### Step 1: Upload your Long-Format .csv file")
uploaded_file = st.file_uploader("Drop your file here", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File Uploaded successfully!")
    
    with st.expander("Preview Raw Data"):
        st.dataframe(df.head())

    # --- 3. CONFIGURATION (Your original selectors) ---
    st.markdown("### Step 2: Configure Data & Stats")
    col1, col2, col3, col4 = st.columns(4)
    with col1: mz_col = st.selectbox("m/z Column", df.columns, index=0)
    with col2: rt_col = st.selectbox("RT Column", df.columns, index=1)
    with col3: sample_col = st.selectbox("Sample ID", df.columns, index=4 if len(df.columns)>4 else 0)
    with col4: intensity_col = st.selectbox("Intensity", df.columns, index=2)

    # --- 4. ADVANCED SETTINGS ---
    st.markdown("### Step 3: Pro Cleaning & Stats Settings")
    c_a, c_b, c_c, c_d = st.columns(4)
    with c_a:
        mz_digits = st.slider("m/z Rounding", 1, 5, 3)
    with c_b:
        min_presence = st.slider("Min Presence (%)", 0, 100, 50)
    with c_c:
        p_val_thresh = st.number_input("P-value Significance", value=0.05, step=0.01)
    with c_d:
        scaling_method = st.selectbox("Scaling for PCA", ["Pareto Scaling", "Auto-Scaling (UV)", "None"])

    # --- 5. THE ENGINE (The logic you built) ---
    if st.button("🚀 Run Full Discovery Pipeline"):
        try:
            # A. Clean data types
            df[mz_col] = pd.to_numeric(df[mz_col], errors='coerce')
            df[rt_col] = pd.to_numeric(df[rt_col], errors='coerce')
            df[intensity_col] = pd.to_numeric(df[intensity_col], errors='coerce')
            df = df.dropna(subset=[mz_col, rt_col, intensity_col])

            # B. Alignment (Your Feature_ID logic)
            df['Feature_ID'] = df[mz_col].round(mz_digits).astype(str) + "_" + df[rt_col].round(1).astype(str)
            pivot_df = df.pivot_table(index='Feature_ID', columns=sample_col, values=intensity_col, aggfunc='mean').fillna(0)
            raw_count = len(pivot_df)

            # C. Filtering (Presence threshold)
            threshold = (min_presence / 100) * len(pivot_df.columns)
            cleaned_df = pivot_df[(pivot_df != 0).sum(axis=1) >= threshold]
            filtered_count = len(cleaned_df)

            # D. Imputation (Essential for Stats)
            min_val = cleaned_df[cleaned_df > 0].min().min()
            data_ready = cleaned_df.replace(0, min_val / 2)

            # E. Normalization (Your TIC logic)
            tic_norm = data_ready.div(data_ready.sum(axis=0), axis=1) * 1000000 

            # F. Statistical Discovery (NEW)
            unique_groups = sorted(list(set([str(s).split('_')[0] for s in tic_norm.columns])))
            
            if len(unique_groups) >= 2:
                g1, g2 = unique_groups[0], unique_groups[1]
                g1_cols = [c for c in tic_norm.columns if c.startswith(g1)]
                g2_cols = [c for c in tic_norm.columns if c.startswith(g2)]
                
                # Calculate means and log2 fold change
                m1, m2 = tic_norm[g1_cols].mean(axis=1), tic_norm[g2_cols].mean(axis=1)
                fc = m2 / m1
                log2fc = np.log2(fc.replace(0, 0.001))
                
                # Perform T-test
                t_stat, p_vals = ttest_ind(tic_norm[g1_cols], tic_norm[g2_cols], axis=1)
                
                stats_df = pd.DataFrame({
                    'Metabolite': tic_norm.index,
                    f'{g1}_Mean': m1,
                    f'{g2}_Mean': m2,
                    'Fold_Change': fc,
                    'Log2FC': log2fc,
                    'p_value': p_vals,
                    '-log10_p': -np.log10(p_vals)
                }).fillna(0)
                
                significant_hits = stats_df[(stats_df['p_value'] < p_val_thresh) & (abs(stats_df['Log2FC']) > 1)]

            # --- 7. TABS FOR OUTPUT (Keeping your tab structure and adding more) ---
            st.markdown("---")
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Tables", "🔵 PCA Plot", "🌋 Volcano Plot", "🏆 Top Biomarkers", "🔥 Heatmap"])
            
            with tab1:
                groups = [str(col).split('_')[0] for col in tic_norm.columns]
                metadata_row = pd.DataFrame([groups], columns=tic_norm.columns, index=['Label (Group)'])
                final_pivoted = pd.concat([metadata_row, data_ready])
                final_normalized = pd.concat([metadata_row, tic_norm])
                
                st.write("Cleaned Table:")
                st.dataframe(final_pivoted.head(5))
                st.download_button("📥 Download Pivoted CSV", final_pivoted.to_csv().encode('utf-8'), "metabo_pivoted.csv")
                st.download_button("📥 Download TIC Normalized CSV", final_normalized.to_csv().encode('utf-8'), "metabo_normalized.csv")

            with tab2:
                # Your PCA logic
                pca_input = tic_norm.T
                if scaling_method == "Auto-Scaling (UV)": pca_input = (pca_input - pca_input.mean()) / pca_input.std()
                elif scaling_method == "Pareto Scaling": pca_input = (pca_input - pca_input.mean()) / np.sqrt(pca_input.std())
                
                pca = PCA(n_components=2)
                components = pca.fit_transform(pca_input)
                exp_var = pca.explained_variance_ratio_ * 100
                pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
                pca_df['Group'] = [str(s).split('_')[0] for s in pca_input.index]
                pca_df['Sample'] = pca_input.index
                
                fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Group', hover_name='Sample',
                                     title=f"PCA Separation (PC1: {exp_var[0]:.1f}%)", template="plotly_white")
                st.plotly_chart(fig_pca, use_container_width=True)

            with tab3:
                # New Volcano Plot
                stats_df['Sig_Color'] = (stats_df['p_value'] < p_val_thresh) & (abs(stats_df['Log2FC']) > 1)
                fig_volcano = px.scatter(stats_df, x='Log2FC', y='-log10_p', color='Sig_Color',
                                         hover_name='Metabolite', color_discrete_map={True: 'red', False: 'gray'},
                                         title=f"Volcano Plot: {g1} vs {g2}")
                fig_volcano.add_hline(y=-np.log10(p_val_thresh), line_dash="dash", line_color="red")
                st.plotly_chart(fig_volcano, use_container_width=True)

            with tab4:
                st.write(f"### Found {len(significant_hits)} Significant Biomarkers")
                st.dataframe(significant_hits.sort_values('p_value'))
                st.download_button("📥 Download Statistical Report", stats_df.to_csv().encode('utf-8'), "discovery_report.csv")

            with tab5:
                # Your heatmap logic
                log_heat = np.log2(tic_norm + 1)
                fig_heat = px.imshow(log_heat.head(100), aspect="auto", color_continuous_scale='Viridis', title="Top 100 Features Heatmap")
                st.plotly_chart(fig_heat, use_container_width=True)

            st.balloons()
            st.success(f"Analysis complete! {raw_count} peaks → {filtered_count} kept.")

        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption("Built by a Scientist for Scientists. Goal: Knowledge → Financial Freedom.")
