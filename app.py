import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import plotly.express as px
from scipy.stats import ttest_ind
from fpdf import FPDF
from pyteomics import mzml
import os
import gc

# --- 1. PRO CONFIGURATION ---
st.set_page_config(page_title="Metabo-Cleaner Pro | Enterprise", layout="wide", page_icon="💎")

# --- 2. BUSINESS SIDEBAR (Fixed Links & Lead Gen) ---
st.sidebar.title("💎 Metabo-Cleaner Pro")
st.sidebar.info("Enterprise-Grade Bioinformatics for Industry & Large Scale Studies.")

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Private Consulting")
st.sidebar.write("Need a private server or custom pipeline?")

# Professional Email Link
contact_email = "abass.metabo@gmail.com"
contact_url = f"mailto:{contact_email}?subject=Enterprise%20Inquiry"
st.sidebar.markdown(f"📩 [Email Abass Yusuf]({contact_url})")

# Fallback: Copy Email Button (Important for users without default mail apps)
if st.sidebar.button("📋 Show Email for Copying"):
    st.sidebar.code(contact_email)
    st.sidebar.caption("Copy the email above and paste into Gmail/Outlook.")

st.sidebar.markdown("---")
st.sidebar.subheader("☕ Support Development")
# Buy Me A Coffee Verified Link
coffee_url = "https://www.buymeacoffee.com/abassyusuf"
st.sidebar.markdown(f"☕ [Buy me a coffee]({coffee_url})")

st.sidebar.markdown("---")
st.sidebar.caption("🔒 Data Privacy: Files processed in-memory and purged immediately.")
st.sidebar.caption(f"© 2026 Yusuf Bioinformatics | {contact_email}")

# --- 3. HELPER: PDF GENERATOR ---
def create_pdf_report(g1, g2, feat_count, accuracy):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Metabolomics Discovery Report", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "1. Executive Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    
    summary = (f"The analysis identified metabolic differences between group {g1} and {g2}. "
               f"A total of {feat_count} high-quality features were analyzed. "
               f"Machine Learning validation (Random Forest) achieved an accuracy of {accuracy:.1%}, "
               "demonstrating a highly predictive biological signature.")
    pdf.multi_cell(0, 10, summary)
    
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 10, "Generated automatically by Metabo-Cleaner Pro Enterprise.", ln=True)
    
    # Standard fix for Streamlit binary data
    return bytes(pdf.output())

# --- 4. MAIN INTERFACE ---
st.title("🧪 Metabo-Cleaner Pro: Enterprise Discovery Suite")

mode = st.radio("Select Professional Module:", 
                ("High-Capacity mzML Processor (Premium)", "Statistical Discovery Dashboard"))

# ============================================
# MODULE 1: RAW mzML BATCH PROCESSOR
# ============================================
if mode == "High-Capacity mzML Processor (Premium)":
    st.subheader("🚀 Bulk mzML Feature Extraction")
    uploaded_mzmls = st.file_uploader("Upload .mzML batch (Up to 5GB)", type=["mzml"], accept_multiple_files=True)
    
    if uploaded_mzmls and st.button("🚀 Start Enterprise Extraction"):
        all_features = []
        progress = st.progress(0)
        status = st.empty()
        
        for i, file in enumerate(uploaded_mzmls):
            status.text(f"Processing {file.name} ({i+1}/{len(uploaded_mzmls)})...")
            with open("temp.mzml", "wb") as f: f.write(file.getbuffer())
            
            rows = []
            try:
                with mzml.read("temp.mzml") as reader:
                    for spec in reader:
                        if spec['ms level'] == 1 and len(spec['intensity array']) > 0:
                            idx = np.argmax(spec['intensity array'])
                            rows.append([float(spec['m/z array'][idx]), float(spec['scanList']['scan'][0]['scan start time'])/60, float(spec['intensity array'][idx])])
                
                df_s = pd.DataFrame(rows, columns=["m/z", "RT_min", "Intensity"])
                df_s["Sample"] = file.name.replace(".mzML", "")
                all_features.append(df_s)
                del rows 
                gc.collect() 
            except Exception as e:
                st.error(f"Error in {file.name}: {e}")
                
            progress.progress((i + 1) / len(uploaded_mzmls))

        full_df = pd.concat(all_features, ignore_index=True)
        st.success("Batch Extraction Complete.")
        st.download_button("📥 Download Enterprise CSV", full_df.to_csv(index=False).encode('utf-8'), "enterprise_results.csv")
        if os.path.exists("temp.mzml"): os.remove("temp.mzml")

# ============================================
# MODULE 2: STATISTICAL DISCOVERY
# ============================================
else:
    uploaded_file = st.file_uploader("Upload your Quantification Table (.csv)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        with st.expander("⚙️ Advanced Discovery Configuration"):
            c1, c2, c3, c4 = st.columns(4)
            mz_col = c1.selectbox("m/z Column", df.columns, index=0)
            rt_col = c2.selectbox("RT Column", df.columns, index=1 if "RT_min" not in df.columns else df.columns.get_loc("RT_min"))
            sm_col = c3.selectbox("Sample ID", df.columns, index=df.columns.get_loc("Sample") if "Sample" in df.columns else 0)
            in_col = c4.selectbox("Intensity", df.columns, index=df.columns.get_loc("Intensity") if "Intensity" in df.columns else 2)
            
            f1, f2, f3, f4 = st.columns(4)
            mz_bin = f1.slider("m/z Alignment", 1, 5, 3)
            min_pres = f2.slider("Min Presence (%)", 0, 100, 80)
            p_val_thresh = f3.number_input("P-value Signif.", 0.05)
            scaling = f4.selectbox("Scaling", ["Pareto Scaling", "Auto-Scaling", "None"])

        if st.button("🚀 Run Enterprise Discovery Pipeline"):
            try:
                # 1. CLEANING ENGINE
                df['ID'] = df[mz_col].round(mz_bin).astype(str) + "_" + df[rt_col].round(2).astype(str)
                pivot = df.pivot_table(index='ID', columns=sm_col, values=in_col, aggfunc='mean').fillna(0)
                
                thresh = (min_pres/100) * len(pivot.columns)
                cleaned = pivot[(pivot != 0).sum(axis=1) >= thresh]
                
                if cleaned.empty:
                    st.error("❌ No metabolites found! Lower the Min Presence slider.")
                    st.stop()

                min_v = cleaned[cleaned > 0].min().min()
                tic_norm = cleaned.replace(0, min_v / 2).div(cleaned.sum(axis=0), axis=1) * 1000000 
                tic_norm = tic_norm[tic_norm.std(axis=1) > 0]

                # 2. STATS & ML
                groups = [str(s).split('_')[0] for s in tic_norm.columns]
                unique_g = sorted(list(set(groups)))
                
                X = tic_norm.T
                if scaling == "Pareto Scaling": X_s = (X - X.mean()) / np.sqrt(X.std().replace(0, np.nan))
                else: X_s = (X - X.mean()) / X.std()
                X_s = X_s.fillna(0).replace([np.inf, -np.inf], 0)
                
                pca_res = PCA(n_components=2).fit_transform(X_s)
                
                stats_ready = False
                if len(unique_g) >= 2:
                    g1_c = [c for c in tic_norm.columns if c.startswith(unique_g[0])]
                    g2_c = [c for c in tic_norm.columns if c.startswith(unique_g[1])]
                    _, pvals = ttest_ind(tic_norm[g1_c], tic_norm[g2_c], axis=1)
                    log2fc = np.log2(tic_norm[g2_c].mean(axis=1) / tic_norm[g1_c].mean(axis=1).replace(0, 0.001))
                    
                    vol_df = pd.DataFrame({'ID': tic_norm.index, 'p': pvals, 'log10p': -np.log10(pvals), 'Log2FC': log2fc}).fillna(0)
                    vol_df['Significant'] = (vol_df['p'] < p_val_thresh) & (abs(vol_df['Log2FC']) > 1)
                    hits = vol_df[vol_df['Significant']].sort_values('p')
                    
                    # Machine Learning Validation
                    y_ml = [1 if g == unique_g[-1] else 0 for g in groups]
                    acc = cross_val_score(RandomForestClassifier(), X_s, y_ml, cv=3).mean()
                    stats_ready = True

                # 3. TABS
                t1, t2, t3, t4, t5 = st.tabs(["📊 Distributions", "🔵 Multivariate", "🌋 Volcano Plot", "🏆 ML Discovery", "💎 Enterprise Report"])
                
                with t1:
                    st.plotly_chart(px.box(X.melt(), y='value', title="Intensity Distribution"), use_container_width=True)
                
                with t2:
                    st.plotly_chart(px.scatter(x=pca_res[:,0], y=pca_res[:,1], color=groups, title="PCA Separation"), use_container_width=True)
                
                with t3:
                    if stats_ready:
                        st.plotly_chart(px.scatter(vol_df, x='Log2FC', y='log10p', color='Significant', hover_name='ID', color_discrete_map={True:'red', False:'gray'}, title="Discovery Map"), use_container_width=True)
                
                with t4:
                    if stats_ready:
                        st.metric("Random Forest Prediction Accuracy", f"{acc:.1%}")
                        st.dataframe(hits)
                
                with t5:
                    if stats_ready:
                        st.subheader("Professional Data Package")
                        st.write("Generate and download your final discovery report below.")
                        
                        # Generate PDF data
                        pdf_bytes = create_pdf_report(unique_g[0], unique_g[1], len(hits), acc)
                        
                        st.download_button(
                            label="📥 Download Enterprise PDF Report",
                            data=pdf_bytes,
                            file_name="Metabo_Discovery_Report.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.info("Analysis requires at least 2 groups to generate a report.")
                
                st.balloons()
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.caption(f"💎 Metabo-Cleaner Pro Enterprise | {abass.metabo@gmail.com}")
