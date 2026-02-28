import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import plotly.express as px
from scipy.stats import ttest_ind, linregress
from fpdf import FPDF
from pyteomics import mzml
import os
import gc

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Metabo-Cleaner Pro | Enterprise", layout="wide")

# --- GLOBAL SETTINGS ---
contact_email = "abass.metabo@gmail.com"
payment_url = "https://sandbox.flutterwave.com/donate/rgxclpstozwl"

# --- 2. SIDEBAR ---
st.sidebar.title("Metabo-Cleaner Pro")
st.sidebar.info("Enterprise-Grade Bioinformatics for Pharma & Lead Discovery.")

st.sidebar.markdown("---")
st.sidebar.subheader("Private Consulting")
contact_url = f"mailto:{contact_email}?subject=Enterprise%20Inquiry"
st.sidebar.markdown(f"[Email Lead Architect]({contact_url})")
if st.sidebar.button("Show Email Address"):
    st.sidebar.code(contact_email)

st.sidebar.markdown("---")
st.sidebar.subheader("Research Fund")
st.sidebar.markdown(f"[Sponsor this Project]({payment_url})")

st.sidebar.markdown("---")
st.sidebar.caption(f"© 2026 Yusuf Bioinformatics | {contact_email}")

# --- 3. HELPER: PDF GENERATOR ---
def create_pdf_report(g1, g2, feat_count, accuracy, leads_count):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Pharma-Grade Discovery Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "1. Executive Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    summary = (f"The analysis identified metabolic differences between group {g1} and {g2}. "
               f"A total of {feat_count} high-quality features were analyzed. "
               f"Identification found {leads_count} Lipinski-compliant drug candidates. "
               f"Validation Accuracy (Random Forest): {accuracy:.1%}.")
    pdf.multi_cell(0, 10, summary)
    return bytes(pdf.output())

# --- 4. MAIN INTERFACE ---
st.title("Metabo-Cleaner Pro: Drug Discovery Edition")

mode = st.radio("Select Analysis Module:", 
                ("Raw Data Processor", "Drug Discovery & Machine Learning Dashboard"))

# ============================================
# MODULE 1: RAW mzML BATCH PROCESSOR
# ============================================
if mode == "Raw Data Processor":
    st.subheader("Enterprise Batch Feature Extraction")
    uploaded_mzmls = st.file_uploader("Select .mzML files", type=["mzml"], accept_multiple_files=True)
    
    if uploaded_mzmls and st.button("Start Extraction"):
        all_features = []
        progress_bar = st.progress(0)
        status = st.empty()
        for i, file in enumerate(uploaded_mzmls):
            status.text(f"Processing: {file.name}")
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
                del rows; gc.collect() 
            except Exception as e: st.error(f"Error: {e}")
            progress_bar.progress((i + 1) / len(uploaded_mzmls))
        full_df = pd.concat(all_features, ignore_index=True)
        st.success("Batch Complete.")
        st.download_button("Download Feature Table", full_df.to_csv(index=False).encode('utf-8'), "metabo_features.csv")
        if os.path.exists("temp.mzml"): os.remove("temp.mzml")

# ============================================
# MODULE 2: DISCOVERY DASHBOARD
# ============================================
else:
    uploaded_file = st.file_uploader("Upload Quantified Table (.csv)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        with st.expander("Pharma Discovery Configuration"):
            c1, c2, c3, c4 = st.columns(4)
            mz_col = c1.selectbox("m/z Column", df.columns, index=0)
            rt_col = c2.selectbox("RT Column", df.columns, index=1 if "RT_min" not in df.columns else df.columns.get_loc("RT_min"))
            sm_col = c3.selectbox("Sample ID", df.columns, index=df.columns.get_loc("Sample") if "Sample" in df.columns else 0)
            in_col = c4.selectbox("Intensity", df.columns, index=df.columns.get_loc("Intensity") if "Intensity" in df.columns else 2)
            
            f1, f2, f3, f4 = st.columns(4)
            mz_bin = f1.slider("Alignment bin", 1, 5, 3)
            min_pres = f2.slider("80% Rule filter", 0, 100, 80)
            ion_mode = f3.selectbox("Polarity", ["Negative [M-H]-", "Positive [M+H]+"])
            scaling = f4.selectbox("Scaling", ["Pareto Scaling", "Auto-Scaling", "None"])

            st.markdown("---")
            st.write("Dose-Response Settings (Optional)")
            dose_map = st.text_input("Enter Doses (Format -> Group:Value, Group:Value)", placeholder="e.g., APL:0, MCL:100")

        if st.button("Run Pipeline"):
            try:
                # 1. CLEANING
                df['Temp_ID'] = df[mz_col].round(mz_bin).astype(str) + "_" + df[rt_col].round(2).astype(str)
                pivot = df.pivot_table(index='Temp_ID', columns=sm_col, values=in_col, aggfunc='mean').fillna(0)
                cleaned = pivot[(pivot != 0).sum(axis=1) >= (min_pres/100)*len(pivot.columns)]
                
                if cleaned.empty: st.error("No features left. Adjust filters."); st.stop()
                
                min_v = cleaned[cleaned > 0].min().min()
                tic_norm = cleaned.replace(0, min_v / 2).div(cleaned.sum(axis=0), axis=1) * 1000000 
                tic_norm = tic_norm[tic_norm.std(axis=1) > 0]

                # 2. STATS & ML
                groups = [str(s).split('_')[0] for s in tic_norm.columns]
                unique_g = sorted(list(set(groups)))
                X = tic_norm.T
                if scaling == "Pareto Scaling": X_s = (X - X.mean()) / np.sqrt(X.std().replace(0, np.nan))
                else: X_s = (X - X.mean()) / X.std()
                
                pca_res = PCA(n_components=2).fit_transform(X_s.fillna(0))
                
                # Discovery Calculations
                g1_c, g2_c = [c for c in tic_norm.columns if c.startswith(unique_g[0])], [c for c in tic_norm.columns if c.startswith(unique_g[1])]
                _, pvals = ttest_ind(tic_norm[g1_c], tic_norm[g2_c], axis=1)
                log2fc = np.log2(tic_norm[g2_c].mean(axis=1) / tic_norm[g1_c].mean(axis=1).replace(0, 0.001))
                
                # Build stats_df and Ensure ID is a COLUMN, not an index
                stats_df = pd.DataFrame({
                    'ID': tic_norm.index,
                    'p': pvals,
                    'log10p': -np.log10(pvals),
                    'Log2FC': log2fc
                }).reset_index(drop=True)
                stats_df['Significant'] = (stats_df['p'] < 0.05) & (abs(stats_df['Log2FC']) > 1)
                
                y_ml = [1 if g == unique_g[-1] else 0 for g in groups]
                acc = cross_val_score(RandomForestClassifier(), X_s.fillna(0), y_ml, cv=3).mean()

                # --- 3. DOSE RESPONSE LOGIC ---
                dose_ready = False
                if dose_map and ":" in dose_map:
                    try:
                        d_dict = {item.split(":")[0].strip(): float(item.split(":")[1].strip()) for item in dose_map.split(",")}
                        sample_doses = [float(d_dict.get(g, 0)) for g in groups]
                        if len(set(sample_doses)) > 1:
                            r_values = []
                            for i in range(len(tic_norm)):
                                r_val = linregress(sample_doses, tic_norm.iloc[i].values)[2]
                                r_values.append(r_val**2)
                            stats_df['Dose_R2'] = r_values
                            dose_ready = True
                    except:
                        st.warning("Dose format incorrect. Skipping correlation.")

                # TABS
                t1, t2, t3, t4, t5 = st.tabs(["📊 Quality", "🔵 Multivariate", "🌋 Discovery", "💊 Pharma Leads", "📋 Export Report"])
                
                with t1: st.plotly_chart(px.box(X.melt(), y='value', title="Intensity Distribution"), use_container_width=True)
                with t2: st.plotly_chart(px.scatter(x=pca_res[:,0], y=pca_res[:,1], color=groups, title="Group Separation"), use_container_width=True)
                with t3: st.plotly_chart(px.scatter(stats_df, x='Log2FC', y='log10p', color='Significant', hover_name='ID', title="Discovery Map"), use_container_width=True)
                
                with t4:
                    st.subheader("Lead Identification & Lipinski Filtering")
                    hits = stats_df[stats_df['Significant']].copy()
                    hits['m/z'] = hits['ID'].apply(lambda x: float(x.split('_')[0]))
                    hits['Neutral Mass'] = (hits['m/z'] + 1.0078) if ion_mode.startswith("Neg") else (hits['m/z'] - 1.0078)
                    hits['Lipinski_MW'] = hits['Neutral Mass'] < 500
                    hits['PubChem'] = hits['m/z'].apply(lambda x: f"https://pubchem.ncbi.nlm.nih.gov/#query={x}")
                    
                    st.metric("Drug-Like Leads identified", len(hits[hits['Lipinski_MW']]))
                    
                    display_cols = ['ID', 'Neutral Mass', 'Lipinski_MW', 'p']
                    if dose_ready: display_cols.append('Dose_R2')
                    display_cols.append('PubChem')
                    
                    st.dataframe(hits[display_cols].sort_values('p'), column_config={"PubChem": st.column_config.LinkColumn("Identify")})

                with t5:
                    st.subheader("Final Data Package")
                    pdf_bytes = create_pdf_report(unique_g[0], unique_g[1], len(stats_df[stats_df['Significant']]), acc, len(hits[hits['Lipinski_MW']]))
                    st.download_button(label="Download Professional PDF Report", data=pdf_bytes, file_name="Discovery_Report.pdf", mime="application/pdf")
                
                st.balloons()
            except Exception as e: st.error(f"Pipeline Error: {e}")

st.markdown("---")
st.caption(f"Metabo-Cleaner Pro Enterprise | {contact_email}")
