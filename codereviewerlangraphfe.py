import streamlit as st
import requests
import base64

st.set_page_config(page_title="Kinisi Auditor", layout="wide")

st.title("🛡️ Kinisi-Code SpringBoot Auditor")

with st.sidebar:
    st.header("Report Settings")
    report_choice = st.radio("Select Download Format:", ["Markdown (.md)", "PDF (.pdf)"])
    st.divider()
    st.info("Uses GitHub PAT for secure repo access.")

if st.button("🚀 Start Audit & Generate Report", type="primary"):
    with st.spinner("Analyzing SpringBoot Repository..."):
        response = requests.post("http://localhost:8000/review")
        if response.status_code == 200:
            result = response.json()
            st.success("Analysis Complete!")

            # Categorization Logic [cite: 1160-1164]
            errors = [i for i in result['issues'] if i['severity'] in ["Critical", "High"]]
            warnings = [i for i in result['issues'] if i['severity'] in ["Medium", "Low"]]

            # Dashboard [cite: 1032-1040]
            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 ERRORS", len(errors))
            c2.metric("🟡 WARNINGS", len(warnings))
            c3.metric("Score", f"{result['quality_score']}/100")

            # Report Generation & Download [cite: 1266-1271]
            if "Markdown" in report_choice:
                # Assuming backend or a helper generates this string
                report_text = f"# Audit Report\n\nErrors: {len(errors)}" 
                st.download_button("📥 Download Markdown Report", report_text, file_name="audit.md")
            else:
                # For PDF, typically the backend would save the file and frontend serves it
                st.info("PDF Generation triggered. (In production, the backend returns the binary file).")
                # Example of a simple encoded download for demo:
                st.download_button("📥 Download PDF Report", data="BinaryDataPlaceholder", file_name="audit.pdf")

        else:
            st.error("Audit failed. Check backend logs.")
