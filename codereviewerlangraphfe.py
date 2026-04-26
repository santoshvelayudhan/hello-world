import os
import shutil
import git
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_mistralai import ChatMistralAI
from fpdf import FPDF # New dependency for PDF

app = FastAPI()

# Configuration (Use Environment Variables in Production)
GITHUB_USER = "username"
GITHUB_PAT = "your_pat_token" 
REPO_NAME = "Kinisi-code"

class CodeIssue(BaseModel):
    severity: str
    category: str
    line_number: int
    title: str
    description: str
    suggestion: str
    confidence: float

# --- Report Generation Logic ---
class ReportGenerator:
    @staticmethod
    def to_markdown(result: Dict) -> str:
        issues = result.get("issues", [])
        errors = [i for i in issues if i['severity'] in ["Critical", "High"]]
        warnings = [i for i in issues if i['severity'] in ["Medium", "Low"]]
        
        md = f"# 🛡️ SpringBoot Audit Report: {REPO_NAME}\n\n"
        md += f"**Quality Score:** {result.get('quality_score', 'N/A')}/100\n"
        md += f"**Errors (🔴):** {len(errors)} | **Warnings (🟡):** {len(warnings)}\n\n"
        md += "## 🔍 Detailed Findings\n"
        
        for i in issues:
            type_label = "ERROR" if i['severity'] in ["Critical", "High"] else "WARNING"
            md += f"### [{type_label}] {i['title']} (Line {i['line_number']})\n"
            md += f"- **Category:** {i['category']}\n"
            md += f"- **Description:** {i['description']}\n"
            md += f"- **Suggested Fix:** {i['suggestion']}\n\n"
        return md

    @staticmethod
    def to_pdf(result: Dict, filename: str):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(190, 10, f"SpringBoot Audit Report: {REPO_NAME}", ln=True, align='C')
        
        pdf.set_font("Arial", '', 12)
        pdf.cell(190, 10, f"Quality Score: {result.get('quality_score')}/100", ln=True)
        
        for i in result.get("issues", []):
            pdf.set_text_color(255, 0, 0) if i['severity'] in ["Critical", "High"] else pdf.set_text_color(255, 165, 0)
            pdf.set_font("Arial", 'B', 12)
            pdf.multi_cell(190, 10, f"[{i['severity'].upper()}] {i['title']} (Line {i['line_number']})")
            
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", '', 10)
            pdf.multi_cell(190, 7, f"Fix: {i['suggestion']}")
            pdf.ln(5)
        pdf.output(filename)

# --- LangGraph Nodes & API remain similar to previous turn ---
# [Insert previous LangGraph logic here...]

@app.post("/review")
async def run_audit():
    # Simulation of Graph Execution returning AnalysisResult-like dict [cite: 48-56]
    # For a real run, call graph.invoke()
    return {
        "quality_score": 85.0, 
        "issues": [
            {"severity": "Critical", "category": "Security", "line_number": 12, "title": "SQL Injection", "description": "Raw string concatenation in query", "suggestion": "Use PreparedStatements", "confidence": 0.95}
        ],
        "summary": "Audit completed successfully."
    }
