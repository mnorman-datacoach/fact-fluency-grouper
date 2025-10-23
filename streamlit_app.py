# streamlit_app.py
# ------------------------------------------------------------
# Fact Fluency Grouper — robust PDF parsing (no OCR)
# - Anchors each % to the nearest operation header (Add/Sub/Mul/Div)
# - Matches % to the student's row by vertical y-center (prevents off-by-one)
# - Heuristics for 1/2/3 numbers -> MNT/WTF/NS
# - Small-group: prioritize high WTF + low MNT, ignore high NS
# - Display order everywhere: MNT | WTF | NS
# - CSV & PDF export; optional AI question (fallback if no key)
# ------------------------------------------------------------

import io
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# ---------------- UI ----------------
st.set_page_config(page_title="Fact Fluency Grouper", page_icon=None, layout="centered")
st.title("Fact Fluency Grouper")
st.caption("Upload an i-Ready Fact Fluency report → NS/WTF/MNT grouping → small group (high WTF + low MNT, low NS) → CSV/PDF export.")

col1, col2 = st.columns(2)
with col1:
    OPERATION = st.radio("Operation", ["Addition", "Subtraction", "Multiplication", "Division"], horizontal=True)
with col2:
    GRADE = st.selectbox("Grade", ["K", "1", "2", "3", "4", "5", "6"])

DOK = st.radio("DOK Level", ["1", "2", "3"], horizontal=True)
USE_AI = st.toggle("Generate AI question (optional)", value=True)

uploaded = st.file_uploader("Upload i-Ready Fact Fluency PDF (preferred) or CSV", type=["pdf", "csv"])
st.markdown("---")

# ---------------- Helpers ----------------
def clamp(n, lo, hi):
    return max(lo, min(hi, n))

def choose_group_size(n_students):
    return clamp(int(round(0.2 * max(n_students, 1))), 3, 6)

def assign_group(ns, wtf, mnt):
    # Largest wins; tie favors needier tier: NS > WTF > MNT
    trip = [("NS", ns), ("WTF", wtf), ("MNT", mnt)]
    order = {"NS": 3, "WTF": 2, "MNT": 1}
    trip.sort(key=lambda x: (x[1], order[x[0]]), reverse=True)
    return trip[0][0]

def recommend_focus(df_op):
    if df_op.empty:
        return "WTF"
    share_ns = (df_op["group"] == "NS").mean()
    share_wtf = (df_op["group"] == "WTF").mean()
    wtf_median = float(df_op["wtf"].median())
    if share_ns >= 0.20:
        return "NS"
    elif (wtf_median < 70) or (share_wtf >= 0.50):
        return "WTF"
    else:
        return "MNT"

# ---------------- Small-group prioritization (your rule) ----------------
def select_priority_students(df_op, focus_key):
    """
    Your prioritization:
      1) High NS not a priority (skip for now).
      2) High WTF AND low MNT (relative to peers) are the priority.
      3) High WTF AND high MNT are not the priority.
    Implementation:
      - Primary cut: WTF >= 60th percentile, MNT <= 40th percentile, NS <= 25
      - If too few, relax once: WTF >= median, MNT <= median, NS <= 35
      - Severity: prefer lower MNT, then higher WTF, penalize NS.
    """
    data = df_op.copy()
    if data.empty:
        return []

    group_size = choose_group_size(len(data))

    # Percentile cutoffs
    wtf_hi = float(data["wtf"].quantile(0.60))
    mnt_lo = float(data["mnt"].quantile(0.40))

    # NS caps
    ns_cap_primary = 25
    ns_cap_relaxed = 35

    # Primary candidate set
    cand = data[(data["wtf"] >= wtf_hi) & (data["mnt"] <= mnt_lo) & (data["ns"] <= ns_cap_primary)].copy()

    # Relax once if needed
    if len(cand) < group_size:
        wtf_med = float(data["wtf"].median())
        mnt_med = float(data["mnt"].median())
        relaxed = data[(data["wtf"] >= wtf_med) & (data["mnt"] <= mnt_med) & (data["ns"] <= ns_cap_relaxed)].copy()
        cand = pd.concat([cand, relaxed]).drop_duplicates()

    if cand.empty:
        return []

    # Severity score
    cand["severity"] = 0.7 * (100 - cand["mnt"]) + 0.3 * cand["wtf"] - 0.5 * cand["ns"]
    cand = cand.sort_values(by=["severity", "wtf", "mnt"], ascending=[False, False, True]).head(group_size)

    return [
        {"name": r["student"], "ns": int(r["ns"]), "wtf": int(r["wtf"]), "mnt": int(r["mnt"])}
        for _, r in cand.iterrows()
    ]

# ---------------- CSV parser ----------------
def parse_csv_to_dataframe(file: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file))
    cols_lower = {c.lower(): c for c in df.columns}
    req = ["student", "operation", "ns", "wtf", "mnt"]
    missing = [c for c in req if c not in cols_lower]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Required: {req}")
    df = df.rename(columns={cols_lower[c]: c for c in req})
    for k in ["ns", "wtf", "mnt"]:
        df[k] = pd.to_numeric(df[k], errors="coerce").fillna(0).astype(int).clip(0, 100)
    df["operation"] = df["operation"].astype(str).str.capitalize()
    return df[["student", "operation", "ns", "wtf", "mnt"]]

# ---------------- PDF parser (Y-anchored, header-anchored; no OCR) ----------------
def parse_pdf_to_dataframe(file: bytes) -> pd.DataFrame:
    """
    Strategy:
      1) Use PyMuPDF to get spans (text + bbox).
      2) Detect x-centers of operation headers: Addition/Subtraction/Multiplication/Division.
      3) For each student name span, collect only % spans whose vertical center is within ±ROW_TOL of the name's y-center.
      4) Assign each % to the nearest header by x-center for that row.
      5) For each cell, map 1/2/3 numbers -> MNT/WTF/NS with heuristics.
      6) Fallback to pdfplumber text if headers not found.
    """
    def clamp_pct(v):
        try:
            return int(np.clip(int(v), 0, 100))
        except Exception:
            return 0

    # Interpret one operation cell (percent list in left->right visual order)
    def interpret_cell(nums):
        nums = [clamp_pct(n) for n in nums if n is not None]
        nums = [n for n in nums if 0 <= n <= 100]
        if not nums:
            return {"mnt": 0, "wtf": 0, "ns": 100}

        if len(nums) >= 3:
            # Assume left->right: MNT, WTF, NS
            mnt, wtf, ns = nums[0], nums[1], nums[2]
            return {"mnt": mnt, "wtf": wtf, "ns": ns}

        if len(nums) == 2:
            a, b = nums
            s = a + b
            if s == 100:
                # Two numbers span full bar → either (WTF, NS) or (MNT, WTF)
                if b >= 75 or a <= 25:
                    return {"mnt": 0, "wtf": a, "ns": b}   # light blue + gray
                else:
                    return {"mnt": a, "wtf": b, "ns": 0}   # dark blue + light blue
            # Not the full bar: treat as MNT + WTF with NS remainder
            return {"mnt": a, "wtf": b, "ns": clamp_pct(100 - s)}

        # Single number: default to WTF sliver unless it's extremely high
        x = nums[0]
        if x >= 90:
            return {"mnt": 0, "wtf": 0, "ns": x}
        return {"mnt": 0, "wtf": x, "ns": clamp_pct(100 - x)}

    # ---------- PyMuPDF: header-anchored, Y-locked extraction ----------
    try:
        import fitz  # PyMuPDF

        OPS = ["Addition", "Subtraction", "Multiplication", "Division"]
        ROW_TOL = 7.0  # vertical tolerance in pixels for matching % to a student's row

        def xcenter(s): return (s["x0"] + s["x1"]) / 2.0
        def ycenter(s): return (s["y0"] + s["y1"]) / 2.0

        doc = fitz.open(stream=file, filetype="pdf")
        rows = []

        name_pat = re.compile(r"^[A-Z][A-Za-z\-\.' ]+, ?[A-Z][A-Za-z\-\.' ]+$")

        for page in doc:
            spans = []
            for b in page.get_text("dict")["blocks"]:
                for l in b.get("lines", []):
                    for s in l.get("spans", []):
                        spans.append({
                            "text": s["text"].strip(),
                            "x0": s["bbox"][0], "y0": s["bbox"][1],
                            "x1": s["bbox"][2], "y1": s["bbox"][3],
                        })

            if not spans:
                continue

            # Header x-centers
            header_map = {}
            for sp in spans:
                t = sp["text"]
                for op in OPS:
                    if t.startswith(op):  # e.g., "Addition (+)"
                        header_map[op] = (sp["x0"] + sp["x1"]) / 2.0

            if len(header_map) < 2:
                continue  # try next page or fallback

            # Precompute % spans with centers
            pct_spans = []
            for sp in spans:
                if re.fullmatch(r"\d{1,3}%", sp["text"]):
                    pct_spans.append({
                        **sp,
                        "xc": (sp["x0"] + sp["x1"]) / 2.0,
                        "yc": (sp["y0"] + sp["y1"]) / 2.0,
                        "val": int(sp["text"].replace("%", "")),
                    })

            # For each student name span, collect row % by y-center proximity
            for sp in spans:
                t = sp["text"]
                if not t or not name_pat.match(t):
                    continue

                student = t
                y_mid = (sp["y0"] + sp["y1"]) / 2.0

                # Collect only % spans near this student's y
                row_pcts = [ps for ps in pct_spans if abs(ps["yc"] - y_mid) <= ROW_TOL]

                # Bucket by nearest header x
                buckets = {op: [] for op in OPS}
                for ps in row_pcts:
                    nearest = min(header_map.items(), key=lambda kv: abs(kv[1] - ps["xc"]))[0]
                    buckets[nearest].append(ps["val"])

                # Interpret each operation cell and emit rows
                for op in OPS:
                    # values are already tied to the correct column; sort to preserve left→right feel
                    vals = sorted(buckets[op])
                    cell = interpret_cell(vals)
                    rows.append({
                        "student": student,
                        "operation": op,
                        "ns": cell["ns"],
                        "wtf": cell["wtf"],
                        "mnt": cell["mnt"],
                    })

        if rows:
            df = pd.DataFrame(rows).drop_duplicates(subset=["student", "operation"])
            return df

    except Exception:
        pass

    # ---------- Fallback: pdfplumber text (best-effort only) ----------
    try:
        import pdfplumber
        rows = []
        with pdfplumber.open(io.BytesIO(file)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                name_re = r"[A-Z][A-Za-z\-\.' ]+, ?[A-Z][A-Za-z\-\.' ]+"
                line_re = re.compile(rf"^\s*({name_re}).*?(\d{{1,3}})\s*%(\D+(\d{{1,3}})\s*%)?(\D+(\d{{1,3}})\s*%)?")
                for line in text.splitlines():
                    m = line_re.search(line)
                    if not m:
                        continue
                    name = m.group(1).strip()
                    nums = [int(x) for x in re.findall(r"(\d{1,3})\s*%", line)]
                    c = interpret_cell(nums)
                    for op in ["Addition","Subtraction","Multiplication","Division"]:
                        rows.append({"student": name, "operation": op, "ns": c["ns"], "wtf": c["wtf"], "mnt": c["mnt"]})
        if rows:
            return pd.DataFrame(rows).drop_duplicates(subset=["student","operation"])
    except Exception:
        pass

    raise ValueError("Could not extract rows from this PDF. Use the CSV fallback for now.")

# ---------------- AI question with fallback ----------------
def offline_fallback_question(operation: str, grade: str, dok: str) -> str:
    if dok == "1":
        ex = {"Addition":"Add 23 + 41.","Subtraction":"Subtract 54 − 18.",
              "Multiplication":"Find 3 × 4.","Division":"Find 12 ÷ 3."}
        return ex.get(operation, "Solve the problem shown.")
    if dok == "2":
        ex = {"Addition":"Maria has 23 stickers and buys 41 more. How many now?",
              "Subtraction":"There are 54 apples; 18 are used for pies. How many left?",
              "Multiplication":"Draw an array to show 3 × 4. How many in all?",
              "Division":"Share 12 cookies equally among 3 friends. How many each?"}
        return ex.get(operation, "Solve the word problem.")
    ex = {"Addition":"Show two ways to make 64 using addition and explain your reasoning.",
          "Subtraction":"A number minus 18 equals 36. What could the number be? Explain.",
          "Multiplication":"If 8 × 7 = 56, how can you use that fact to find 8 × 9? Explain.",
          "Division":"Explain how you can use multiplication to check 63 ÷ 7."}
    return ex.get(operation, "Explain your strategy to solve the problem.")

def generate_group_question(operation: str, grade: str, dok: str) -> str:
    if not USE_AI:
        return ""
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("No OPENAI_API_KEY set")
        client = OpenAI(api_key=api_key)
        op_word = {"Addition":"addition","Subtraction":"subtraction",
                   "Multiplication":"multiplication","Division":"division"}[operation]
        prompt = ( "You are an elementary math teacher. "
                   f"Create exactly ONE grade {grade} {op_word} problem at DOK level {dok}. "
                   "Keep it concise and printable. Do not include the answer. Output only the problem." )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"You create short, age-appropriate math prompts."},
                      {"role":"user","content":prompt}],
            temperature=0.4, max_tokens=80,
        )
        return re.sub(r"\s+"," ", resp.choices[0].message.content.strip())
    except Exception:
        return offline_fallback_question(operation, grade, dok)

# ---------------- PDF export ----------------
def build_pdf(groups, operation, grade, dok, created_ts, focus_key=None, priority_group=None):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.7*inch, rightMargin=0.7*inch,
                            topMargin=0.7*inch, bottomMargin=0.7*inch)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Fact Fluency Grouping Report</b>", styles["Title"]))
    story.append(Paragraph(f"Operation: {operation} • Grade: {grade} • DOK: {dok} • Generated: {created_ts}", styles["Normal"]))
    story.append(Spacer(1, 0.25*inch))

    if priority_group:
        story.append(Paragraph("<b>Recommended Focus & Small Group</b>", styles["Heading2"]))
        story.append(Paragraph("Pulling students who are high WTF but low MNT (and not high NS) to convert practice into stable fluency.", styles["BodyText"]))
        data_p = [["Student","MNT %","WTF %","NS %"]] + [
            [r["name"], r["mnt"], r["wtf"], r["ns"]] for r in priority_group
        ]
        t_p = Table(data_p, colWidths=[3.2*inch, 0.9*inch, 0.9*inch, 0.9*inch])
        t_p.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
            ("GRID",(0,0),(-1,-1),0.25,colors.grey),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ]))
        story.append(t_p)
        story.append(Spacer(1, 0.28*inch))

    for g in ["NS","WTF","MNT"]:
        story.append(Paragraph(g, styles["Heading2"]))
        if g == "WTF":
            story.append(Paragraph("<i>Note: prioritizing high WTF + low MNT (low NS) for conceptual reteaching.</i>", styles["BodyText"]))
            story.append(Spacer(1, 0.08*inch))
        q = generate_group_question(operation, grade, dok)
        if q:
            story.append(Paragraph(f"<b>Sample Question:</b> {q}", styles["BodyText"]))
            story.append(Spacer(1, 0.12*inch))
        data = [["Student","MNT %","WTF %","NS %"]] + [
            [s["name"], s["mnt"], s["wtf"], s["ns"]] for s in groups[g]
        ]
        t = Table(data, colWidths=[3.2*inch, 0.9*inch, 0.9*inch, 0.9*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
            ("GRID",(0,0),(-1,-1),0.25,colors.grey),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.26*inch))

    doc.build(story)
    buf.seek(0)
    return buf

# ---------------- Main app ----------------
df = None
if uploaded:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = parse_csv_to_dataframe(uploaded.read())
        else:
            df = parse_pdf_to_dataframe(uploaded.read())
    except Exception as e:
        st.error(f"Parse error: {e}")
        st.info("Tip: Use a CSV with columns: student, operation, ns, wtf, mnt.")

if df is not None and not df.empty:
    df_op = df[df["operation"] == OPERATION]
    if df_op.empty:
        st.warning(f"No rows found for “{OPERATION}”.")
    else:
        df_op = df_op.copy()
        df_op["group"] = df_op.apply(lambda r: assign_group(r["ns"], r["wtf"], r["mnt"]), axis=1)

        st.subheader("Preview (MNT | WTF | NS)")
        for _, r in df_op.sort_values("student").iterrows():
            st.text(f"{r['student']}  ·  MNT {r['mnt']} | WTF {r['wtf']} | NS {r['ns']}")

        groups = {"NS": [], "WTF": [], "MNT": []}
        for _, r in df_op.sort_values("student").iterrows():
            groups[r["group"]].append({"name": r["student"], "ns": int(r["ns"]), "wtf": int(r["wtf"]), "mnt": int(r["mnt"])})

        focus_key = recommend_focus(df_op)
        priority_group = select_priority_students(df_op, focus_key)
        st.success(f"Focus Group: {focus_key}  •  Suggested size: {choose_group_size(len(df_op))}")
        if priority_group:
            st.info("Recommended Small Group (high WTF + low MNT, low NS)")
            for r in priority_group:
                st.text(f"{r['name']}  ·  MNT {r['mnt']} | WTF {r['wtf']} | NS {r['ns']}")

        created_ts = datetime.now().strftime("%b %d, %Y %I:%M %p")
        pdf_buf = build_pdf(groups, OPERATION, GRADE, DOK, created_ts,
                            focus_key=focus_key, priority_group=priority_group)
        st.download_button("Download PDF (printable packet)",
                           data=pdf_buf.read(),
                           file_name=f"fact_fluency_{OPERATION.lower()}_{datetime.now().strftime('%Y%m%d')}.pdf",
                           mime="application/pdf")
else:
    st.info("Upload a PDF (or CSV) to get started.")
