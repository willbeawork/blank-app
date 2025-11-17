import streamlit as st
import pandas as pd
import re
from io import StringIO

st.set_page_config(page_title="FH -> Markdown", layout="wide")
st.title("FH table→ Markdown Formatted Summary")
st.markdown("Paste tab-separated family-history data, edit the table, and generate a markdown summary.")

# --- Initialise session_state for selection ---
if "selected_option" not in st.session_state:
    st.session_state.selected_option = "Paste Raw Data"   # Default pre-selected

def select_option(option):
    st.session_state.selected_option = option

left, middle, right = st.columns(3)

with left:
    if st.button("Paste Raw Data",width="stretch"):
        select_option("Paste Raw Data")

with middle:
    if st.button("CSV/XLSX File",width="stretch"):
        select_option("CSV/XLSX File")

with right:
    if st.button("Manual Input",width="stretch"):
        select_option("Manual Input")

st.write(f"🔍 **Current Input Mode:** {st.session_state.selected_option}")

# -----------------------
# Helper: parse raw pasted data (multiline merging logic you provided)
# -----------------------
def parse_raw_data(raw_data: str) -> pd.DataFrame:
    if not raw_data or not raw_data.strip():
        return pd.DataFrame()

    # Clean and split lines
    lines = [line.strip() for line in raw_data.strip().split('\n') if line.strip()]
    if not lines:
        return pd.DataFrame()

    header = lines[0].split('\t')
    data_lines = lines[1:]

    # Combine multiline records
    records = []
    temp = []
    for line in data_lines:
        if line.count('\t') >= 2:  # new full row
            if temp:
                records.append(temp)
            temp = [line]
        else:
            temp.append(line)
    if temp:
        records.append(temp)

    processed = []
    for record in records:
        full_line = ' '.join(record).replace('\t', '\t', 1)
        parts = full_line.split('\t')

        # Ensure same length as header
        while len(parts) < len(header):
            parts.append('')
        processed.append(dict(zip(header, parts)))

    return pd.DataFrame(processed)

# -----------------------
# UI: input area + optional file upload
# -----------------------
with st.expander("How to use"):
    st.write(
        """
        - Paste tab-separated text (including header row) into the box below, or upload a CSV/TSV/Excel file.
        - Edit the table directly using the editor.
        - Click **Generate Markdown** to create the summary.
        """
    )

col1, col2 = st.columns([2, 1])

with col1:
    raw_data = st.text_area(
        "Paste raw table data here (tab-separated). Include header row. Example header:",
        value="Relationship\tFirst Name\tDiagnosis (laterality, hormones, subtype)@age\tConfirmed/Not confirmed/Abroad",
        height=200,
        help="You can copy table rows from Excel/Google Sheets then paste here."
    )

with col2:
    uploaded = st.file_uploader("Or upload CSV / TSV / XLSX", type=["csv", "tsv", "xlsx", "xls"])
    st.write("Preview / examples")
    st.write("- Header names must include the diagnosis column name shown above.")

# If uploaded file present, read it and override raw_data
if uploaded:
    try:
        if uploaded.name.endswith((".xls", ".xlsx")):
            df_uploaded = pd.read_excel(uploaded)
        else:
            try:
                df_uploaded = pd.read_csv(uploaded)
            except Exception:
                uploaded.seek(0)
                df_uploaded = pd.read_csv(uploaded, sep='\t')
        df = df_uploaded.copy()
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        df = pd.DataFrame()
else:
    # Parse pasted raw_data
    if raw_data and raw_data.strip():
        df = parse_raw_data(raw_data)
    else:
        # Empty default
        df = pd.DataFrame(columns=[
            "Relationship",
            "First Name",
            "Diagnosis (laterality, hormones, subtype)@age",
            "Confirmed/Not confirmed/Abroad"
        ])

# Normalize columns if slightly different names (common variations)
expected_diag_col = "Diagnosis (laterality, hormones, subtype)@age"
alt_diag_cols = [c for c in df.columns if "diagnos" in c.lower() and "age" in c.lower()]

if expected_diag_col not in df.columns and alt_diag_cols:
    df = df.rename(columns={alt_diag_cols[0]: expected_diag_col})

# Ensure columns exist
for col in ["Relationship", "First Name", expected_diag_col, "Confirmed/Not confirmed/Abroad"]:
    if col not in df.columns:
        df[col] = ""

# Show editable table
st.subheader("Review / Edit Table")
st.markdown("WARNING: Any information in brackets will be removed from the diagnosis section")
edited_df = st.data_editor(df, use_container_width=True)
# -----------------------
# Text transformation logic (your provided functions, adjusted)
# -----------------------
word_dict = {
    "maternal" : ["Mat", "mat"],
    "paternal" : ["Pat", "pat"],
    "sibling" : ["Sib", "sib"],
    "father" : ["Dad", "dad"],
    "mother" : ["Mum", "mum"],
    "sister" : ["Sis", "sis"],
    "brother" : ["Bro", "bro"],
    "aunt" : ["Aunt", "aunt"],
    "uncle" : ["Uncle", "unc", "Unc"],
    "niece" : ["Niece", "niece"],
    "nephew" : ["Nephew", "nephew"],
    "cousin" : ["Cous", "Cous,", "cous", "cous,"],
    "maternal grandmother": ["MGM", "MGM,"],
    "paternal grandmother" : ["PGM", "PGM,"],
    "paternal grandfather" : ["PGF" , "PGF,"],
    "maternal grandfather" : ["MGF" , "MGF,"],
    "grandfather" : ["GF"],
    "grandmother" : ["GM"],
    "maternal great grandfather" : ["MGGF"],
    "paternal great grandfather" : ["PGGF"],
    "paternal great grandmother" : ["PGGM"],
    "maternal great grandmother" : ["MGGM"]
}

second_dict = {
    "Your father" : "Father",
    "Your mother" : "Mother",
    "Your brother" : "Brother",
    "Your sister" : "Sister",
    "Your paternal" : "paternal",
    "Your maternal" : "maternal",
    "You" : "Patient"
}

def text_change(text, word_dict):
    if pd.isna(text):
        return ""
    for update, originals in word_dict.items():
        for original in originals:
            # skip certain replacements if part of a longer word
            if original.lower() in ["sis", "mat", "pat", "bro","gm","gf","GM","GF"]:
                # only match if the word is standalone (not part of sister/maternal etc)
                pattern = rf'(?<!\w){re.escape(original)}(?![a-zA-Z])'
            else:
                pattern = rf'\b{re.escape(original)}\b'
            text = re.sub(pattern, update, text)
    return text

def second_change(incomplete_text, second_dict):
    if pd.isna(incomplete_text):
        return ""
    for new, old in second_dict.items():
        incomplete_text = incomplete_text.replace(old, new)
    return incomplete_text

def convert_at_symbols(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # Replace @? with 'at an unknown age'
    text = re.sub(r'@\?', ' at an unknown age', text)
    # Replace @ followed by a decade (e.g. 50s) with 'in their 50s'
    text = re.sub(r'@(\d{2})s', r' in their \1s', text)
    # Replace @ followed by a number (age) with ' aged 45'
    text = re.sub(r'@(\d{1,3})(?!\d|s)', r' aged \1', text)
    return text

def clean_diagnosis(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    # change crc to colorectal
    text = re.sub(r'\bcrc\b', 'colorectal', text)
    # exceptions where we shouldn't append 'cancer'
    exceptions = ['leukaemia', 'lymphoma', 'melanoma', 'multiple myeloma','polyps']
    if any(exc in text for exc in exceptions):
        return text
    # split before age info (which starts with patterns we've used)
    parts = re.split(r'(@\?|@\d{1,3}s?| aged \d{1,3}| in their \d{2}s| at an unknown age)', text)
    if len(parts) > 1:
        return parts[0].strip() + ' cancer ' + ''.join(parts[1:]).strip()
    else:
        # If no age info, just add ' cancer' to the end (unless already contains 'cancer')
        if 'cancer' in text:
            return text
        return text + ' cancer'

# Apply conversions to the edited dataframe copy
proc_df = edited_df.copy()
proc_df = proc_df.fillna("")

# Relationships transformation
proc_df['Relationship_clean'] = proc_df['Relationship'].apply(lambda x: text_change(str(x), word_dict))
proc_df['Relationship_clean'] = proc_df['Relationship_clean'].apply(lambda x: second_change(x, second_dict))

# Diagnosis transformation
proc_df[expected_diag_col] = proc_df[expected_diag_col].astype(str)
proc_df[expected_diag_col] = proc_df[expected_diag_col].apply(convert_at_symbols)
proc_df[expected_diag_col] = proc_df[expected_diag_col].apply(clean_diagnosis)
# remove bracketed text
proc_df[expected_diag_col] = proc_df[expected_diag_col].str.replace(r'\(.*?\)', '', regex=True).str.strip()

# -----------------------
# Build the markdown summary lines
# -----------------------
def make_phrase(confirmation_status: str) -> str:
    s = str(confirmation_status).lower()
    if 'not confirmed' in s:
        return 'was reportedly diagnosed with'
    elif 'unconfirmed' in s:
         return 'was reportedly diagnosed with'
    elif 'confirmed' in s:
        return 'was diagnosed with'
    else:
        return 'was reportedly diagnosed with'

def build_line(row) -> str:
    rel = str(row.get('Relationship_clean','')).strip()
    first = str(row.get('First Name','')).strip()
    diag = str(row.get(expected_diag_col,'')).strip()
    conf = str(row.get('Confirmed/Not confirmed/Abroad','')).strip()
    phrase = make_phrase(conf)
    # Format: Relationship, Firstname, phrase diagnosis - status
    if first and first.lower() not in ['nan','none','']:
        return f"{rel}, {first}, {phrase} {diag} - {conf}"
    else:
        return f"{rel}, {phrase} {diag} - {conf}"

def no_name_build_line(row) -> str:
    rel = str(row.get('Relationship_clean', '')).strip()
    diag = str(row.get(expected_diag_col, '')).strip()
    conf = str(row.get('Confirmed/Not confirmed/Abroad', '')).strip()
    phrase = make_phrase(conf)
    # Format: - Relationship phrase diagnosis - status (no first name)
    return f"{rel} {phrase} {diag} - *{conf}*"

def no_confs_build_line(row) -> str:
    rel = str(row.get('Relationship_clean', '')).strip()
    first = str(row.get('First Name','')).strip()
    diag = str(row.get(expected_diag_col, '')).strip()
    phrase = "was diagnosed with"
    # Format: - Relationship, phrase diagnosis (no conf info)
    return f"{rel}, {first}, {phrase} {diag}"

def no_confs_and_no_name_build_line(row) -> str:
    rel = str(row.get('Relationship_clean', '')).strip()
    diag = str(row.get(expected_diag_col, '')).strip()
    phrase = "was diagnosed with"
    # Format: - Relationship default phrase diagnosis (no first name or confs)
    return f"{rel} {phrase} {diag}"


# Generate markdown lines
proc_df['summary_line'] = proc_df.apply(build_line, axis=1)
markdown_output = "\n\n".join([ln for ln in proc_df['summary_line'] if ln.strip()])

proc_df['summary_line_no_confs'] = proc_df.apply(no_confs_build_line, axis=1)
noconfs_markdown_output = "\n\n".join([ln for ln in proc_df['summary_line_no_confs'] if ln.strip()])

proc_df['summary_line_no_name'] = proc_df.apply(no_name_build_line, axis=1)
noname_markdown_output = "\n\n".join([ln for ln in proc_df['summary_line_no_name'] if ln.strip()])

proc_df['no_confs_and_no_name_build_line'] = proc_df.apply(no_confs_and_no_name_build_line, axis=1)
no_confs_and_no_name_markdown_output = "\n\n".join([ln for ln in proc_df['no_confs_and_no_name_build_line'] if ln.strip()])

# -----------------------
# Output UI: show markdown and allow download
# -----------------------
st.subheader("Generated Markdown Summary")

remove_first_name = st.toggle("Remove First Name", value=False)
remove_conf_info = st.toggle("Remove Confirmation Information", value=False)

# Select the correct markdown output
if remove_first_name and remove_conf_info:
    selected_markdown = no_confs_and_no_name_markdown_output
elif remove_first_name:
    selected_markdown = noname_markdown_output
elif remove_conf_info:
    selected_markdown = noconfs_markdown_output
else:
    selected_markdown = markdown_output

# --- Output text field ---
st.text_area(
    "Generated Output",
    value=selected_markdown,
    height=300,
    disabled=True)

# Small preview of the processed columns for debugging
with st.expander("Processed columns preview (debug)"):
    st.dataframe(proc_df[[
        "Relationship", "Relationship_clean", "First Name", expected_diag_col, "Confirmed/Not confirmed/Abroad"
    ]].head(50), use_container_width=True)
    
