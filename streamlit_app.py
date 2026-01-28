import streamlit as st
import pandas as pd
import re
from io import StringIO
from st_copy import copy_button

# --- Initialise session_state for selection ---
if "selected_option" not in st.session_state:
    st.session_state.selected_option = "Paste Raw Data"   # Default pre-selected

def select_option(option):
    st.session_state.selected_option = option

st.set_page_config(page_title="FH -> Markdown", layout="wide")
st.title("FH table -> Formatted Summary")
st.write(f"Developed for use by cancer genetics clinicans")
st.write(f"🔍 **Current Input Mode:** {st.session_state.selected_option}")

#Make an empty df for manual input mode
if "manual_df" not in st.session_state:
    # Initialize empty editable DataFrame for Manual Input
    st.session_state.manual_df = pd.DataFrame(columns=[
        "Relationship",
        "First Name",
        "Diagnosis (laterality, hormones, subtype)@age",
        "Confirmed/Not confirmed/Abroad"
    ])

#Define a function to 'Parse' (Reformat pasted data)
def parse_raw_data(raw_data: str) -> pd.DataFrame:
    if not raw_data or not raw_data.strip():
        return pd.DataFrame()

    lines = [line.strip() for line in raw_data.strip().split('\n') if line.strip()]
    if not lines:
        return pd.DataFrame()

    header = lines[0].split("\t")
    data_lines = lines[1:]

    records = []
    temp = []

    # combine multiline rows
    for line in data_lines:
        if line.count("\t") >= 2:
            if temp:
                records.append(temp)
            temp = [line]
        else:
            temp.append(line)
    if temp:
        records.append(temp)

    processed = []
    for record in records:
        full_line = " ".join(record)
        parts = full_line.split("\t")

        while len(parts) < len(header):
            parts.append("")
        processed.append(dict(zip(header, parts)))

    return pd.DataFrame(processed)

def select_option(option):
    st.session_state.selected_option = option

left, middle, right = st.columns(3)

with left:
    if st.button("*Paste Raw Data*",width="stretch"):
        select_option("Paste Raw Data")

with middle:
    if st.button("*CSV/XLSX File*",width="stretch"):
        select_option("CSV/XLSX File")

with right:
    if st.button("*Manual Input*",width="stretch"):
        select_option("Manual Input")

# -----------------------
# Define session states and respective input UIs
# -----------------------


# --- PASTE RAW DATA ---

if st.session_state.selected_option == "Paste Raw Data":
    st.subheader("Paste raw table data here")
    with st.expander("How to use"):
        st.write(
            """
            - Paste tab-separated text (including header row) into the box below, or upload a CSV/TSV/Excel file.
            - Edit the table directly using the editor.
            - Click **Generate Markdown** to create the summary.
            """
        )
    with st.form("raw_data_form"):
        st.write("Include header row")

        raw_data = st.text_area(
            label="",
            value="Relationship\tFirst Name\tDiagnosis (laterality, hormones, subtype)@age\tConfirmed/Not confirmed/Abroad",
            height=200,
            help="You can copy table rows from Excel/Google Sheets then paste here."
)

        # Submit button (also triggered by CTRL+Enter)
        submitted = st.form_submit_button("Submit")

    # Handle submission
    if submitted:
        st.session_state.raw_submitted = raw_data
        st.session_state.raw_df = parse_raw_data(raw_data)


# --- FILE UPLOAD ---
if st.session_state.selected_option == "CSV/XLSX File":  
    st.write("- Header names must be: " )
    st.write("Relationship | First Name | Diagnosis (additional info)@age | Confirmed/Not confirmed/Abroad")
    uploaded = st.file_uploader("Or upload CSV / TSV / XLSX", type=["csv", "tsv", "xlsx", "xls"])
    st.write("Preview / examples")

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

# --- MANUAL INPUT ---
if st.session_state.selected_option == "Manual Input":
    st.subheader("Input & Edit Table")
    st.markdown("⚠️WARNING: Any information in brackets will be removed from the diagnosis section")
    # Show editable DataFrame
    edited_df = st.data_editor(
        st.session_state.manual_df,
        use_container_width=True,
        num_rows="dynamic",   # allows adding new rows
        hide_index=True       # optional: hides the index column
    )
    # --- Example table in an expander ---
    with st.expander("***Example Table (Read-only)***"):
        example_data = pd.DataFrame([
        {
            "Relationship": "Father",
            "First Name": "Dennis",
            "Diagnosis (laterality, hormones, subtype)@age": "Prostate@56 (Adenocarcinoma, Confined to prostate)",
            "Confirmed/Not confirmed/Abroad": "Confirmed"
        },
        {
            "Relationship": "Paternal Aunt",
            "First Name": "Alice",
            "Diagnosis (laterality, hormones, subtype)@age": "Breast@60 (Grade 1, ER+)",
            "Confirmed/Not confirmed/Abroad": "Confirmed, In Australia"
        }
        ])

        # Style example table: italics + grey background
        def style_example(row):
            return ['font-style: italic; background-color: #f0f0f0'] * len(row)


        st.dataframe(example_data.style.apply(style_example, axis=1), use_container_width=True)

# Normalize columns if slightly different names (common variations)
# FIX: Ensure df always exists depending on mode
if st.session_state.selected_option == "Paste Raw Data":
    df = parse_raw_data(raw_data)

elif st.session_state.selected_option == "CSV/XLSX File":
    if uploaded:
        df = df_uploaded.copy()
    else:
        df = pd.DataFrame()

elif st.session_state.selected_option == "Manual Input":
    df = edited_df.copy()

expected_diag_col = "Diagnosis (laterality, hormones, subtype)@age"
alt_diag_cols = [c for c in df.columns if "diagnos" in c.lower() and "age" in c.lower()]

if expected_diag_col not in df.columns and alt_diag_cols:
    df = df.rename(columns={alt_diag_cols[0]: expected_diag_col})

# Ensure columns exist
required_cols = ["Relationship", "First Name", expected_diag_col,
                 "Confirmed/Not confirmed/Abroad"]

for col in required_cols:
    if col not in df.columns:
        df[col] = ""


# Show editable table for raw data and file upload inputs but not manual input
if st.session_state.selected_option in ["Paste Raw Data", "CSV/XLSX File"]:
    st.subheader("Review / Edit Table")
    st.markdown("WARNING: Any information in brackets will be removed from the diagnosis section")
    edited_df = st.data_editor(df, use_container_width=True)

# -----------------------
# Text transformation logic (your provided functions, adjusted)
# -----------------------
word_dict = {
    "maternal": ["mat", "mat"],
    "maternal " : ["M.", "m.","m. "],
    "paternal": ["pat", "pat"],
    "paternal " : ["P.", "p.", "p. "],
    "sibling": ["sib", "sib"],
    "father": ["dad", "dad"],
    "mother": ["mum", "mum"],
    "sister": ["sis", "sis"],
    "brother": ["bro", "bro"],
    "aunt": ["aunt", "aunt"],
    "uncle": ["uncle", "unc", "unc"],
    "niece": ["niece", "niece"],
    "nephew": ["nephew", "nephew"],
    "cousin": ["cous", "cous,", "cous", "cous,"],
    "maternal grandmother": ["mgm", "mgm,"],
    "paternal grandmother": ["pgm", "pgm,"],
    "paternal grandfather": ["pgf", "pgf,"],
    "maternal grandfather": ["mgf", "mgf,"],
    "grandfather": ["gf"],
    "grandmother": ["gm"],
    "great grandfather": ["ggf"],
    "great grandmother": ["ggm"],
    "maternal great grandfather": ["mggf"],
    "paternal great grandfather": ["pggf"],
    "paternal great grandmother": ["pggm"],
    "maternal great grandmother": ["mggm"],
    "":["my"]
}

second_dict = {
    "Your father": "father",
    "Your mother": "mother",
    "Your brother": "brother",
    "Your sister": "sister",
    "Your paternal": "paternal",
    "Your maternal": "maternal",
    "You": ["you","patient","me"]
}

def text_change(text, word_dict):
    text = str(text).lower()
    if pd.isna(text):
        return ""
    for update, originals in word_dict.items():
        for original in originals:
            # skip certain replacements if part of a longer word
            if original.lower() in ["sis", "mat", "pat", "bro","gm","gf"]:
                # only match if the word is standalone (not part of sister/maternal etc)
                pattern = rf'(?<!\w){re.escape(original)}(?![a-zA-Z])'
            else:
                pattern = rf'\b{re.escape(original)}\b'
            text = re.sub(pattern, update, text)
    return text

def second_change(incomplete_text, second_dict):
    if pd.isna(incomplete_text):
        return ""
    text = str(incomplete_text)
    for new, old in second_dict.items():
        if isinstance(old, list):
            for o in old:
                pattern = rf'\b{o}\b'
                text = re.sub(pattern, new, text, flags=re.IGNORECASE)
        else:
            pattern = rf'\b{old}\b'
            text = re.sub(pattern, new, text, flags=re.IGNORECASE)
    return text

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


def clean_single_dx(text):
    """Cleans one *single* diagnosis."""
    text = text.lower().strip()
    text = convert_at_symbols(text)
    # change crc to colorectal
    text = re.sub(r'\bcrc\b', 'colorectal', text)
    # exceptions where we shouldn't append 'cancer'
    exceptions = [
        'leukaemia', 'lymphoma', 'melanoma', 'multiple myeloma',
        'polyps', 'cancer of', 'dcis', 'lcis', 'brain tumour']
    if any(exc in text for exc in exceptions):
        return re.sub(r'\bcancer(?:\s+cancer)+', 'cancer', text).strip()
    # split of age info
    parts = re.split(r'(@\?|@\d{1,3}s?| aged \d{1,3}| in their \d{2}s| at an unknown age)', text)
    if len(parts) > 1:
        dx = parts[0].strip()
        if 'cancer' not in dx:
            dx += ' cancer'
        result = dx + ' ' + ''.join(parts[1:]).strip()
    else:
        if 'cancer' in text:
            result = text
        else:
            result = text + ' cancer'
    # remove duplicate cancer
    result = re.sub(r'\bcancer(?:\s+cancer)+', 'cancer', result)
    return result.strip()


def clean_diagnosis(text):
    """Handles multiple diagnoses by splitting after the first space following each '@...'."""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    # If no @ present → simple path
    if '@' not in text:
        return clean_single_dx(text)
    parts = []
    start = 0
    # Pattern to match @?, @55, @70s, etc.
    age_pattern = re.compile(r'@( \?|\d{1,3}s?|\?)')
    # Find each @ and follow until the next space
    for match in re.finditer(r'@', text):
        # Find the end of the age info: @55, @?, @70s…
        age_match = re.match(r'@\d{1,3}s?|@\?', text[match.start():])
        if age_match:
            age_end = match.start() + len(age_match.group())
        else:
            age_end = match.start() + 1  # fallback
        # Find first space AFTER age block
        next_space = text.find(' ', age_end)
        if next_space != -1:
            # Add text from previous start to this split point
            parts.append(text[start:next_space].strip())
            start = next_space + 1
    # Add final piece
    parts.append(text[start:].strip())
    # Clean each diagnosis
    cleaned = [clean_single_dx(p) for p in parts if p]
    return " and ".join(cleaned)

def parse_confirmation(text: str) -> str:
    """
    Standardize user-entered confirmation status.
    
    Returns one of:
        - 'Confirmed'
        - 'Not confirmed'
        - 'Unconfirmed'
    or flags with 🚩Review🚩 if invalid.
    
    Handles optional trailing spaces or full stops.
    """
    if pd.isna(text) or str(text).strip() == "":
        return ""
    
    # Remove leading/trailing whitespace and trailing period
    cleaned_text = str(text).strip().rstrip(".").lower()
    
    # Standardize the three allowed options
    if cleaned_text == "confirmed":
        return "Confirmed"
    elif cleaned_text == "not confirmed":
        return "Not confirmed"
    elif cleaned_text == "unconfirmed":
        return "Unconfirmed"
    else:
        # Flag invalid entries
        return f"{text} 🚩Review🚩"
    
# Apply conversions to the edited dataframe copy
proc_df = edited_df.copy()
proc_df = proc_df.fillna("")

# Relationships transformation
proc_df['Relationship_clean'] = proc_df['Relationship'].apply(lambda x: text_change(str(x), word_dict))
proc_df['Relationship_clean'] = proc_df['Relationship_clean'].apply(lambda x: second_change(x, second_dict))

# Diagnosis transformation
proc_df[expected_diag_col] = proc_df[expected_diag_col].astype(str)
proc_df[expected_diag_col] = proc_df[expected_diag_col].apply(clean_diagnosis)
proc_df[expected_diag_col] = proc_df[expected_diag_col].apply(convert_at_symbols)
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
    conf = parse_confirmation(row.get('Confirmed/Not confirmed/Abroad',''))
    phrase = make_phrase(conf)
    if rel.lower() == 'you':
        return f"{rel} were diagnosed with {diag}- *{conf}*"
    # Format: Relationship, Firstname, phrase diagnosis - status
    if first and first.lower() not in ['nan','none','']:
        return f"{rel}, {first}, {phrase} {diag} - *{conf}*"
    else:
        return f"{rel}, {phrase} {diag} - *{conf}*"

def no_name_build_line(row) -> str:
    rel = str(row.get('Relationship_clean', '')).strip()
    diag = str(row.get(expected_diag_col, '')).strip()
    conf = str(row.get('Confirmed/Not confirmed/Abroad', '')).strip()
    phrase = make_phrase(conf)
    if rel.lower() == 'you':
        return f"{rel} were diagnosed with {diag}- *{conf}*"
    # Format: - Relationship phrase diagnosis - status (no first name)
    return f"{rel} {phrase} {diag} - *{conf}*"

def no_confs_build_line(row) -> str:
    rel = str(row.get('Relationship_clean', '')).strip()
    first = str(row.get('First Name','')).strip()
    diag = str(row.get(expected_diag_col, '')).strip()
    phrase = "was diagnosed with"
    # Format: - Relationship, phrase diagnosis (no conf info)
    if rel.lower() == 'you':
        return f"{rel} were diagnosed with {diag}"
    # Format: Relationship, Firstname, phrase diagnosis - status
    if first and first.lower() not in ['nan','none','']:
        return f"{rel}, {first}, {phrase} {diag}"
    else:
        return f"{rel}, {phrase} {diag}"

def no_confs_and_no_name_build_line(row) -> str:
    rel = str(row.get('Relationship_clean', '')).strip()
    diag = str(row.get(expected_diag_col, '')).strip()
    phrase = "was diagnosed with"
    if rel.lower() == 'you':
        return f"{rel} were diagnosed with {diag}*"
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

# -----------------------
# Create container style
# -----------------------

css = """
<style>
    .st-key-styled_container{
    background-color:lightgray;
    padding:2rem;
    }

 .st-key-styled_container div[data-testid="stText"] div{
    color:black;

}
</style>"""
st.html(css)

# -----------------------
# Create output
# -----------------------

with st.container(key="styled_container"):
    st.markdown(selected_markdown)

copy_button(
    (selected_markdown),
    tooltip="Copy this text",
    copied_label="Copied!",
    icon="st",
)

# Small preview of the processed columns for debugging
with st.expander("Processed columns preview (debug)"):
    st.dataframe(proc_df[[
        "Relationship", "Relationship_clean", "First Name", expected_diag_col, "Confirmed/Not confirmed/Abroad"
    ]].head(50), use_container_width=True)
    
