import json
import base64
import pandas as pd
import pytesseract
import os
import tempfile
import traceback
import re
import logging

from pdf2image import convert_from_path
from pathlib import Path
from typing import Dict
from pytesseract import Output
from openpyxl import load_workbook

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    Lambda handler with Function URL support
    Processes multiple documents and returns merged CSV
    """

    try:
        logger.info("=== Lambda Invocation Started ===")
        logger.info(f"Event keys: {list(event.keys())}")

        # Parse request body
        if 'body' in event:
            logger.info("Parsing body from event")
            if event.get('isBase64Encoded', False):
                body = base64.b64decode(event['body']).decode('utf-8')
            else:
                body = event['body']
            request_data = json.loads(body)
        else:
            logger.info("Using event directly as request_data")
            request_data = event

        # Handle multiple files
        files = request_data.get('files', [])
        if not files:
            # Backward compatibility for single file
            file_content = request_data.get('file_content')
            file_name = request_data.get('file_name', 'document.pdf')
            file_type = request_data.get('file_type', 'unknown')
            if file_content:
                files = [{
                    'file_type': file_type,
                    'file_name': file_name,
                    'file_format': file_name.split('.')[-1].lower(),
                    'file_data': file_content
                }]

        if not files:
            raise ValueError("files array is required")

        logger.info(f"Processing {len(files)} files")
        for f in files:
            logger.info(f"  - {f.get('file_type')}: {f.get('file_name')}")
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        processed_data = {}
        
        # Process each file
        for file_info in files:
            file_type = file_info.get('file_type')
            file_name = file_info.get('file_name')
            file_format = file_info.get('file_format')
            file_data = file_info.get('file_data')
            
            if not all([file_type, file_name, file_format, file_data]):
                print(f"Skipping incomplete file info: {file_info}")
                continue
                
            print(f"Processing {file_type}: {file_name}")
            
            # Decode and save file
            decoded_data = base64.b64decode(file_data)
            input_file = os.path.join(temp_dir, file_name)
            
            with open(input_file, 'wb') as f:
                f.write(decoded_data)
            
            # Process based on file type and format
            try:
                logger.info(f"Processing {file_type} ({file_format})...")
                result_df = process_file_by_type(input_file, file_type, file_format)
                processed_data[file_type] = result_df
                logger.info(f"Processed {file_type}: {len(result_df)} records, {len(result_df.columns)} columns")
                logger.info(f"   Columns: {list(result_df.columns)[:5]}...")  # Log first 5 columns
            except Exception as e:
                logger.error(f"Error processing {file_type}: {e}")
                logger.error(traceback.format_exc())
                processed_data[file_type] = pd.DataFrame({'error': [str(e)]})

            # Cleanup individual file
            os.remove(input_file)

        logger.info(f"Processed data summary: {list(processed_data.keys())}")

        # Merge and map data based on your original logic
        logger.info("Starting merge and mapping...")
        final_result = merge_and_map_data(processed_data)
        if final_result is not None and not final_result.empty:
            logger.info(f"Final result: {len(final_result)} rows, {len(final_result.columns)} columns")
        else:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({
                    'success': False,
                    'error': 'Missing key data for processing.'
                })
            }
        
        # Generate CSV
        csv_output = final_result.to_csv(index=False, encoding='utf-8')
        csv_base64 = base64.b64encode(csv_output.encode('utf-8')).decode('utf-8')
        
        # Cleanup temp directory
        os.rmdir(temp_dir)

        response_body = {
            'success': True,
            'csv_content': csv_base64,
            'file_name': "danlon_processed_output.csv",
            'records_processed': len(final_result),
            'files_processed': len(files),
            'processing_summary': {k: len(v) for k, v in processed_data.items()}
        }

        logger.info(f"=== Processing Complete ===")
        logger.info(f"Records processed: {len(final_result)}")
        logger.info(f"Files processed: {len(files)}")
        logger.info(f"Summary: {response_body['processing_summary']}")

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps(response_body)
        }

    except Exception as e:
        logger.error(f"=== Error Occurred ===")
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())

        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'success': False,
                'error': str(e)
            })
        }
    
def extract_table_from_section(
    image,
    columns_dict: dict | list,
    y_threshold: int = 10,
    lang: str = "dan",
    optional: bool = False
) -> pd.DataFrame | None:
    data = pytesseract.image_to_data(image, lang=lang, output_type=Output.DATAFRAME, config="--psm 4")

    if optional:
        lines = data["text"].dropna().tolist()

        if any(any(col in line for col in columns_dict.keys()) for line in lines):
            print("Columns found for optional section")
        else:
            print("Column not found for optional section")
            return None

    data = data[data['conf'] > 0]
    data = data.dropna(subset=['text'])
    data["x_center"] = data["left"] + data["width"] / 2
    data["y_center"] = data["top"] + data["height"] / 2

    if type(columns_dict) == list:
        if "Saldo" in data['text'].values:
            columns_dict = columns_dict[0]
        else:
            columns_dict = columns_dict[1]

    # Assign each word to a column based on its horizontal x position
    def assign_column(x):
        for col, (xmin, xmax) in columns_dict.items():
            if xmin <= x <= xmax:
                return col
        return None
    
    data["column"] = data["x_center"].apply(assign_column)
    data = data.sort_values(by="y_center").reset_index(drop=True)

    rows = []
    current_row_y = None
    current_row = {col: [] for col in columns_dict}

    for _, w in data.iterrows():
        if w["text"] in columns_dict.keys():  # skip header row
            continue

        if current_row_y is None:
            current_row_y = w["y_center"]

        # Detect new row
        if abs(w["y_center"] - current_row_y) > y_threshold:
            # Merge words within each column (sorted left-to-right)
            sorted_row = {}
            for col, words in current_row.items():
                words_sorted = sorted(words, key=lambda x: x["left"])
                sorted_row[col] = " ".join([x["text"] for x in words_sorted])
            rows.append(sorted_row)

            current_row = {col: [] for col in columns_dict}
            current_row_y = w["y_center"]

        col = w["column"]
        if col:
            current_row[col].append(w)

    # Append last row
    sorted_row = {}
    for col, words in current_row.items():
        words_sorted = sorted(words, key=lambda x: x["left"])
        sorted_row[col] = " ".join([x["text"] for x in words_sorted])
    rows.append(sorted_row)

    df = pd.DataFrame(rows)
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    df = df.dropna(how="all").reset_index(drop=True)

    # drop row if first col no val
    first_col = df.columns[0]
    df = df[df[first_col].astype(bool)]
    df = df.reset_index(drop=True)

    return df

# 
def process_file_by_type(file_path, file_type, file_format):
    """
    Route to appropriate processor based on file type
    """
    if file_type == "employee_payslip" and file_format == "pdf":
        return process_payslip(file_path, file_format)
    elif file_type == "employee_active" and file_format == "xlsx": # excel file only for now
        return pd.read_excel(file_path)
    elif file_type == "employee_holiday" and file_format == "xlsx": # excel file only for now
        return pd.read_excel(file_path)
    elif file_type == "employee_general" and file_format == "pdf":
        return process_employee_general(file_path, file_format)
    elif file_type == "employee_list" and file_format == "pdf":
        return process_employee_list(file_path, file_format)
    else:
        raise ValueError(f"Unsupported file type: {file_type} with format: {file_format}")
    
def extract_column_from_df(df: pd.DataFrame) -> Dict:
    result = {}

    if df.empty or df.shape[1] < 2:
        return result  

    main_col = df.columns[0]
    other_cols = df.columns[1:]

    for _, row in df.iterrows():
        main_val = str(row[main_col]).strip()
        if not main_val:
            continue

        for col in other_cols:
            val = str(row[col]).strip()
            if val and val not in ["", "nan", "None"]:
                key = f"{main_val} - {col}"
                result[key] = val

    return result

def ocr_process(pdf_path: Path):
    pages = convert_from_path(pdf_path)
    all_text = ""

    for _, page in enumerate(pages):
        text = pytesseract.image_to_string(page, lang="dan", config="--psm 4")
        all_text += text + "\n\n"

    return all_text

def process_payslip(file_path: Path, file_format: str):
    """
    Process payslip PDF with coordinates from your original code
    """
    coordinates = [(94, 293, 431, 457),
        (860, 298, 1611, 545),
        (112, 577, 1609, 1475),
        (113, 1480, 862, 1761),
        (861, 1480, 1614, 1761),
        (111, 1761, 865, 2208),
        (862, 1764, 1613, 2212),
        (112, 1291, 1613, 1472)
    ]

    coordinates_dict = {
        "top_left": coordinates[0],
        "top_right": coordinates[1],
        "middle": coordinates[2],
        "middle_left": coordinates[3],
        "middle_right": coordinates[4],
        "lower_left": coordinates[5],
        "lower_right": coordinates[6],
        "middle_middle": coordinates[7]
    }

    pages = convert_from_path(file_path)
    res = []

    for page in pages:
        res_dict = {}

        for section_name, coordinate in coordinates_dict.items():
            x1,y1,x2,y2 = coordinate
            cropped = page.crop((x1, y1, x2, y2))

            if section_name == "top_left":
                text = pytesseract.image_to_string(cropped, lang="dan", config="--psm 4")
                lines = [line.strip() for line in text.splitlines() if line.strip()]

                name = None
                address_parts = []
                postcode = None

                for line in lines:
                    print(line)
                    if not name:
                        name = line
                        continue
                    
                    # Check for postal code (4 digits at start)
                    if re.match(r"^\d{4}", line):
                        postcode = line
                        break  # stop after postcode (no more data below)
                    else:
                        address_parts.append(line)

                # If postcode not found, try the last non-empty line
                if not postcode and address_parts:
                    last_line = address_parts[-1]
                    if re.match(r"^\d{4}", last_line):
                        postcode = last_line
                        address_parts = address_parts[:-1]

                # Fill the result dict
                res_dict["Navn"] = name or ""
                res_dict["Adresse"] = " ".join(address_parts).strip()
                res_dict["Post-nr"] = postcode or ""
                
            if section_name == "top_right":
                text = pytesseract.image_to_string(cropped, lang="dan", config="--psm 4")
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                for i, line in enumerate(lines):
                    line = line.strip()
                    if ":" in line:
                        key, val = line.split(":", 1)
                        res_dict[key.strip()] = val.strip()
                    else:
                        # handle lines without ':' (continuation lines)
                        if res_dict:
                            last_key = list(res_dict.keys())[-1]
                            res_dict[last_key] += " " + line.strip()

            if section_name == "middle":
                middle_columns_dict = {
                    "Tekst": (2, 747),
                    "Grundlag": (748, 966),
                    "Sats": (965, 1128),
                    "Udbetalt": (1130, 1313),
                    "Trukket": (1313, 1492)
                }

                middle_df = extract_table_from_section(cropped, middle_columns_dict, 20, "dan", False)
                middle_dict = extract_column_from_df(middle_df)
                res_dict.update(middle_dict)

            if section_name == "middle_left":
                middle_left_columns_dict = {
                    "Ferieregnskab": (0, 358),
                    "Perioden": (389, 520),
                    "Ferieår til dato": (531, 748),
                }

                middle_left_df = extract_table_from_section(cropped, middle_left_columns_dict, 20, "dan", False)
                middle_left_dict = extract_column_from_df(middle_left_df)
                res_dict.update(middle_left_dict)

            if section_name == "middle_right":
                middle_right_columns_dict = [
                    {
                        "Ferieregnskab": (2, 601),
                        "Saldo": (651, 751),
                    },
                    {
                        "Ferieregnskab": (2, 346),
                        "Udbetalt": (354, 526),
                        "Trukket": (541, 750),
                    }
                ]

                middle_right_df = extract_table_from_section(cropped, middle_right_columns_dict, 20, "dan", False)
                middle_right_dict = extract_column_from_df(middle_right_df)
                res_dict.update(middle_right_dict)

            if section_name == "lower_left":
                lower_left_columns_dict = {
                    "Saldo": (2, 352),
                    "Perioden": (361, 520),
                    "År til dato": (533, 753),
                }

                lower_left_df = extract_table_from_section(cropped, lower_left_columns_dict, 20, "dan", False)
                lower_left_dict = extract_column_from_df(lower_left_df)
                res_dict.update(lower_left_dict)

            if section_name == "lower_right":
                lower_right_columns_dict = {
                    "Saldo": (2, 352),
                    "Perioden": (361, 520),
                    "År til dato": (533, 753),
                }

                lower_right_df = extract_table_from_section(cropped, lower_right_columns_dict, 20, "dan", False)
                lower_right_dict = extract_column_from_df(lower_right_df)
                res_dict.update(lower_right_dict)
            
            # check if the payslip has additional middle section
            if section_name == "middle_middle":
                middle_middle_columns_dict = {
                    "Ferieafregning, fratrædelse": (0, 401),
                    "Dage": (419, 663),
                    "Brutto": (675, 854),
                    "AM-bidrag": (868, 1058),
                    "A-skat": (1077, 1289),
                    "Netto": (1310, 1499)
                }

                middle_middle_df = extract_table_from_section(cropped, middle_middle_columns_dict, 20, "dan", True)
                if middle_middle_df is not None:
                    middle_middle_dict = extract_column_from_df(middle_middle_df)
                    res_dict.update(middle_middle_dict)

        if res_dict and res_dict["Navn"] != "Marianne Trolle" and res_dict["Navn"] != "Lisbeth Hald Anderse":        
            res.append(res_dict)
        else:
            print("no record found")

    print("OCR complete")
    df = pd.DataFrame.from_records(res)

    return df

def process_employee_general(file_path, file_type):
    """
    Process employee PDF based on your parse_employee_txt_new logic
    """
    text = ocr_process(file_path)
    employees = re.split(r"Frikort brugt:\s*[\d,\.]+", text)
    records = []
    
    key_value_pattern = re.compile(r"^(.+?)\s+([\w\d,\.%\-]+)$")
    sub_headers = ["Generelt", "Lønoplysninger", "Ansættelsesoplysninger", "Kontakt", "Ferie", "SH", "Fritvalgsordning", "Dage/Timer", "Skattekort"]
    
    for emp in employees:
        emp = emp.strip()
        if not emp:
            continue

        data = {}
        lines = [line.strip() for line in emp.splitlines() if line.strip()]

        for line in lines:
            # --- Normal colon-based case ---
            if line in sub_headers:
                continue

            if ":" in line:
                key, value = line.split(":", 1)
                key, value = key.strip(), value.strip()

            # --- Case 2: "key value" (no colon) ---
            elif match := key_value_pattern.match(line):
                key, value = match.group(1).strip(), match.group(2).strip()

            # --- Case 3: key only or invalid ---
            else:
                continue

            # Skip if no value or header-like entry
            if not value or key in sub_headers:
                continue

            data[key] = value

        # Add the trailing "Frikort brugt" value if found
        match = re.search(r"Frikort brugt:\s*([\d,\.]+)", emp)
        if match:
            data["Frikort brugt"] = match.group(1)

        if data:  
            records.append(data)

    # --- Build final DataFrame ---
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.iloc[:, 1:]  # Drop first column if unwanted

    if "Fritvalgsordning 1%" in df.columns:
        df = df.rename(columns={"Fritvalgsordning 1%": "Fritvalgsordning 1 %"})

    return df

def process_employee_list(file_path, file_type):
    """
    Process employee list PDF based on your parse_employee_list_txt logic
    """
    text = ocr_process(file_path)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # Find header and filter lines
    headers = ["Navn", "Medarb.nr.", "Lønperiode", "Adresse", "Adresse 2", "Post-nr. og by"]
    lønperiode_options = {"Måned bagud", "14-dags", "(TOM/FINDES IKKE I DANLØN)"}
    ignore_patterns = ["Din Forsikringsmægler ApS", "Medarbejderliste", "Side "]
    
    # Filter lines
    for i, line in enumerate(lines):
        if line.strip().startswith("Navn"):
            lines = lines[i+1:]
            break
    
    lines = [line for line in lines if not any(pat in line for pat in ignore_patterns)]
    lines = [line.strip().split() for line in lines]
    
    rows = []
    for tokens in lines:
        current = {}

        # --- Navn: all tokens until we hit a number ---
        navn_parts = []
        while tokens and not re.fullmatch(r"\d+", tokens[0]):
            navn_parts.append(tokens.pop(0))
        current["Navn"] = " ".join(navn_parts).strip()

        # --- Medarb.nr. ---
        current["Medarb.nr."] = tokens.pop(0) if tokens and re.fullmatch(r"\d+", tokens[0]) else ""

        # --- Lønperiode: one of the known phrases ---
        lønperiode_parts = []
        while tokens:
            token = tokens.pop(0)
            token = token.strip(".,;:")
            lønperiode_parts.append(token)
            if " ".join(lønperiode_parts) in lønperiode_options:
                break
            
        current["Lønperiode"] = " ".join(lønperiode_parts)

        # --- Post-nr. og by (last two tokens, always starts with digits) ---
        post_idx = None
        for j, tok in enumerate(tokens):
            if re.fullmatch(r"\d{4}", tok):  # exactly 4 digits
                post_idx = j
                break

        if post_idx is not None:
            # Split into adresse + post-nr
            current["Adresse"] = " ".join(tokens[:post_idx]).strip()
            current["Post-nr. og by"] = " ".join(tokens[post_idx:]).strip()
        else:
            # fallback: no post number detected
            current["Adresse"] = " ".join(tokens).strip()
            current["Post-nr. og by"] = ""

        # --- Adresse ---
        current["Adresse"] = " ".join(tokens).strip()
        current["Adresse 2"] = ""  # always blank for now

        rows.append(current)

    df = pd.DataFrame(rows, columns=headers)
    df = df.dropna(subset=["Navn", "Medarb.nr."])
    df = df[df["Navn"].str.strip() != ""]

    return df

def load_danlon_mapping():
    """
    Load the Danlon mapping Excel file
    """
    def excel_col_letter(n: int) -> str:
        letters = ""
        while n >= 0:
            n, remainder = divmod(n, 26)
            letters = chr(65 + remainder) + letters
            n -= 1
        return letters

    mapping_path = os.path.join(os.path.dirname(__file__), 'mapping', 'danlon_mapping_new.xlsx')
    
    try:
        # Read the mapping file
        wb = load_workbook(mapping_path, data_only=False)
        ws = wb["Mapping"]

        rows = list(ws.iter_rows(values_only=False))

        data = [[cell.value for cell in row] for row in rows]

        danlon_mapping_df = pd.DataFrame(data)
        
        # Extract the mapping rows (based on your original logic)
        mapping_dan_row = danlon_mapping_df.iloc[0, 1:].tolist()
        mapping_eng_row = danlon_mapping_df.iloc[1, 1:].tolist()
        mapping_input_col = danlon_mapping_df.iloc[4, 1:].tolist()
        mapping_file = danlon_mapping_df.iloc[5, 1:].tolist()
        mapping_datatype = danlon_mapping_df.iloc[7, 1:].tolist()
        mapping_guidance = danlon_mapping_df.iloc[8, 1:].tolist()
        mapping_comments = danlon_mapping_df.iloc[10, 1:].tolist()
        
        # Create excel references
        excel_refs = [excel_col_letter(i) for i in range(len(danlon_mapping_df.columns))]
        excel_refs = excel_refs[1:]
        
        # Create the final mapping DataFrame
        danlon_mapping = pd.DataFrame({
            "danish": mapping_dan_row,
            "english": mapping_eng_row,
            "input_column": mapping_input_col,
            "file": mapping_file,
            "datatype": mapping_datatype,
            "guidance": mapping_guidance,
            "comments": mapping_comments,
            "excel_ref": excel_refs
        })
        
        return danlon_mapping
        
    except Exception as e:
        logger.error(f"Error loading mapping file: {e}")
        logger.error(traceback.format_exc())
        # Return empty mapping if file not found
        return pd.DataFrame()

def apply_danlon_mapping(merged_data):
    """
    Apply Danlon mapping to merged data (simplified from your original mapping function)
    """
    def evaluate_excel_formula(df, formula, danlon_mapping):
        if not isinstance(formula, str):
            return None
        
        expr = formula.lstrip("=").replace(",", ".")

        tokens = re.findall(r"\b[A-Z]{1,3}\b", expr)

        for token in tokens:
            match = danlon_mapping.loc[danlon_mapping["excel_ref"] == token, "danish"]
            if not match.empty:
                colname = match.iloc[0]
                expr = re.sub(rf"\b{token}\b", f"df['{colname}']", expr)
        try:
            result = eval(expr)
        except Exception as e:
            print(f"Error evaluating formula {formula}: {e}")
            result = None

        return result
    
    danlon_mapping = load_danlon_mapping()
    
    if danlon_mapping.empty:
        print("No mapping file found, returning data as-is")
        return merged_data

    indkomsttype_mapping = {
        "A-Indkomstmodtager/lønansat (kode 00)": "A-INDKOMSTMODTAGER/LØNANSAT (KODE 00)",
        "Anden personlig indkomst/ej lønansat (kode 04)": "ANDEN PERSONLIG INDKOMST/EJ LØNANSAT (kode 04)",
        "B-indkomst modtager/ej lønansat (kode 05)": "B-INDKOMST MODTAGER/EJ LØNANSAT (KODE 05)",
        "§48E (forskerordningen) (kode 08)": "§ 48 E (FORSKERORDNINGEN) (KODE 08)",
        "Skattefri indkomst (kode 09) - udlændinge": "SKATTEFRI INDKOMST (KODE 09)",
        "Skattefri indkomst (kode 09)": "SKATTEFRI INDKOMST (KODE 09)"
    }

    barselsfond_mapping = {
        "Barsel.dk - fuldt medlem": "BARSEL.DK",
        "Frivillig DA barsel": "DA BARSEL",
        "Obligatorisk DA barsel": "DA BARSEL",
        "Barsel.dk - delvis medlem": "BARSEL.DK - DELVIST MEDLEM",
        "Anden barselsordning": "ANDEN BARSELSFOND",
    }

    atp_type_mapping = {
        "Almindelig (A)": "ALMINDELING A",
        "Ingen": None,
    }

    lønperiode_mapping = {
        "Måned bagud": "BAGUDBETALT MÅNEDSLØN",
        "14-dags": "14 DAGE",
    }
    ferieordning_mapping = {
        "Løbende (feriekasse)": "TIMELØNNET (MED FERIEPENGE)",
        "Ingen": "FAST LØN",
        "Ferie med løn": "FAST LØN",
        "Løbende (eget)": "TIMELØNNET (MED FERIEPENGE)"
    }
    
    columns = danlon_mapping["danish"].tolist()

    final_output_df = pd.DataFrame(columns=columns)
    for _, row in danlon_mapping.iterrows():
        try:
            if pd.isna(row["danish"]):
                continue
            print("mapping: ", row["danish"])
            danish_col = row["danish"]
            input_col = row["input_column"]

            if not isinstance(input_col, str) or pd.isna(input_col):
                input_col = ""

            # Build possible column names to look for
            possible_cols = []
            if input_col:
                possible_cols = [input_col, f"{input_col}_df1", f"{input_col}_df2"]
                
                if "-" in input_col:
                    parts = [p.strip() for p in input_col.split("-")]
                    if len(parts) >= 2:
                        fixed = f"{parts[0]} - {parts[1].capitalize()}"
                        possible_cols.append(fixed)

            # Find first column that exists in input_df
            found_col = next((col for col in possible_cols if col in merged_data.columns), None)

            if found_col:
                print(f"found_col raw value: {repr(found_col)}")
                found_col = found_col.strip()           # remove spaces / newlines
                found_col = found_col.replace("\xa0", " ")  # replace non-breaking spaces
                found_col = found_col.replace("-", "-")     # replace non-breaking hyphen
                found_col = found_col.replace("–", "-")     # replace en dash
                found_col = found_col.replace("—", "-")
                print(f"Normalized found_col: '{found_col}'")

                if found_col == "Postnummer og By":
                    merged_data[found_col] = merged_data[found_col].astype(str).str.strip()

                    merged_data["Postnummer_split"] = merged_data[found_col].str.extract(r"(\d+)")
                    merged_data["By_split"] = merged_data[found_col].str.replace(r"\d+", "", regex=True).str.strip()

                    if danish_col == "Postnummer":
                        final_output_df[danish_col] = merged_data["Postnummer_split"]
                        print("✅ Extracted Postnummer")

                    elif danish_col == "By":
                        final_output_df[danish_col] = merged_data["By_split"]
                        print("✅ Extracted By")

                # --- Indkomsttype Mapping ---
                elif found_col == "Indkomsttype":
                    print("found_col is Indkomsttype")
                    merged_data[found_col] = merged_data[found_col].astype(str).str.strip()
                    merged_data[found_col] = (
                        merged_data[found_col]
                        .map(indkomsttype_mapping)
                        .fillna("A-INDKOMSTMODTAGER/LØNANSAT (KODE 00)")
                    )
                    final_output_df[danish_col] = merged_data[found_col]
                    print("✅ Mapping applied for Indkomsttype")

                # --- Barselsfond Mapping ---
                elif found_col == "Barsel":
                    print("found_col is Barselsfond")
                    merged_data[found_col] = merged_data[found_col].astype(str).str.strip()
                    merged_data[found_col] = merged_data[found_col].map(barselsfond_mapping)
                    final_output_df[danish_col] = merged_data[found_col]
                    print("✅ Mapping applied for Barselsfond (unmapped values remain NaN)")

                # --- ATP-type Mapping ---
                elif found_col == "ATP-ordning":
                    print("found_col is ATP-type")
                    merged_data[found_col] = merged_data[found_col].astype(str).str.strip()
                    merged_data[found_col] = (
                        merged_data[found_col]
                        .map(atp_type_mapping)
                        .fillna("ALMINDELING A")
                    )
                    final_output_df[danish_col] = merged_data[found_col]
                    print("✅ Mapping applied for ATP-type")

                # --- Lønperiode Mapping ---
                elif found_col == "Lønperiode":
                    print("found_col is Lønperiode")
                    merged_data[found_col] = merged_data[found_col].astype(str).str.strip()
                    merged_data[found_col] = (
                        merged_data[found_col]
                        .map(lønperiode_mapping)
                        .fillna("FORUDBETAL MÅNEDSLØN")
                    )
                    final_output_df[danish_col] = merged_data[found_col]
                    print("✅ Mapping applied for Lønperiode")
                
                # --- Løntype Mapping ---
                elif found_col == "Ferieordning":
                    print("found_col is Ferieordning")
                    merged_data[found_col] = merged_data[found_col].astype(str).str.strip()
                    merged_data[found_col] = (
                        merged_data[found_col]
                        .map(ferieordning_mapping)
                        .fillna("FAST LØN")
                    )
                    final_output_df[danish_col] = merged_data[found_col]
                    print("✅ Mapping applied for Lønperiode")

                else:
                    # Normal case: copy directly
                    final_output_df[danish_col] = merged_data[found_col]

            else:
                final_output_df[danish_col] = pd.NA

        except Exception as e:
            print("error occurs: ", e)
    
    if "Ferieordning" in final_output_df.columns:
        final_output_df["Indtægtsart"] = final_output_df["Ferieordning"].apply(
            lambda x: "Direktør" if str(x).strip().lower() == "FAST LØN" else "Medarbejder"
        )
        print("✅ Derived column Indtægtsart created based on Ferieordning") 
    
    if "Adresse" in final_output_df.columns:
        if "Postnummer" in final_output_df.columns:
            final_output_df["Adresse"] = final_output_df.apply(
                lambda row: str(row["Adresse"]).replace(str(row["Postnummer"]), "")
                if pd.notna(row["Adresse"]) and pd.notna(row["Postnummer"])
                else row["Adresse"],
                axis=1
            )
        if "By" in final_output_df.columns:
            final_output_df["Adresse"] = final_output_df.apply(
                lambda row: str(row["Adresse"]).replace(str(row["By"]), "")
                if pd.notna(row["Adresse"]) and pd.notna(row["By"])
                else row["Adresse"],
                axis=1
            )

    # fill up calculated columns
    for _, row in danlon_mapping.iterrows():
        if str(row["input_column"]).strip() == "(calculated field)":
            print(row["danish"])
            formula = row.get("datatype")
            colname = row["danish"]
            print("formula", formula)
            print("colname", colname)
            res = evaluate_excel_formula(final_output_df, formula, danlon_mapping)
            final_output_df[colname] = res

    return final_output_df

def merge_and_map_data(processed_data):
    """
    Merge processed data based on your original logic and apply mapping
    """
    key = None
    # Start with the main employee data if available
    if "employee_active" in processed_data:
        key = "employee_active"
        merged = processed_data["employee_active"].copy()
    # elif:
    #     return "no active employee list"
    elif "employee_general" in processed_data:
        key = "employee_general"
        merged = processed_data["employee_general"].copy()
    elif "employee_payslip" in processed_data:
        # If no employee data, create empty DataFrame with payslip data
        key = "employee_payslip"
        merged = processed_data.get("employee_payslip", pd.DataFrame())
    else:
        return None
    
    # Merge with holiday data if available
    if "employee_holiday" in processed_data and not merged.empty:
        logging.info("merging employee holiday")
        holiday_df = processed_data["employee_holiday"]
        if "Navn" in merged.columns and "Navn" in holiday_df.columns:
            merged = pd.merge(merged, holiday_df, on="Navn", how="left", suffixes=('_df1', '_df2'))
    
    # Merge with employee list if available
    if "employee_list" in processed_data and not merged.empty:
        logging.info("merging employee list")
        employee_list_df = processed_data["employee_list"]
        if "Medarb.nr." in merged.columns and "Medarb.nr." in employee_list_df.columns:
            # Ensure consistent data types
            merged["Medarb.nr."] = merged["Medarb.nr."].astype(str)
            employee_list_df["Medarb.nr."] = employee_list_df["Medarb.nr."].astype(str)
            merged = pd.merge(merged, employee_list_df, on="Medarb.nr.", how="left", suffixes=('_df1', '_df2'))
    
    # Merge with employee data if available
    if "employee_general" in processed_data and not merged.empty and key != "employee_general":
        logging.info("merging employee general")
        employee_df = processed_data["employee_general"]
        if "Medarb.nr." in merged.columns and "Medarb.nr." in employee_df.columns:
            merged["Medarb.nr."] = merged["Medarb.nr."].astype(str)
            employee_df["Medarb.nr."] = employee_df["Medarb.nr."].astype(str)
            merged = pd.merge(merged, employee_df, on="Medarb.nr.", how="left", suffixes=('_df1', '_df2'))
    
    if "employee_payslip" in processed_data and not merged.empty and key != "employee_payslip":
        logging.info("merging employee_payslip")
        payslip_df = processed_data["employee_payslip"]
        if "Medarb.nr." in merged.columns and "Medarb.nr." in payslip_df.columns:
            merged["Medarb.nr."] = merged["Medarb.nr."].astype(str)
            payslip_df["Medarb.nr."] = payslip_df["Medarb.nr."].astype(str)
            merged = pd.merge(merged, payslip_df, on="Medarb.nr.", how="left", suffixes=('_df1', '_df2'))
    
    # If merged is still empty, just return the largest available dataset
    if merged.empty:
        largest_key = max(processed_data.keys(), key=lambda k: len(processed_data[k]), default=None)
        if largest_key:
            merged = processed_data[largest_key]
    
    # Apply Danlon mapping
    if not merged.empty:
        logging.info("start mapping merged employee")
        mapped_result = apply_danlon_mapping(merged)
    
    return mapped_result