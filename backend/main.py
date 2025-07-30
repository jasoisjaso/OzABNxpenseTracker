import sqlite3
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from typing import Optional
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
import pytesseract
import logging
import re
import csv
from datetime import datetime, timedelta

# --- Configuration ---
UPLOADS_DIR = Path("uploads")
DATABASE_FILE = Path("database/expenses.db")
UPLOADS_DIR.mkdir(exist_ok=True)
DATABASE_FILE.parent.mkdir(exist_ok=True)

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Database Initialization ---
def init_db():
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vendor TEXT NOT NULL,
                amount REAL NOT NULL,
                expense_date TEXT NOT NULL,
                description TEXT,
                image_path TEXT,
                category TEXT,
                gst REAL,
                payment_method TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recurring_expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vendor TEXT NOT NULL,
                amount REAL NOT NULL,
                description TEXT,
                category TEXT,
                payment_method TEXT,
                frequency TEXT NOT NULL,
                start_date TEXT NOT NULL,
                next_due_date TEXT NOT NULL,
                end_date TEXT
            )
        """)
        cursor.execute("PRAGMA table_info(recurring_expenses)")
        recurring_columns = [column[1] for column in cursor.fetchall()]
        if 'end_date' not in recurring_columns:
            cursor.execute("ALTER TABLE recurring_expenses ADD COLUMN end_date TEXT")
            logger.info("Added 'end_date' column to 'recurring_expenses' table.")
        conn.commit()

        # Add 'tags' column to expenses table if it doesn't exist
        cursor.execute("PRAGMA table_info(expenses)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'tags' not in columns:
            cursor.execute("ALTER TABLE expenses ADD COLUMN tags TEXT")
            logger.info("Added 'tags' column to 'expenses' table.")

        # Add 'is_tax_deductible' column to expenses table if it doesn't exist
        cursor.execute("PRAGMA table_info(expenses)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'is_tax_deductible' not in columns:
            cursor.execute("ALTER TABLE expenses ADD COLUMN is_tax_deductible INTEGER DEFAULT 0")
            logger.info("Added 'is_tax_deductible' column to 'expenses' table.")

        conn.close()
        logger.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise

# --- FastAPI Application ---
app = FastAPI(title="Digital Shoebox API")

def extract_expense_data(ocr_text: str):
    vendor = "Unknown Vendor"
    amount = 0.0
    expense_date = ""
    category = "Uncategorized"
    payment_method = "Unknown"

    # Normalize text for easier parsing (e.g., remove extra spaces, convert to lowercase for some checks)
    normalized_text = ocr_text.lower()
    lines = ocr_text.split('\n')

    # --- Amount Extraction ---
    # Look for patterns like $XX.XX, XX.XX, or words like TOTAL, BALANCE, AMOUNT DUE
    # More robust regex for amount, considering common total indicators
    amount_patterns = [
        r'(?:total|balance|amount due|subtotal|grand total)\s*[€£]?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))', # e.g., Total $123.45, Balance 1.234,56
        r'[€£]?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))\s*(?:total|balance|amount due)', # e.g., $123.45 Total
        r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))' # General number with two decimal places
    ]
    for pattern in amount_patterns:
        amount_match = re.search(pattern, normalized_text, re.IGNORECASE)
        if amount_match:
            # Clean the matched amount (remove commas, replace comma decimal with dot)
            matched_amount = amount_match.group(1).replace(',', '')
            if '.' in matched_amount and ',' in amount_match.group(1): # Handle European decimal comma
                matched_amount = matched_amount.replace('.', '').replace(',', '.')
            amount = float(matched_amount)
            break

    # --- Date Extraction ---
    # Common Australian date formats: DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, DD Mon YYYY
    date_patterns = [
        r'\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b', # DD/MM/YY or DD-MM-YYYY
        r'\b(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\b', # YYYY-MM-DD
        r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})\b' # DD Mon YYYY
    ]
    for pattern in date_patterns:
        date_match = re.search(pattern, ocr_text, re.IGNORECASE)
        if date_match:
            raw_date = date_match.group(1)
            try:
                # Attempt to parse and normalize to YYYY-MM-DD
                if re.match(r'\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}', raw_date):
                    # Handle DD/MM/YY or DD-MM-YYYY
                    parts = re.split(r'[-/.]', raw_date)
                    if len(parts[2]) == 2: # Convert YY to YYYY
                        parts[2] = '20' + parts[2] if int(parts[2]) < 50 else '19' + parts[2]
                    expense_date = f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
                elif re.match(r'\d{4}[-/.]\d{1,2}[-/.]\d{1,2}', raw_date):
                    # YYYY-MM-DD
                    parts = re.split(r'[-/.]', raw_date)
                    expense_date = f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
                elif re.match(r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}', raw_date, re.IGNORECASE):
                    # DD Mon YYYY
                    from datetime import datetime
                    expense_date = datetime.strptime(raw_date, '%d %b %Y').strftime('%Y-%m-%d')
                break
            except ValueError:
                pass # Keep expense_date as empty if parsing fails

    # --- Vendor Extraction ---
    # Look for common company indicators or specific keywords
    # This is still challenging and often requires more advanced NLP or a database of known vendors.
    # For now, we'll try to find common patterns or keywords.
    vendor_keywords = {
        "coles": "Coles", "woolworths": "Woolworths", "bunnings": "Bunnings",
        "kmart": "Kmart", "target": "Target", "aldi": "Aldi", "iga": "IGA",
        "officeworks": "Officeworks", "telstra": "Telstra", "optus": "Optus",
        "vodafone": "Vodafone", "agl": "AGL", "origin energy": "Origin Energy",
        "qantas": "Qantas", "virgin australia": "Virgin Australia",
        "uber": "Uber", "menulog": "Menulog", "deliveroo": "Deliveroo",
        "amazon": "Amazon", "ebay": "eBay", "microsoft": "Microsoft",
        "google": "Google", "apple": "Apple", "xero": "Xero", "hnry": "Hnry",
        "rounded": "Rounded", "ato": "ATO", "council": "Council"
    }
    for keyword, v_name in vendor_keywords.items():
        if keyword in normalized_text:
            vendor = v_name
            break
    
    # Fallback for vendor: try to find a prominent line that looks like a company name
    if vendor == "Unknown Vendor":
        for line in lines:
            if len(line.strip()) > 5 and len(line.strip()) < 30 and not re.search(r'\d', line) and not re.search(r'total|amount|date', line, re.IGNORECASE):
                # Simple heuristic: line is not too short/long, no numbers, no common expense words
                vendor = line.strip()
                break

    # --- Category Suggestion (based on vendor or keywords) ---
    category_keywords = {
        "shopify": "Software & Subscriptions",
        "facebook": "Advertising & Marketing",
        "instagram": "Advertising & Marketing",
        "google ads": "Advertising & Marketing",
        "marketing": "Advertising & Marketing",
        "post office": "Shipping & Postage",
        "auspost": "Shipping & Postage",
        "pack": "Packaging & Materials",
        "box": "Packaging & Materials",
        "officeworks": "Office Supplies",
        "staples": "Office Supplies",
        "qantas": "Travel",
        "virgin australia": "Travel",
        "flight": "Travel",
        "uber": "Travel",
        "accountant": "Professional Fees",
        "legal": "Professional Fees",
        "consulting": "Professional Fees",
        "telstra": "Utilities",
        "optus": "Utilities",
        "agl": "Utilities",
        "energy": "Utilities",
        "bank fee": "Bank Fees",
        "website": "Website & Hosting",
        "hosting": "Website & Hosting",
        "domain": "Website & Hosting",
        "other business": "Other Business Expenses",
        "council": "Utilities" # Keeping council under utilities for now, as rates are often utility-like
    }
    for keyword, cat_name in category_keywords.items():
        if keyword in normalized_text or keyword in vendor.lower():
            category = cat_name
            break

    # --- Payment Method Suggestion ---
    payment_keywords = {
        "cash": "Cash", "eftpos": "Debit Card", "debit": "Debit Card",
        "credit card": "Credit Card", "visa": "Credit Card", "mastercard": "Credit Card",
        "paypal": "PayPal", "bank transfer": "Bank Transfer", "bpay": "Bank Transfer"
    }
    for keyword, pm_name in payment_keywords.items():
        if keyword in normalized_text:
            payment_method = pm_name
            break

    return {
        "vendor": vendor,
        "amount": amount,
        "expense_date": expense_date,
        "category": category,
        "payment_method": payment_method,
        "category_suggestions": get_category_suggestions(ocr_text, vendor)
    }

def get_category_suggestions(ocr_text: str, vendor: str) -> list[str]:
    """
    Provides a list of suggested categories based on keywords in the OCR text and vendor name.
    """
    suggestions = set()
    normalized_text = ocr_text.lower()
    
    category_keywords = {
        "Software & Subscriptions": ["shopify", "microsoft", "adobe", "xero", "hnry", "rounded", "zoom"],
        "Advertising & Marketing": ["facebook", "instagram", "google ads", "marketing", "advertising"],
        "Shipping & Postage": ["post office", "auspost", "dhl", "fedex"],
        "Packaging & Materials": ["pack", "box", "packaging", "materials"],
        "Office Supplies": ["officeworks", "staples", "winc"],
        "Travel": ["qantas", "virgin australia", "flight", "uber", "taxi", "hotel", "accommodation"],
        "Professional Fees": ["accountant", "legal", "consulting", "lawyer"],
        "Utilities": ["telstra", "optus", "agl", "origin energy", "energy", "water", "internet", "council"],
        "Bank Fees": ["bank fee", "interest"],
        "Website & Hosting": ["website", "hosting", "domain", "godaddy", "ventraip"],
        "Other Business Expenses": ["other business"]
    }

    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in normalized_text or keyword in vendor.lower():
                suggestions.add(category)

    # Return a list, with the primary suggestion first if found
    sorted_suggestions = list(suggestions)
    if "Uncategorized" not in sorted_suggestions:
        sorted_suggestions.append("Uncategorized")
        
    return sorted_suggestions

@app.on_event("startup")
async def startup_event():
    init_db()
    await check_and_create_recurring_expenses()

# Mount the 'uploads' directory to serve static files (the receipt images)
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")



@app.get("/tax_guide.html")
async def get_tax_guide():
    """
    Serves the tax guide HTML file.
    """
    return FileResponse("tax_guide.html")


@app.post("/ocr/upload")
async def ocr_upload(file: UploadFile = File(...)):
    """
    Accepts an image, saves it, performs OCR, and returns the text.
    """
    try:
        # Ensure a unique filename
        file_path = UPLOADS_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Perform OCR
        try:
            img = Image.open(file_path)
            # Apply EXIF orientation if available
            img = ImageOps.exif_transpose(img)
            # Convert to RGB to handle potential RGBA or other modes
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            text = pytesseract.image_to_string(img)
        
            extracted_data = extract_expense_data(text)

            logger.info(f"OCR successful for {file.filename}")
            
            return {
                "text": text,
                "image_path": f"/uploads/{file.filename}",
                "extracted_data": extracted_data
            }
        except Exception as img_e:
            logger.error(f"Image processing or OCR failed for {file.filename}: {img_e}")
            raise HTTPException(status_code=500, detail=f"Image processing or OCR failed: {img_e}")
    except Exception as e:
        logger.error(f"OCR upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during OCR processing: {e}")

@app.post("/expenses")
async def create_expense(
    vendor: str = Form(...),
    amount: float = Form(...),
    expense_date: str = Form(...),
    description: str = Form(""),
    image_file: Optional[UploadFile] = File(None),
    category: str = Form("Uncategorized"),
    is_gst_inclusive: bool = Form(False),
    payment_method: str = Form("Unknown"),
    tags: Optional[str] = Form(None),
    is_tax_deductible: bool = Form(False)
):
    """
    Saves a new expense record to the database.
    """
    calculated_gst = 0.0
    if is_gst_inclusive:
        calculated_gst = amount / 11
    else:
        calculated_gst = 0.0

    image_path = None
    if image_file:
        try:
            file_location = UPLOADS_DIR / image_file.filename
            with open(file_location, "wb+") as file_object:
                shutil.copyfileobj(image_file.file, file_object)
            image_path = f"/uploads/{image_file.filename}"
        except Exception as e:
            logger.error(f"Failed to save uploaded image: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save image: {e}")

    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO expenses (vendor, amount, expense_date, description, image_path, category, gst, payment_method, tags, is_tax_deductible) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (vendor, amount, expense_date, description or "", image_path or "", category, calculated_gst, payment_method, tags, is_tax_deductible)
        )
        conn.commit()
        conn.close()
        logger.info(f"Expense created for vendor: {vendor}")
        return {"status": "success", "vendor": vendor, "amount": amount, "category": category, "gst": calculated_gst, "payment_method": payment_method, "tags": tags, "is_tax_deductible": is_tax_deductible, "image_path": image_path}
    except sqlite3.Error as e:
        logger.error(f"Failed to create expense: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


from datetime import datetime, timedelta
import io
from dateutil.relativedelta import relativedelta

@app.get("/expenses")
async def get_expenses(search: str = None, category: str = None):
    """
    Retrieves all expense records from the database, with optional search and category filters.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT id, vendor, amount, expense_date, description, image_path, category, gst, payment_method, tags, is_tax_deductible FROM expenses"
        params = []
        conditions = []

        if search:
            conditions.append("(vendor LIKE ? OR description LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%"])
        
        if category:
            conditions.append("category = ?")
            params.append(category)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY expense_date DESC"

        cursor.execute(query, params)
        expenses = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return expenses

    except sqlite3.Error as e:
        logger.error(f"Failed to retrieve expenses: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.get("/expenses/summary")
async def get_expenses_summary(start_date: str = None, end_date: str = None):
    """
    Retrieves a summary of expenses, including totals, category breakdown, and daily spending.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build WHERE clause based on dates
        where_clause = ""
        params = []
        conditions = []

        if start_date and end_date:
            conditions.append("expense_date BETWEEN ? AND ?")
            params.extend([start_date, end_date])
        elif start_date:
            conditions.append("expense_date >= ?")
            params.append(start_date)
        elif end_date:
            conditions.append("expense_date <= ?")
            params.append(end_date)

        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)

        # Total amount and total GST
        total_query = f"SELECT SUM(amount) as total_amount, SUM(gst) as total_gst, COUNT(id) as total_transactions FROM expenses {where_clause}"
        cursor.execute(total_query, params)
        totals = cursor.fetchone()

        # Category breakdown
        category_query = f"SELECT category, SUM(amount) as total_amount, SUM(gst) as total_gst FROM expenses {where_clause} GROUP BY category ORDER BY category"
        cursor.execute(category_query, params)
        category_breakdown = [dict(row) for row in cursor.fetchall()]

        # Daily spending
        daily_spending_query = f"SELECT expense_date, SUM(amount) as daily_total FROM expenses {where_clause} GROUP BY expense_date ORDER BY expense_date"
        cursor.execute(daily_spending_query, params)
        daily_spending_data = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return {
            "total_amount": totals["total_amount"] if totals["total_amount"] is not None else 0.0,
            "total_gst": totals["total_gst"] if totals["total_gst"] is not None else 0.0,
            "total_transactions": totals["total_transactions"] if totals["total_transactions"] is not None else 0,
            "average_spending_per_transaction": (totals["total_amount"] / totals["total_transactions"]) if totals["total_transactions"] > 0 else 0.0,
            "category_breakdown": category_breakdown,
            "daily_spending": daily_spending_data
        }

    except sqlite3.Error as e:
        logger.error(f"Failed to retrieve expense summary: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.get("/expenses/payment_method_breakdown")
async def get_payment_method_breakdown(start_date: str = None, end_date: str = None):
    """
    Retrieves spending breakdown by payment method.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        where_clause = ""
        params = []
        conditions = []

        if start_date and end_date:
            conditions.append("expense_date BETWEEN ? AND ?")
            params.extend([start_date, end_date])
        elif start_date:
            conditions.append("expense_date >= ?")
            params.append(start_date)
        elif end_date:
            conditions.append("expense_date <= ?")
            params.append(end_date)

        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)

        query = f"SELECT payment_method, SUM(amount) as total_amount FROM expenses {where_clause} GROUP BY payment_method ORDER BY total_amount DESC"
        cursor.execute(query, params)
        payment_method_breakdown = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return payment_method_breakdown

    except sqlite3.Error as e:
        logger.error(f"Failed to retrieve payment method breakdown: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.get("/expenses/tags_breakdown")
async def get_tags_breakdown(start_date: str = None, end_date: str = None):
    """
    Retrieves spending breakdown by tags.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        where_clause = ""
        params = []
        conditions = []

        if start_date and end_date:
            conditions.append("expense_date BETWEEN ? AND ?")
            params.extend([start_date, end_date])
        elif start_date:
            conditions.append("expense_date >= ?")
            params.append(start_date)
        elif end_date:
            conditions.append("expense_date <= ?")
            params.append(end_date)

        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)

        query = f"SELECT tags, amount FROM expenses {where_clause}"
        cursor.execute(query, params)
        raw_tags_data = cursor.fetchall()
        conn.close()

        aggregated_tags = {}
        for row in raw_tags_data:
            try:
                # Ensure tags is treated as a string before splitting
                tags_str = str(row["tags"]) if row["tags"] is not None else ""
                if tags_str.strip() != '':
                    tags_list = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
                    for tag in tags_list:
                        amount = row["amount"] if row["amount"] is not None else 0.0
                        aggregated_tags[tag] = aggregated_tags.get(tag, 0.0) + amount
            except Exception as tag_e:
                logger.error(f"Error processing tags for row: {row}. Error: {tag_e}")
                # Continue processing other rows even if one fails

        tags_breakdown = []
        for tag, total_amount in aggregated_tags.items():
            tags_breakdown.append({"tags": tag, "total_amount": total_amount})

        tags_breakdown.sort(key=lambda x: x["total_amount"], reverse=True)

        return tags_breakdown

    except sqlite3.Error as e:
        logger.error(f"Database error in get_tags_breakdown: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in get_tags_breakdown: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.get("/expenses/tax_deductible_summary")
async def get_tax_deductible_summary(start_date: str = None, end_date: str = None):
    """
    Retrieves a summary of tax-deductible expenses.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        where_clause = ""
        params = []
        conditions = []

        if start_date and end_date:
            conditions.append("expense_date BETWEEN ? AND ?")
            params.extend([start_date, end_date])
        elif start_date:
            conditions.append("expense_date >= ?")
            params.append(start_date)
        elif end_date:
            conditions.append("expense_date <= ?")
            params.append(end_date)

        conditions.append("is_tax_deductible = 1")

        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)

        query = f"SELECT SUM(amount) as total_deductible_amount, SUM(gst) as total_deductible_gst FROM expenses {where_clause}"
        cursor.execute(query, params)
        tax_summary = cursor.fetchone()
        conn.close()

        return {
            "total_deductible_amount": tax_summary["total_deductible_amount"] if tax_summary["total_deductible_amount"] is not None else 0.0,
            "total_deductible_gst": tax_summary["total_deductible_gst"] if tax_summary["total_deductible_gst"] is not None else 0.0,
        }

    except sqlite3.Error as e:
        logger.error(f"Failed to retrieve tax deductible summary: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.get("/expenses/tax_deductible_summary")
async def get_tax_deductible_summary(start_date: str = None, end_date: str = None):
    """
    Retrieves a summary of tax-deductible expenses.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        where_clause = ""
        params = []
        conditions = []

        if start_date and end_date:
            conditions.append("expense_date BETWEEN ? AND ?")
            params.extend([start_date, end_date])
        elif start_date:
            conditions.append("expense_date >= ?")
            params.append(start_date)
        elif end_date:
            conditions.append("expense_date <= ?")
            params.append(end_date)

        conditions.append("is_tax_deductible = 1")

        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)

        query = f"SELECT SUM(amount) as total_deductible_amount, SUM(gst) as total_deductible_gst FROM expenses {where_clause}"
        cursor.execute(query, params)
        tax_summary = cursor.fetchone()
        conn.close()

        return {
            "total_deductible_amount": tax_summary["total_deductible_amount"] if tax_summary["total_deductible_amount"] is not None else 0.0,
            "total_deductible_gst": tax_summary["total_deductible_gst"] if tax_summary["total_deductible_gst"] is not None else 0.0,
        }

    except sqlite3.Error as e:
        logger.error(f"Failed to retrieve tax deductible summary: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.get("/expenses/{expense_id}")
async def get_expense(expense_id: int):
    """
    Retrieves a single expense record from the database.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, vendor, amount, expense_date, description, image_path, category, gst, payment_method FROM expenses WHERE id = ?", (expense_id,))
        expense = cursor.fetchone()
        conn.close()
        if expense is None:
            raise HTTPException(status_code=404, detail="Expense not found")
        return dict(expense)
    except sqlite3.Error as e:
        logger.error(f"Failed to retrieve expense: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.put("/expenses/{expense_id}")
async def update_expense(
    expense_id: int,
    vendor: str = Form(...),
    amount: float = Form(...),
    expense_date: str = Form(...),
    description: str = Form(""),
    category: str = Form("Uncategorized"),
    is_gst_inclusive: bool = Form(False),
    payment_method: str = Form("Unknown"),
    tags: Optional[str] = Form(None),
    is_tax_deductible: bool = Form(False),
    image_file: Optional[UploadFile] = File(None)
):
    """
    Updates an existing expense record in the database.
    """
    calculated_gst = 0.0
    if is_gst_inclusive:
        calculated_gst = amount / 11
    else:
        calculated_gst = 0.0

    image_path = None
    if image_file:
        try:
            file_location = UPLOADS_DIR / image_file.filename
            with open(file_location, "wb+") as file_object:
                shutil.copyfileobj(image_file.file, file_object)
            image_path = f"/uploads/{image_file.filename}"
        except Exception as e:
            logger.error(f"Failed to save uploaded image for update: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save image: {e}")
    else:
        # Retrieve existing image_path if no new file is uploaded
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT image_path FROM expenses WHERE id = ?", (expense_id,))
        existing_image_path = cursor.fetchone()
        conn.close()
        if existing_image_path:
            image_path = existing_image_path[0]

    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE expenses SET vendor = ?, amount = ?, expense_date = ?, description = ?, category = ?, gst = ?, payment_method = ?, tags = ?, is_tax_deductible = ?, image_path = ? WHERE id = ?",
            (vendor, amount, expense_date, description, category, calculated_gst, payment_method, tags, is_tax_deductible, image_path, expense_id)
        )
        conn.commit()
        conn.close()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Expense not found")
        logger.info(f"Expense with ID {expense_id} updated successfully.")
        return {"status": "success", "message": f"Expense {expense_id} updated"}
    except sqlite3.Error as e:
        logger.error(f"Failed to update expense: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.delete("/expenses/{expense_id}")
async def delete_expense(expense_id: int):
    """
    Deletes an expense record from the database.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
        conn.commit()
        conn.close()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Expense not found")
        logger.info(f"Expense with ID {expense_id} deleted successfully.")
        return {"status": "success", "message": f"Expense {expense_id} deleted"}
    except sqlite3.Error as e:
        logger.error(f"Failed to delete expense: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.post("/recurring_expenses")
async def create_recurring_expense(
    vendor: str = Form(...),
    amount: float = Form(...),
    description: str = Form(""),
    category: str = Form("Uncategorized"),
    payment_method: str = Form("Unknown"),
    frequency: str = Form(...),
    start_date: str = Form(...),
    end_date: Optional[str] = Form(None)
):
    """
    Saves a new recurring expense record to the database.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO recurring_expenses (vendor, amount, description, category, payment_method, frequency, start_date, next_due_date, end_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (vendor, amount, description, category, payment_method, frequency, start_date, start_date, end_date)
        )
        conn.commit()
        conn.close()
        logger.info(f"Recurring expense created for vendor: {vendor}")
        return {"status": "success"}
    except sqlite3.Error as e:
        logger.error(f"Failed to create recurring expense: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.get("/recurring_expenses")
async def get_recurring_expenses():
    """
    Retrieves all recurring expense records from the database.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, vendor, amount, description, category, payment_method, frequency, start_date, next_due_date, end_date FROM recurring_expenses ORDER BY next_due_date ASC")
        recurring_expenses = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return recurring_expenses
    except sqlite3.Error as e:
        logger.error(f"Failed to retrieve recurring expenses: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.delete("/recurring_expenses/{expense_id}")
async def delete_recurring_expense(expense_id: int):
    """
    Deletes a recurring expense record from the database.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM recurring_expenses WHERE id = ?", (expense_id,))
        conn.commit()
        conn.close()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Recurring expense not found")
        logger.info(f"Recurring expense with ID {expense_id} deleted successfully.")
        return {"status": "success", "message": f"Recurring expense {expense_id} deleted"}
    except sqlite3.Error as e:
        logger.error(f"Failed to delete recurring expense: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

async def check_and_create_recurring_expenses():
    """
    Checks for due recurring expenses and creates them as regular expenses.
    """
    today = datetime.today().strftime('%Y-%m-%d')
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM recurring_expenses WHERE next_due_date <= ? AND (end_date IS NULL OR next_due_date <= end_date)", (today,))
        due_expenses = cursor.fetchall()

        for expense in due_expenses:
            # Create a new expense
            await create_expense(
                vendor=expense["vendor"],
                amount=expense["amount"],
                expense_date=expense["next_due_date"],
                description=str(expense["description"]) if expense["description"] is not None else "",
                image_path=None, # Recurring expenses don't have an image
                category=expense["category"],
                payment_method=expense["payment_method"],
                is_gst_inclusive=True # Assuming recurring expenses are GST inclusive
            )

            # Calculate the next due date
            current_next_due_date = datetime.strptime(expense["next_due_date"], '%Y-%m-%d')
            next_due_date = current_next_due_date

            if expense["frequency"] == "monthly":
                next_due_date += relativedelta(months=1)
            elif expense["frequency"] == "quarterly":
                next_due_date += relativedelta(months=3)
            elif expense["frequency"] == "yearly":
                next_due_date += relativedelta(years=1)
            
            # Ensure next_due_date does not exceed end_date if end_date is set
            if expense["end_date"] and next_due_date > datetime.strptime(expense["end_date"], '%Y-%m-%d'):
                # If the next due date goes beyond the end date, we stop further processing for this recurring expense
                # by setting next_due_date to None, so it won't be updated in the database.
                next_due_date = None

            if next_due_date:
                cursor.execute("UPDATE recurring_expenses SET next_due_date = ? WHERE id = ?", (next_due_date.strftime('%Y-%m-%d'), expense["id"]))
            conn.commit()
            
        conn.close()
        logger.info(f"Checked for recurring expenses. {len(due_expenses)} expenses created.")
    except sqlite3.Error as e:
        logger.error(f"Failed to check/create recurring expenses: {e}")

@app.get("/report/csv")
async def get_csv_report(start_date: str = None, end_date: str = None):
    """
    Generates a CSV report of expenses, optionally filtered by date range.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT expense_date, vendor, description, category, amount, gst, payment_method, tags, is_tax_deductible FROM expenses"
        params = []

        if start_date and end_date:
            query += " WHERE expense_date BETWEEN ? AND ?"
            params.append(start_date)
            params.append(end_date)
        elif start_date:
            query += " WHERE expense_date >= ?"
            params.append(start_date)
        elif end_date:
            query += " WHERE expense_date <= ?"
            params.append(end_date)

        query += " ORDER BY expense_date DESC"

        cursor.execute(query, params)
        expenses = cursor.fetchall()
        conn.close()

        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(["Date", "Vendor", "Description", "Category", "Amount", "GST", "Payment Method", "Tags", "Tax Deductible"])

        # Write data rows
        for exp in expenses:
            writer.writerow([
                exp["expense_date"],
                exp["vendor"],
                exp["description"],
                exp["category"],
                f"{exp['amount']:.2f}", # Format amount to 2 decimal places
                f"{exp['gst']:.2f}",    # Format GST to 2 decimal places
                exp["payment_method"],
                exp["tags"],
                "Yes" if exp["is_tax_deductible"] == 1 else "No"
            ])

        output.seek(0)

        # Set filename with date range
        filename = "expenses_report.csv"
        if start_date and end_date:
            filename = f"expenses_report_{start_date}_to_{end_date}.csv"
        elif start_date:
            filename = f"expenses_report_from_{start_date}.csv"
        elif end_date:
            filename = f"expenses_report_to_{end_date}.csv"

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

    except sqlite3.Error as e:
        logger.error(f"Failed to generate CSV report: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.get("/report/summary_csv")
async def get_summary_csv_report(start_date: str = None, end_date: str = None):
    """
    Generates a summarized CSV report of expenses by category, optionally filtered by date range.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT category, SUM(amount) as total_amount, SUM(gst) as total_gst FROM expenses"
        params = []

        if start_date and end_date:
            query += " WHERE expense_date BETWEEN ? AND ?"
            params.extend([start_date, end_date])
        elif start_date:
            query += " WHERE expense_date >= ?"
            params.append(start_date)
        elif end_date:
            query += " WHERE expense_date <= ?"
            params.append(end_date)

        query += " GROUP BY category ORDER BY category"

        cursor.execute(query, params)
        summary_data = cursor.fetchall()
        conn.close()

        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(["Category", "Total Amount", "Total GST"])

        # Write data rows
        for row in summary_data:
            writer.writerow([
                row["category"],
                f"{row['total_amount']:.2f}",
                f"{row['total_gst']:.2f}"
            ])

        output.seek(0)

        # Set filename with date range
        filename = "expenses_summary_report.csv"
        if start_date and end_date:
            filename = f"expenses_summary_report_{start_date}_to_{end_date}.csv"
        elif start_date:
            filename = f"expenses_summary_report_from_{start_date}.csv"
        elif end_date:
            filename = f"expenses_summary_report_to_{end_date}.csv"

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

    except sqlite3.Error as e:
        logger.error(f"Failed to generate summary CSV report: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
