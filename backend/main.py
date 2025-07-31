import sqlite3
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from typing import Optional
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import logging
import re
import csv
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import io
from fastapi.staticfiles import StaticFiles

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

        conn.commit() # Commit changes for expenses table alterations
        conn.close()
        logger.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise

    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS categorization_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL UNIQUE,
                category TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        logger.info("Categorization rules table initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Database error for categorization_rules: {e}")
        raise

# --- FastAPI Application ---
app = FastAPI(title="Digital Shoebox API")

def extract_expense_data(ocr_text: str):
    """
    Extracts expense data from OCR text using a multi-layered approach.
    It first tries to classify the receipt and then applies specific rules.
    """
    lines = ocr_text.split('\n')
    normalized_text = ocr_text.lower()

    # --- Apply Categorization Rules First ---
    category = "Uncategorized"
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT keyword, category FROM categorization_rules")
        rules = cursor.fetchall()
        conn.close()

        for rule in rules:
            if rule["keyword"].lower() in normalized_text:
                category = rule["category"]
                break
    except sqlite3.Error as e:
        logger.error(f"Error fetching categorization rules: {e}")

    # --- Receipt Profiling ---
    profile = "generic"
    if "bunnings" in normalized_text:
        profile = "bunnings"
    elif any(keyword in normalized_text for keyword in ["australia post", "auspost", "lodgement", "parcel post"]):
        profile = "auspost"
    elif "bpay" in normalized_text or ("invoice" in normalized_text and "amount due" in normalized_text):
        profile = "utility_bill"

    # --- Extraction based on Profile ---
    vendor = "Unknown Vendor"
    amount = 0.0
    expense_date = ""
    category = "Uncategorized"
    payment_method = "Unknown"

    # Profile-specific logic
    if profile == "bunnings":
        vendor = "Bunnings Warehouse"
        amount_match = re.search(r'total\s*\$?([\d,]+\.\d{2})', normalized_text, re.IGNORECASE)
        if amount_match:
            amount = float(amount_match.group(1).replace(',', ''))

    elif profile == "auspost":
        vendor = "Australia Post"
        amount_match = re.search(r'total\s*\$?([\d,]+\.\d{2})', normalized_text, re.IGNORECASE)
        if amount_match:
            amount = float(amount_match.group(1).replace(',', ''))

    elif profile == "utility_bill":
        for line in lines[:3]:
            if len(line.strip()) > 3:
                vendor = line.strip()
                break
        amount_match = re.search(r'(?:amount due|new charges|total amount due|please pay)\s*\$?([\d,]+\.\d{2})', normalized_text, re.IGNORECASE)
        if amount_match:
            amount = float(amount_match.group(1).replace(',', ''))

    # --- Generic Fallback Logic ---
    if vendor == "Unknown Vendor":
        # Scoring system for vendor
        best_score = 0
        for line in lines[:8]: # Check top 8 lines
            line = line.strip()
            if len(line) > 3 and len(line) < 40:
                score = 0
                if line.isupper(): # All caps is a good sign
                    score += 10
                if re.search(r'(ltd|pty|inc|warehouse|store|market)', line.lower()): # Keywords
                    score += 10
                if not re.search(r'[\d/:.]', line): # No numbers, slashes, or colons
                    score += 5
                if not any(keyword in line.lower() for keyword in ["total", "date", "cash", "change", "gst", "invoice", "receipt", "address", "phone", "delivery"]):
                    score += 5
                
                if score > best_score:
                    best_score = score
                    vendor = line

    if amount == 0.0:
        amounts = re.findall(r'\$?([\d,]+\.\d{2})', normalized_text)
        if amounts:
            amount = max([float(a.replace(',', '')) for a in amounts])

    # --- Date Extraction (remains the same) ---
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
                if re.match(r'\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}', raw_date):
                    parts = re.split(r'[-/.]', raw_date)
                    if len(parts[2]) == 2: parts[2] = '20' + parts[2]
                    expense_date = f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
                elif re.match(r'\d{4}[-/.]\d{1,2}[-/.]\d{1,2}', raw_date):
                    parts = re.split(r'[-/.]', raw_date)
                    expense_date = f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
                elif re.match(r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}', raw_date, re.IGNORECASE):
                    expense_date = datetime.strptime(raw_date, '%d %b %Y').strftime('%Y-%m-%d')
                break
            except ValueError:
                pass

    # If category is still Uncategorized, try to get suggestions
    if category == "Uncategorized":
        category_suggestions = get_category_suggestions(ocr_text, vendor)
        if category_suggestions:
            category = category_suggestions[0] # Use the first suggestion as the default

    return {
        "vendor": vendor,
        "amount": amount,
        "expense_date": expense_date,
        "category": category,
        "payment_method": payment_method,
        "category_suggestions": get_category_suggestions(ocr_text, vendor) # Still provide suggestions for manual override
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
            # PIL import is here because pytesseract needs it, but it's not directly used by FastAPI
            from PIL import Image, ImageOps
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

@app.get("/upcoming-bills")
async def get_upcoming_bills():
    """
    Retrieves upcoming recurring bills for the next 30 days.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        today = datetime.today().strftime('%Y-%m-%d')
        thirty_days_later = (datetime.today() + timedelta(days=30)).strftime('%Y-%m-%d')

        query = "SELECT vendor, amount, next_due_date FROM recurring_expenses WHERE next_due_date >= ? AND next_due_date <= ? ORDER BY next_due_date ASC"
        
        cursor.execute(query, (today, thirty_days_later))
        upcoming_bills = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return upcoming_bills

    except sqlite3.Error as e:
        logger.error(f"Failed to retrieve upcoming bills: {e}")
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

@app.post("/categorization_rules")
async def create_categorization_rule(
    keyword: str = Form(...),
    category: str = Form(...)
):
    """
    Creates a new categorization rule.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO categorization_rules (keyword, category) VALUES (?, ?)", (keyword, category))
        conn.commit()
        conn.close()
        logger.info(f"Categorization rule created: {keyword} -> {category}")
        return {"status": "success", "keyword": keyword, "category": category}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Rule with this keyword already exists.")
    except sqlite3.Error as e:
        logger.error(f"Failed to create categorization rule: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.get("/categorization_rules")
async def get_categorization_rules():
    """
    Retrieves all categorization rules.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, keyword, category FROM categorization_rules")
        rules = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rules
    except sqlite3.Error as e:
        logger.error(f"Failed to retrieve categorization rules: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.delete("/categorization_rules/{rule_id}")
async def delete_categorization_rule(rule_id: int):
    """
    Deletes a categorization rule by ID.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM categorization_rules WHERE id = ?", (rule_id,))
        conn.commit()
        conn.close()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Rule not found")
        logger.info(f"Categorization rule with ID {rule_id} deleted successfully.")
        return {"status": "success", "message": f"Rule {rule_id} deleted"}
    except sqlite3.Error as e:
        logger.error(f"Failed to delete categorization rule: {e}")
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
                f"{row['gst']:.2f}"
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

@app.get("/api/export/csv")
async def export_all_to_csv():
    """
    Generates a CSV file of all expenses.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT expense_date, vendor, description, category, amount, gst, payment_method, tags, is_tax_deductible FROM expenses ORDER BY expense_date DESC"

        cursor.execute(query)
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
                f"{exp['amount']:.2f}",
                f"{exp['gst']:.2f}",
                exp["payment_method"],
                exp["tags"],
                "Yes" if exp["is_tax_deductible"] == 1 else "No"
            ])

        output.seek(0)

        filename = f"digital_shoebox_export_{datetime.now().strftime('%Y-%m-%d')}.csv"

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

    except sqlite3.Error as e:
        logger.error(f"Failed to generate full CSV export: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

# Mount the 'frontend' directory to serve the main application
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
