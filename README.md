# BillBot

BillBot is a langgraph based agent that extracts, processes, and stores invoice data using **OCR (Tesseract)** and **LLM-powered AI** (Groq LLM). It also allows users to generate and execute SQL queries on the invoice database.

## **Features**

- Extracts text from invoice images using **Tesseract OCR**.
- Uses **Open source model llama-3.3-70b-versatile with Groq LLM** to parse extracted text into structured invoice data.
- Stores invoice details in an SQLite database.
- Provides an **LLM-powered SQL query generator** to retrieve or modify invoice data.
- Supports **automatic routing** between image processing and SQL query execution.
- Uses **LangChain and LangGraph** for efficient AI-based workflows.

---

## **Installation**

### **Prerequisites**

- Python **>=3.11**
- `pip` installed on your system


### **Environment Variables**

Create a `.env` file in the project directory and add:

```sh
GROQ_API_KEY=your_api_key_here
```

---

### **Deploying app using LangGraph local server**

- Install the LangGraph CLi
To begin, install the LangGraph CLI with the following command:
  ```sh
  pip install --upgrade "langgraph-cli[inmem]"
  ```

- Install dependencies:
- Navigate to your project directory (e.g., ...\...\langgraph_agent) and install the required dependencies:
  ```sh
  pip install -e .
  ```

- Start the LangGraph development server by running:
  ```sh
  langgraph dev
  ```


## **Usage**
### **Note:**
-By default, a set of invoice images from [Kaggle](https://www.kaggle.com/dsv/9489831) has been preloaded into the database. If you wish to upload your own invoice data, simply place your images in the batch_1/ folder. When prompted during the LangGraph application setup, respond with 'yes' to populate the database with your new data.


### **1. Extract and Store Invoice Data**

- Place invoice images in the `batch_1/` folder.
- The script will:
  1. Extract text from images.
  2. Parse extracted text using **Groq LLM**.
  3. Store structured data in the SQLite database (`invoices.db`).

### **2. Query the Invoice Database**

- You can interact with the AI-powered system to generate SQL queries for managing invoices.

- To add an invoice to the database or generate a summary from an image, use the following queries:
  ```sh
Add "image_path"
Summarise "image_path" 
  ```

- Other Example queries:
You can also other execute natural language queries, and the AI will convert them into SQL commands for execution:
  ```sh
  Show all unpaid invoices.
  Retrieve invoices issued by "XYZ Corp."
  Find invoices with an amount due greater than $500.
  ```
- This enables seamless invoice management through AI-driven SQL query generation and execution.

---

## **Project Structure**

```
│── batch_1/                   # Folder for invoice images
│── bill.py                     # Main script to extract, parse, and store invoices
│── database1.py                 # Database setup using SQLAlchemy
│── invoices.db                   # SQLite database storing invoices
│── .env                         # Environment variables (GROQ API key)
│── pyproject.toml                # Project configuration
│── requirements.txt              # Python dependencies
```

---

## **Database Schema**

The `invoices` table stores invoice details:

```sql
CREATE TABLE invoices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_id TEXT UNIQUE NOT NULL,
    seller_name TEXT,
    amount_due FLOAT NOT NULL DEFAULT 0.0,
    due_date DATE,
    tax_percent FLOAT NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'Unpaid'
);
```

---

## **Technology Stack**

- **Python 3.11+**
- **OpenCV & Tesseract OCR** (Text extraction)
- **LangChain + Groq LLM** (AI-powered parsing & query generation)
- **SQLAlchemy** (Database ORM)
- **SQLite** (Database storage)

---

## **License**

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## **Author**

**Sandeep Kumar**Contact: [sandeepindramohan@gmail.com](mailto\:sandeepindramohan@gmail.com)

---
