from dotenv import load_dotenv
load_dotenv()
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
GROQ_LLM = ChatGroq(
            model="llama-3.3-70b-versatile", # Corrected the model name
        )

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser

ocr_parser_prompt = PromptTemplate.from_template("""

    Following is extracted text from a invoice using ocr:
    {invoice_text}:

    Extract the following details from this invoice:
    1. Invoice Number (invoice_id) - The unique ID of the invoice.
    2. Seller Name (seller_name) - The name of the seller or company issuing the invoice or bill to or bill from.
    3. Amount Due (amount_due) - The total amount that needs to be paid.
    4. Due Date (due_date) - The deadline for payment in YYYY-MM-DD format.
    5. Tax Percentage (tax_percent) - The tax rate applied to this invoice.

    Ensure the output is in strict JSON format without additional text.
    """
    )

ocr_parser = ocr_parser_prompt | GROQ_LLM | JsonOutputParser()

def parse_invoice_with_groq2(extracted_text):
  structured_data = ocr_parser.invoke({"invoice_text": extracted_text})
  return structured_data

import cv2
import pytesseract


image_folder = "batch_1"

# List all invoice images
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Function to extract text using OCR
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    processed_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # Thresholding
    extracted_text = pytesseract.image_to_string(processed_image)
    return extracted_text


from datetime import datetime   
from invoice_agent.database1 import session
from invoice_agent.database1 import Invoice
from sqlite3 import IntegrityError

def store_invoice_in_db(invoice_data):
    """Handles missing values before inserting into the database and prevents duplicate entries."""

    # Set default values if fields are missing or None
    invoice_id = invoice_data.get("invoice_id") or f"INV-{datetime.now().strftime('%Y%m%d%H%M%S')}"  # Auto-generate if missing
    seller_name = invoice_data.get("seller_name") or "Unknown Seller"  # Default to "Unknown Seller"
    amount_due = invoice_data.get("amount_due") or 0.0  # Default 0.0
    tax_percent = invoice_data.get("tax_percent") or 0.0  # Default 0.0
    status = invoice_data.get("status") or "Unpaid"  # Default "Unpaid"

    # Handle missing or invalid due_date
    try:
        due_date = datetime.strptime(invoice_data["due_date"], "%Y-%m-%d").date() if invoice_data.get("due_date") else None
    except (ValueError, TypeError):
        due_date = None  # Set to NULL if invalid format

    # Check if invoice_id already exists
    existing_invoice = session.query(Invoice).filter_by(invoice_id=invoice_id).first()
    if existing_invoice:
        print(f"Duplicate Entry: Invoice {invoice_id} already exists in the database. Skipping insertion.")
        return

    # Create invoice object
    new_invoice = Invoice(
        invoice_id=invoice_id,
        seller_name=seller_name,
        amount_due=amount_due,
        due_date=due_date,
        tax_percent=tax_percent,
        status=status
    )

    # Insert into database with failsafe for duplicate entry errors
    try:
        session.add(new_invoice)
        session.commit()
        print(f"Stored Invoice {invoice_id} in the database successfully.")
    except IntegrityError:
        session.rollback()  # Rollback in case of failure
        print(f"Error: Invoice {invoice_id} already exists in the database. Skipping insertion.")


user_choice = input("Do you want to populate the database from images? (yes/no): ").strip().lower()

if user_choice in ["yes", "y"]:
    for image_path in image_files[:100]:
        print(f"Processing: {image_path}")

        # Step 1: Extract text from invoice image
        extracted_text = extract_text_from_image(image_path)

        # Step 2: Parse extracted text with Groq AI
        parsed_invoice = parse_invoice_with_groq2(extracted_text)

        # Step 3: Store parsed invoice details in the database
        store_invoice_in_db(parsed_invoice)


#using LLM to Parsed text to Sql query
parsed_to_Sql_prompt = PromptTemplate.from_template("""

  You are an AI assistant that converts parsed text in JSON format into SQL queries for an invoices database.
    The database schema:
    - invoice_id (TEXT, Primary Key)
    - seller_name (TEXT)
    - amount_due (FLOAT)
    - due_date (DATE)
    - tax_percent (FLOAT)
    - status (TEXT)  # "Paid" or "Unpaid"

    The table name is **invoices**.


    Convert the following JSON into an SQL query:
    {parsed_text}

    Ensure the SQL query is valid.

    Give only the raw final SQL query with no markdown formatting, no ```sql tags, and no backticks.
    The output should be exactly the SQL query that can be directly executed.

    """
)

parsed_to_Sql = parsed_to_Sql_prompt | GROQ_LLM | StrOutputParser()


#using LLM to generate sql query from user query

sql_query_gen_prompt = PromptTemplate.from_template("""
    You are an AI assistant that converts user questions into SQL queries for an invoices database.
    The database schema:
    - invoice_id (INT, Primary Key) (Must be a number)
    - seller_name (TEXT)
    - amount_due (FLOAT)
    - due_date (DATE)
    - tax_percent (FLOAT)
    - status (TEXT)  # "Paid" or "Unpaid"

    The table name is **invoices**.


    Convert the following user question into an SQL query:
    {user_query}

    Ensure the SQL query is valid.

    Give only the raw final SQL query with no markdown formatting, no ```sql tags, and no backticks.
    The output should be exactly the SQL query that can be directly executed.

    """
)

sql_query = sql_query_gen_prompt | GROQ_LLM | StrOutputParser()


from sqlalchemy import text
from typing import List, Dict

# function to execute sql queries
def execute_sql(sql_query, session=session):
    try:
        # Execute the raw SQL query
        result = session.execute(text(sql_query))

        if sql_query.strip().upper().startswith('SELECT'):
            # For SELECT queries, return the results as before
            column_names = result.keys()
            rows = result.fetchall()
            results = [dict(zip(column_names, row)) for row in rows]
            return results
        else:
            # For INSERT, UPDATE, DELETE queries, commit the changes and return rowcount
            session.commit()
            return f"Number of rows updated = {result.rowcount}"

    except Exception as e:
        session.rollback()  # Rollback any changes in case of error
        raise


# Using LLM to summarise the actions taken by agent.
summariser_prompt = PromptTemplate.from_template(
    """
    You are the end node of an AI agent, your job is to summarise whatever action the agent took.

    You will receive user query
    {user_query}

    and
    result
    {result}
    after sql query execution,

    this can be

    either text stating the number of rows updated in case UPDATE , DELETE or INSERT query
    in this case " You should reply based on query what was updated ? "

    or  in dictonary format in case of SELECT query
    in this case "you should change this dictionary into human readable format? ."

    Ensure that you give only final answers, user should only know the final output not how it was done

    Also you should reply as a human would reply to query
    """
)

summariser = summariser_prompt | GROQ_LLM | StrOutputParser()


#Router Functions

from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class RouteQuery1(BaseModel):
  next : Literal["image" , "Sql"] = Field(..., description= "Given the user query decide to route it to Image path or to Sql Path.",)

structured_llm_router1 = GROQ_LLM.with_structured_output(RouteQuery1)

system1 = """You are an expert at routing a user question to image path or Sql path
Image path should be used when the user shares a image path in the question or it mentions a image that they want to process.
Sql path be used when the user asks queries reagarding the database or ask to add or update in database or in general things that does not have a image or image path.

"""
route_prompt1 = ChatPromptTemplate.from_messages(
    [
        ("system", system1),
        ("human", "{query}"),
    ]
)

route_chain1 = route_prompt1 | structured_llm_router1

class RouteQuery2(BaseModel):
   """Route a user query to the most relevant next path."""
   next : Literal["add" , "summarize"] = Field(..., description= "Given the user query decide to route it to add or to summarize",)


structured_llm_router2 = GROQ_LLM.with_structured_output(RouteQuery2)

system = """

You are tasked with determining the appropriate action for user queries based on their content. Your goal is to classify each query into one of two categories: "add" or "summarize".

1. Analyze the content of the user query.
2. If the query starts with any of the following keywords or phrases related to summarization: "summary", "Give summary", "Provide summary", "summarize", "summarise", "summarization", "recap", "brief", "extract", "overview", or any similar terms, route the query to "summarize".
3. If the query consists solely of an image path (e.g., "C:/images/photo.jpg") or does not mention summarization at all, route it to "add".
4. Ensure that your classification is based solely on the presence of the keywords or the format of the query, without inferring additional context or meaning.

Provide a clear output indicating the category assigned to the query: "add" or "summarize".

"""
route_prompt2 = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{query}"),
    ]
)

route_chain2 = route_prompt2 | structured_llm_router2

from typing_extensions import TypedDict, List, Dict, Optional
from langgraph.graph import END, StateGraph, START

class ChatState(TypedDict, total=False):
    messages: List[Dict[str, str]]  # Stores chat history
    image_path: Optional[str]  # Stores uploaded image path
    user_query: Optional[str]  # Stores user text input
    extracted_text: Optional[str]  # Stores extracted text
    parsed_data: Optional[Dict]  # Stores parsed invoice data
    generated_sql: Optional[str]  # Stores generated SQL command
    sql_results: Optional[List[Dict]]  # Stores database results
    summary: Optional[str]  # Stores summarized response

def route_query1(state: ChatState):
    """Routes input to either Image Processing or SQL Query Execution."""
    user_query = state["messages"][-1]["content"]  # Get latest user message
    route_chain = route_chain1.invoke({"query": user_query})
    if route_chain.next == "image":
        return "image"
    elif route_chain.next == "Sql":
        return "Sql"


import re
def extract_text_from_images(state: ChatState):
    """Extracts text from an uploaded invoice image."""
    user_query = state["messages"][-1]["content"]

    match = re.search(r'"(.*?)"', user_query)
    
    if match:
        image_path = match.group(1)  
    else:
        image_path = None

    extracted_text = extract_text_from_image(image_path)
    state["messages"].append({"role": "assistant", "content": "Text extracted from the invoice."})

    return {"extracted_text": extracted_text}

def parse_invoice_with_groq(state: ChatState):
    """Parses invoice text into structured data."""
    extracted_text = state["extracted_text"]
    parsed_data = ocr_parser.invoke({"invoice_text": extracted_text})

    state["messages"].append({"role": "assistant", "content": "Invoice data parsed successfully."})
    return {"parsed_data": parsed_data}

def parsed_sql_gen(state: ChatState):
    """Generates SQL commands from parsed invoice data."""
    parsed_data = state["parsed_data"]
    generated_sql = parsed_to_Sql.invoke({"parsed_text" : parsed_data})

    state["messages"].append({"role": "assistant", "content": "Generated SQL query for invoice storage."})
    return {"generated_sql": generated_sql}

def route_query2(state: ChatState):
    """Decides whether to store the parsed invoice or summarize it."""
    user_query = state["messages"][-1]["content"]
    route_chain = route_chain2.invoke({"query": user_query})

    if route_chain.next == "add":
        print("add")
        return "add"
    elif route_chain.next == "summarize":
        print("summarize")
        return "summarize"

def store_invoice(state: ChatState):
    """Stores extracted invoice details into the database."""
    parsed_data = state["parsed_data"]
    store_invoice_in_db(parsed_data)
    state["messages"].append({"role": "assistant", "content": "Invoice successfully stored."})
    return {}

def sql_query_gen(state: ChatState):
    """Generates an SQL query based on user input."""
    user_query = state["messages"][-1]["content"]
    generated_sql = sql_query.invoke({"user_query": user_query})

    state["messages"].append({"role": "assistant", "content": "Generated SQL query."})
    return {"generated_sql": generated_sql}

def execute_sql_query(state: ChatState):
    """Executes a generated SQL query and retrieves results."""
    generated_sql = state["generated_sql"]
    sql_results =   execute_sql(generated_sql)
    state["messages"].append({"role": "assistant", "content": "Executed SQL query successfully."})
    return {"sql_results": sql_results}

def summarise(state: ChatState):
    """Summarizes either invoice data or SQL query results."""
    user_query = state["messages"][-1]["content"]
    if "sql_results" in state:
        summary = summariser.invoke({"user_query": user_query, "result": state['sql_results']})
    else:
        summary = summariser.invoke({"user_query": user_query, "result": state["parsed_data"]})

    state["messages"].append({"role": "assistant", "content": summary})
    return {"summary": summary}

builder = StateGraph(ChatState, input=ChatState, output=ChatState)



# Add Nodes
builder.add_node("route_query1", route_query1)
builder.add_node("extract_text_from_images", extract_text_from_images)
builder.add_node("parse_invoice_with_groq", parse_invoice_with_groq)
builder.add_node("parsed_sql_gen", parsed_sql_gen)
builder.add_node("route_query2", route_query2)
builder.add_node("store_invoice", store_invoice)
builder.add_node("summarise", summarise)
builder.add_node("sql_query_gen", sql_query_gen)
builder.add_node("execute_sql_query", execute_sql_query)

# Define Conditional Routing
builder.add_conditional_edges(START, route_query1,

  {
    "image": "extract_text_from_images",
    "Sql": "sql_query_gen"
}
)

# Invoice Processing Path
builder.add_edge("extract_text_from_images", "parse_invoice_with_groq")
builder.add_edge("parse_invoice_with_groq", "parsed_sql_gen")
builder.add_conditional_edges("parsed_sql_gen", route_query2,

{
    "add": "store_invoice",
    "summarize": "summarise"
})

# SQL Query Path
builder.add_edge("sql_query_gen", "execute_sql_query")
builder.add_edge("execute_sql_query", "summarise")

# Finalizing Graph
builder.add_edge("store_invoice", END)
builder.add_edge("summarise", END)

# Compile Workflow
graph = builder.compile()

# Initialize Chat
state = {"messages": []}

# Start Chat
while True:
    user_message = input("You: ")
    state["messages"].append({"role": "user", "content": user_message})

    if user_message.lower() == "end":
        break

    # Run LangGraph Workflow
    state = graph.invoke(state)

    # Print Responses
    for msg in state["messages"]:
        print(f"{msg['role'].capitalize()}: {msg['content']}")

