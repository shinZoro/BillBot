from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Date
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError

# Initialize database
DATABASE_URL = "sqlite:///invoices.db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# Define Invoice Table
class Invoice(Base):
    __tablename__ = "invoices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    invoice_id = Column(String, unique=True, nullable=False)
    seller_name = Column(String, nullable=True)  # Allow NULL values
    amount_due = Column(Float, nullable=False, default=0.0)  # Default to 0.0
    due_date = Column(Date, nullable=True)  # Allow NULL values
    tax_percent = Column(Float, nullable=False, default=0.0)  # Default to 0.0
    status = Column(String, nullable=False, default="Unpaid")  # Default "Unpaid"

# Create the table if not already created
Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()