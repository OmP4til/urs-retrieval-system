from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import ARRAY
from datetime import datetime
import numpy as np

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    requirements = relationship("Requirement", back_populates="document")

class Requirement(Base):
    __tablename__ = 'requirements'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'))
    requirement_id = Column(String(50))  # e.g., "R1_1"
    text = Column(Text, nullable=False)
    page_number = Column(Integer)
    embedding = Column(ARRAY(Float))  # Store vector as array
    comments = Column(JSON)
    responses = Column(JSON)
    document = relationship("Document", back_populates="requirements")

def init_db(db_url):
    """Initialize database and create tables."""
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return engine

def get_session(engine):
    """Create a new database session."""
    Session = sessionmaker(bind=engine)
    return Session()