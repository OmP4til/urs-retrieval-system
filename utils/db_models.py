
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, DateTime, JSON, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from pgvector.sqlalchemy import Vector
from datetime import datetime

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False, unique=True)
    upload_date = Column(DateTime, default=datetime.utcnow)
    requirements = relationship("Requirement", back_populates="document", cascade="all, delete-orphan")

class Requirement(Base):
    __tablename__ = 'requirements'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'))
    requirement_id = Column(String(50))
    text = Column(Text, nullable=False)
    page_number = Column(Integer)
    embedding = Column(Vector(384))  # 384 is the dimension for all-MiniLM-L6-v2
    comments = Column(JSON, default=list)
    responses = Column(JSON, default=list)
    document = relationship("Document", back_populates="requirements")

def init_db(db_url):
    """Initialize database and create tables."""
    engine = create_engine(db_url, pool_pre_ping=True)
    
    # Create pgvector extension if not exists
    with engine.connect() as conn:
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))
        conn.commit()
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    # Create index for faster similarity search
    with engine.connect() as conn:
        # Check if index exists
        result = conn.execute(text("""
            SELECT 1 FROM pg_indexes 
            WHERE indexname = 'idx_requirements_embedding'
        """)).fetchone()
        
        if not result:
            conn.execute(text("""
                CREATE INDEX idx_requirements_embedding 
                ON requirements 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """))
            conn.commit()
    
    return engine

def get_session(engine):
    """Create a new database session."""
    Session = sessionmaker(bind=engine)
    return Session()