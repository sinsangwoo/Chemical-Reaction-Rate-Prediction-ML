"""Database models and connection for Chemical Reaction ML Platform.

Uses SQLAlchemy ORM with PostgreSQL/SQLite support.
Stores users, predictions, and analytics data.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# Database URL from environment or default to SQLite
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./chemical_ml.db"
)

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class
Base = declarative_base()


class User(Base):
    """User model for authentication.
    
    Attributes:
        id: Primary key
        email: Unique email address
        username: Unique username
        hashed_password: Bcrypt hashed password
        is_active: Account status
        is_verified: Email verification status
        created_at: Registration timestamp
        predictions: Related predictions
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="user")


class Prediction(Base):
    """Prediction history model.
    
    Attributes:
        id: Primary key
        user_id: Foreign key to User
        reactants: List of reactant SMILES
        products: List of product SMILES
        conditions: Reaction conditions (JSON)
        model_used: Model type
        prediction_value: Predicted rate
        uncertainty_data: Uncertainty estimates (JSON)
        created_at: Prediction timestamp
        user: Related user
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Reaction data
    reactants = Column(JSON, nullable=False)
    products = Column(JSON, nullable=False)
    conditions = Column(JSON, nullable=False)
    
    # Prediction results
    model_used = Column(String, nullable=False)
    prediction_value = Column(Float, nullable=False)
    uncertainty_data = Column(JSON)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship("User", back_populates="predictions")


class APIKey(Base):
    """API key model for programmatic access.
    
    Attributes:
        id: Primary key
        user_id: Foreign key to User
        key: Hashed API key
        name: Friendly name for key
        is_active: Key status
        created_at: Creation timestamp
        last_used_at: Last usage timestamp
    """
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime)


def get_db():
    """Get database session.
    
    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    # Create tables
    print("Creating database tables...")
    init_db()
    print("✓ Database initialized successfully")
