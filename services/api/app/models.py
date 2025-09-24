from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from .db import Base

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    size_bytes = Column(Integer)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    pages = Column(Integer, default=0)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    chunk_idx = Column(Integer, index=True)
    page = Column(Integer, default=0)
    text = Column(Text)

    document = relationship("Document", back_populates="chunks")
