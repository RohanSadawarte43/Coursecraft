from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import ARRAY

Base  = declarative_base()

class Course(Base):
    __tablename__ = 'course'
    id = Column(Integer, primary_key=True)
    course_name = Column(String)
    course_code = Column(String)
    college = Column(String)
    field = Column(String)
    major = Column(String)
    instructor = Column(String)
    description = Column(String)
    time_created = Column(DateTime(timezone=True), server_default=func.now())
    time_updated = Column(DateTime(timezone=True), onupdate=func.now())

class Buzzword(Base):
    __tablename__ = 'buzzword'
    id = Column(Integer, primary_key=True)
    buzzword = Column(String, unique=True)
    description = Column(String)
    time_created = Column(DateTime(timezone=True), server_default=func.now())
    time_updated = Column(DateTime(timezone=True), onupdate=func.now())