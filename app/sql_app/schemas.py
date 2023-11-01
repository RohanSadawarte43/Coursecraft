# build a schema using pydantic
from pydantic import BaseModel


class Course(BaseModel):
    course_name: str
    course_code: str
    college: str
    field: str
    major: str
    instructor: str
    description: str

    class Config:
        orm_mode = True



class Buzzword(BaseModel):
    buzzword: str
    description: str

    class Config:
        orm_mode = True