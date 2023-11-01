import uvicorn
from fastapi import FastAPI
from fastapi_sqlalchemy import DBSessionMiddleware, db

from fastapi.middleware.cors import CORSMiddleware

from app.sql_app.schemas import Course as SchemaCourse
from app.sql_app.schemas import Buzzword as SchemaBuzzword

from app.sql_app.models import Course as ModelCourse
from app.sql_app.models import Buzzword as ModelBuzzword

import os
from dotenv import load_dotenv

load_dotenv('.env')


app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# to avoid csrftokenError
app.add_middleware(DBSessionMiddleware, db_url=os.environ['DATABASE_URL'])

@app.post('/course/', response_model=SchemaCourse)
async def create_course(course: SchemaCourse):
    db_course = ModelCourse(**course.dict())
    db.session.add(db_course)
    db.session.commit()
    return db_course

@app.get('/courses/')
async def course():
    course = db.session.query(ModelCourse).all()
    return course

@app.post('/buzzword/', response_model=SchemaBuzzword)
async def create_buzzword(buzzword: SchemaBuzzword):
    db_buzzword = ModelBuzzword(**buzzword.dict())
    db.session.add(db_buzzword)
    db.session.commit()
    return db_buzzword

@app.get('/buzzwords/')
async def buzzword():
    buzzwords = db.session.query(ModelBuzzword).all()
    return buzzwords

# To run locally
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)