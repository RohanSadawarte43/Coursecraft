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



from pydantic import BaseModel
class QuestionAnswerInput(BaseModel):
    prompt: str

@app.post('/give-coursework/')
async def create_buzzword(input_data: QuestionAnswerInput):
    prompt = input_data.prompt
    return prompt

# # from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# from transformers import (
#     AutoConfig,
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig
# )

# import torch
# import torch.nn as nn

# from peft import (
#     LoraConfig,
#     PeftConfig,
#     PeftModel,
#     get_peft_model,
#     prepare_model_for_kbit_training
# )

# PEFT_MODEL = "ironsquire/coursecraft-falcon-7b"

# config = PeftConfig.from_pretrained(PEFT_MODEL)

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# model = AutoModelForCausalLM.from_pretrained(
#     config.base_model_name_or_path,
#     return_dict=True,
#     quantization_config=bnb_config,
#     device_map="auto",
#     trust_remote_code=True
# )

# tokenizer=AutoTokenizer.from_pretrained(config.base_model_name_or_path)
# tokenizer.pad_token = tokenizer.eos_token

# model = PeftModel.from_pretrained(model, PEFT_MODEL)


# generation_config = model.generation_config
# generation_config.max_new_tokens = 200
# generation_config.temperature = 0.7
# generation_config.top_p = 0.7
# generation_config.num_return_sequences = 1
# generation_config.pad_token_id = tokenizer.eos_token_id
# generation_config.eos_token_id = tokenizer.eos_token_id
# # model_name = "ironsquire/coursecraft-falcon-7b"
# # model = AutoModelForQuestionAnswering.from_pretrained(model_name)
# # tokenizer = AutoTokenizer.from_pretrained(model_name)


# # %%time
# device = "cuda:0"

# prompt = """
# <human>: Please suggest a course work for a course on object-oriented data structures in C++.
# <assistant>:
# """.strip()

# encoding = tokenizer(prompt, return_tensors="pt").to(device)
# with torch.inference_mode():
#     outputs = model.generate(
#         input_ids = encoding.input_ids,
#         attention_mask = encoding.attention_mask,
#         generation_config = generation_config
#     )

# print(tokenizer.decode(outputs[0], skip_special_tokens=True))



@app.post("/question-answering")
def question_answering(input_data: QuestionAnswerInput):
    prompt = input_data.prompt

    inputs = tokenizer(prompt, return_tensors="pt")
    # start_logits, end_logits = model(**inputs).logits.split(1, dim=-1)

    # start_index = start_logits.argmax().item()
    # end_index = end_logits.argmax().item()

    # answer = tokenizer.decode(inputs["input_ids"][0][start_index:end_index+1])

    return {"answer": inputs}


# To run locally
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)