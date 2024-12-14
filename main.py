from fastapi import FastAPI
import random

app = FastAPI()

@app.get('/')
async def root():
    return {'example': 'this is an example', 'data': 0}