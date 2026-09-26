from fastapi import FastAPI
from src.controller.service_controller import inspection_router, communication_router


app = FastAPI()
app.include_router(inspection_router, tags=['DISKS INSPECTION'], prefix="/ai")
app.include_router(communication_router, tags=['PLC COMMUNICATION'], prefix="/communication")
