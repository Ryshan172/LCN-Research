from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router

app = FastAPI()

# Cors Settings
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True, # For CORS between internal API calls,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include routes
app.include_router(router)