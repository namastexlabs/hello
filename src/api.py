from fastapi import FastAPI, Query
from typing import Optional
import json
from datetime import datetime
from src.database import setup_database

app = FastAPI()

@app.get("/healthz")
async def healthz():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/status")
async def status():
    """Get the current status of the processing queue and total processed files."""
    global global_queue_size
    global total_processed_files
    return {
        "total_files_processed": total_processed_files,
        "queue_size": global_queue_size
    }

@app.get("/search-files")
async def search_files(
    file_id: Optional[int] = Query(None),
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None)
):
    """Search for files in the processed_files table based on various criteria."""
    db_connection = setup_database()
    cursor = db_connection.cursor()
    
    query = "SELECT * FROM processed_files WHERE 1=1"
    params = []

    if file_id:
        query += " AND id = ?"
        params.append(file_id)
    
    if start_time:
        try:
            start_datetime = datetime.strptime(start_time, "%Y-%m-%d_%H-%M-%S")
            query += " AND recorded_at >= ?"
            params.append(start_datetime)
        except ValueError:
            return {"error": "Invalid start_time format. Use YYYY-MM-DD_HH-MM-SS"}
    
    if end_time:
        try:
            end_datetime = datetime.strptime(end_time, "%Y-%m-%d_%H-%M-%S")
            query += " AND recorded_at <= ?"
            params.append(end_datetime)
        except ValueError:
            return {"error": "Invalid end_time format. Use YYYY-MM-DD_HH-MM-SS"}

    cursor.execute(query, params)
    results = cursor.fetchall()
    
    db_connection.close()
    
    return [
        {
            "id": row[0],
            "file_path": row[1],
            "transcription_data": json.loads(row[2]),
            "text": row[3],
            "status": row[4],
            "recorded_at": row[5],
            "processed_at": row[6]
        }
        for row in results
    ]
