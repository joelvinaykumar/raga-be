import sqlite3
from datetime import datetime

from fastapi import HTTPException

DB_NAME = "rag_app.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_application_logs():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     session_id TEXT,
                     user_query TEXT,
                     gpt_response TEXT,
                     model TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def insert_application_logs(session_id, user_query, gpt_response, model):
    conn = get_db_connection()
    conn.execute('INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
                 (session_id, user_query, gpt_response, model))
    conn.commit()
    conn.close()

def get_chat_history(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at', (session_id,))
    messages = []
    for row in cursor.fetchall():
        messages.extend([
            {"role": "user", "content": row['user_query']},
            {"role": "assistant", "content": row['gpt_response']}
        ])
    conn.close()
    return messages

def create_document_store():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     filename TEXT,
                     filesize INT,
                     session_id VARCHAR(10),
                     upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def insert_document_record(filename, filesize, session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO document_store (filename, filesize, session_id) VALUES (?, ?, ?)', (filename, filesize, session_id))
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id

def delete_document_record(file_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
    conn.commit()
    conn.close()
    return True

def get_all_documents(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''SELECT id, filename, filesize, upload_timestamp 
        FROM document_store 
        WHERE session_id = ? 
        ORDER BY upload_timestamp DESC''',
        (session_id,)
    )
    documents = cursor.fetchall()
    conn.close()
    return [dict(doc) for doc in documents]

def get_all_sessions():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''SELECT session_id, user_query
                    FROM (
                    SELECT session_id, user_query,
                            ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY created_at ASC) AS rn
                    FROM application_logs
                    ) t
                    WHERE rn = 1''')
    sessions = cursor.fetchall()
    conn.close()
    return [dict(session) for session in sessions]

def delete_session(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''SELECT session_id FROM application_logs
                    WHERE session_id = ?;
                    ''', (session_id,))
    session = cursor.fetchall()
    if session:
        cursor.execute('''DELETE FROM application_logs
                    WHERE session_id = ?;
                    ''', (session_id,))
        conn.commit()
        conn.close()
        return {"message": "Deleted successfully"}
    else:
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")
        # return {"message": "Session Not Found"}

# Initialize the database tables
create_application_logs()
create_document_store()