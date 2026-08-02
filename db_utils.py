import sqlite3
import os
import tempfile
from fastapi import HTTPException


def _get_db_path() -> str:
    default = os.path.join(os.getcwd(), "rag_app.db")
    try:
        with open(default, "a"):
            pass
        return default
    except OSError:
        return os.path.join(tempfile.gettempdir(), "rag_app.db")


DB_NAME = _get_db_path()

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

def create_rag_store():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS rag_store
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     rag_id TEXT,
                     name TEXT,
                     description TEXT,
                     top_k REAL,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def create_rag_document_store():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS rag_document_store
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     rag_id TEXT,
                     filename TEXT,
                     filesize INT,
                     upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
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
    cursor.execute('''SELECT s.session_id, s.user_query, sk.knowledgebase_id
                    FROM (
                    SELECT session_id, user_query,
                            ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY created_at ASC) AS rn
                    FROM application_logs
                    ) s
                    LEFT JOIN session_knowledgebase sk ON s.session_id = sk.session_id
                    WHERE s.rn = 1''')
    sessions = cursor.fetchall()
    conn.close()
    return [dict(session) for session in sessions]

def delete_session(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT session_id FROM application_logs WHERE session_id = ?;', (session_id,))
    session = cursor.fetchone()
    if session:
        cursor.execute('DELETE FROM application_logs WHERE session_id = ?;', (session_id,))
        conn.commit()
        conn.close()
        return {"message": "Deleted successfully"}
    else:
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")

# RAG management helper functions

def create_rag_entry(rag_id: str, name: str, description: str, top_k: float):
    conn = get_db_connection()
    conn.execute('INSERT INTO rag_store (rag_id, name, description, top_k) VALUES (?, ?, ?, ?)',
                 (rag_id, name, description, top_k))
    conn.commit()
    conn.close()

def get_rag_entry(rag_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM rag_store WHERE rag_id = ?', (rag_id,))
    rag = cursor.fetchone()
    conn.close()
    return rag

def delete_rag_entry(rag_id: str):
    conn = get_db_connection()
    conn.execute('DELETE FROM rag_store WHERE rag_id = ?', (rag_id,))
    conn.commit()
    conn.close()
    return True

def list_rags():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM rag_store ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# Document store for each RAG

def insert_rag_document(rag_id: str, filename: str, filesize: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO rag_document_store (rag_id, filename, filesize) VALUES (?, ?, ?)',
                 (rag_id, filename, filesize))
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return doc_id

def get_rag_documents(rag_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM rag_document_store WHERE rag_id = ? ORDER BY upload_timestamp DESC', (rag_id,))
    docs = cursor.fetchall()
    conn.close()
    return [dict(doc) for doc in docs]

def delete_rag_document(doc_id: int):
    conn = get_db_connection()
    conn.execute('DELETE FROM rag_document_store WHERE id = ?', (doc_id,))
    conn.commit()
    conn.close()
    return True


def create_session_knowledgebase():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS session_knowledgebase
                    (session_id TEXT PRIMARY KEY,
                     knowledgebase_id TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def set_session_knowledgebase(session_id: str, knowledgebase_id: str | None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT session_id FROM session_knowledgebase WHERE session_id = ?', (session_id,))
    existing = cursor.fetchone()
    if existing:
        conn.execute('UPDATE session_knowledgebase SET knowledgebase_id = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?',
                     (knowledgebase_id, session_id))
    else:
        conn.execute('INSERT INTO session_knowledgebase (session_id, knowledgebase_id) VALUES (?, ?)',
                     (session_id, knowledgebase_id))
    conn.commit()
    conn.close()

def get_session_knowledgebase(session_id: str) -> str | None:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT knowledgebase_id FROM session_knowledgebase WHERE session_id = ?', (session_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row['knowledgebase_id']
    return None

# Initialize the database tables
create_application_logs()
create_document_store()
create_rag_store()
create_rag_document_store()
create_session_knowledgebase()