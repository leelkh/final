import sqlite3
import pandas as pd

db_path = "user_info.db"

try:
    # 嘗試連接資料庫
    db = sqlite3.connect(db_path)
    cursor = db.cursor()
    # 創建資料表 COMPANY
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS USER
        (User_ID    CHAR(50) PRIMARY KEY,
        account  CHAR(50) UNIQUE,
        password  CHAR(50))
    ''')
    db.commit()
    print("Connected to the database.")
except sqlite3.Error as e:
    print("Error connecting to the database:", e)
finally:
    if db:
        db.close()

def run_query(query):
    return pd.read_sql_query(query, db)#創造一個function

def droptable():
    cursor.execute('''
        DROP TABLE COMPANY
    ''')
    db.commit()

def check_credentials(username, password):
    db_path = "user_info.db"
    db = sqlite3.connect(db_path)
    cursor = db.cursor()

    # 查詢是否有符合的帳號
    cursor.execute("SELECT * FROM USER WHERE account=?", (username,))
    user = cursor.fetchone()

    if user:
        # 如果找到帳號，檢查密碼是否正確
        if user[2] == password:  # 假設密碼欄位在第三個位置
            db.close()
            return("Credentials are correct.")
        else:
            db.close()
            return("Incorrect password.")
    else:
        db.close()
        return("Username not found.")
def register_account(username, password):
    db_path = "user_info.db"
    db = sqlite3.connect(db_path)
    cursor = db.cursor()

    # 查詢是否已經存在相同的帳號
    cursor.execute("SELECT * FROM COMPANY WHERE account=?", (username,))
    existing_user = cursor.fetchone()

    if existing_user:
        print("Username already exists. Please choose another username.")
    else:
        # 取得下一個可用的 key 值
        cursor.execute("SELECT MAX(User_ID) FROM COMPANY")
        last_id = cursor.fetchone()[0]
        next_id = last_id + 1 if last_id is not None else 1
            
        # 將新帳號和密碼儲存到資料庫中，手動指定 key 值
        cursor.execute("INSERT INTO COMPANY (User_ID, account, password) VALUES (?, ?, ?)", (next_id, username, password))
        db.commit()
        print("Account successfully registered.")