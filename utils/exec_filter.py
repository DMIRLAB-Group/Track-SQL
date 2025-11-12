import sqlite3, os
from func_timeout import func_set_timeout, FunctionTimedOut

def get_cursor_from_path(sqlite_path):
    try:
        if not os.path.exists(sqlite_path):
            print("Openning a new connection %s" % sqlite_path)
        connection = sqlite3.connect(sqlite_path, check_same_thread = False)
    except Exception as e:
        print(sqlite_path)
        raise e
    connection.text_factory = lambda b: b.decode(errors="ignore")
    cursor = connection.cursor()
    return cursor

@func_set_timeout(120)
def execute_sql(cursor, sql):
    cursor.execute(sql)

    return cursor.fetchall()

def exec_filter(pred_sql, db_file_path):
        cursor = get_cursor_from_path(db_file_path)
        try:
            # Note: execute_sql will be success for empty string
            assert len(pred_sql) > 0, "pred sql is empty!"

            results = execute_sql(cursor, pred_sql)
            # self.results.append(results)
            # if the current sql has no execution error, we record and return it
            cursor.close()
            cursor.connection.close()
        except Exception as e:
            print(pred_sql)
            print(e)
            cursor.close()
            cursor.connection.close()
            return False
        except FunctionTimedOut as fto:
            print(pred_sql)
            print(fto)
            del cursor
            return False
        return True