import mysql.connector


def log_result(predictions, db_config):
    cnx = mysql.connector.connect(**db_config)
    cursor = cnx.cursor()

    query = "INSERT INTO results (prediction) VALUES (%s)"
    cursor.execute(query, (predictions[0],))

    cnx.commit()
    cursor.close()
    cnx.close()
