import pymysql

host = "172.30.1.46"
port = 3306
user = "root"
password = "mhncity@364"
charset = "utf8"

def getConnection(dbName):
    conn = pymysql.connect(host=host,
                            port=port,
                            user=user,
                            password=password,
                            database=dbName,
                            charset=charset)
    return conn
