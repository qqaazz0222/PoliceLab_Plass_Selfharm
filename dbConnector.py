import pymysql

host = "" #input host address
port = 3306 #change port(optinal)
user = "" #input database user
password = "" #input database password
charset = "utf8" #change charset(optinal)

def getConnection(dbName):
    conn = pymysql.connect(host=host,
                            port=port,
                            user=user,
                            password=password,
                            database=dbName,
                            charset=charset)
    return conn
