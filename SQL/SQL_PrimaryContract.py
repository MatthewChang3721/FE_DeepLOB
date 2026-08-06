import csv
import pymysql
from datetime import datetime,timedelta

INSTRUMENT_ID = 'IM%'
TODAY = datetime.now().date()
DATE = TODAY - timedelta(days = 50)
TODAY = TODAY.strftime(r'%Y-%m-%d')
DATE = DATE.strftime(r'%Y-%m-%d')


conn = pymysql.connect(
        host = '192.168.1.118',
        user = 'replay1',
        password = 'replayMvt*',
        database = 'mvtdb',
        port = 3306,
        ssl_disabled=True
    )
sql = f"""
        SELECT
        TradingDay,
        InstrumentID 
    FROM
        `TradeableInstrument` 
    WHERE
        TradingDay >= '{DATE}' 
        AND TradingDay != '{TODAY}' 
        AND InstrumentID LIKE '{INSTRUMENT_ID}' 
        AND TradeableType = 'PRIMARY' 
    ORDER BY
        TradingDay ASC
    """

try: 
    with conn.cursor() as cursor:
        cursor.execute(sql)
        column_names = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        with open('Contract.csv', 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            # 写入表头
            writer.writerow(column_names)
            
            # 批量写入所有数据行（如果数据量巨大，可改用循环逐行写入）
            writer.writerows(rows)

finally:
    conn.close()