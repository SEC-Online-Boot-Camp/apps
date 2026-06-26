import socket

import psycopg2
from azure.storage.blob import BlobServiceClient
from flask import Flask, Response

app = Flask(__name__)

DB = {
    "host": "pg-sample.postgres.database.azure.com",
    "user": "pgadmin",
    "password": "Pg_Admin",
    "dbname": "shopdb",
    "sslmode": "require",
}

STORAGE_CONNECTION_STRING = "<Azure Storage接続文字列>"
CONTAINER_NAME = "container-sample"


@app.route("/")
def index():
    # データベースに接続して商品一覧を取得する
    db_ok = False
    rows = ""
    try:
        conn = psycopg2.connect(**DB)
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, name, category, price, image_file FROM products ORDER BY id"
            )
            products = cursor.fetchall()
        conn.close()
        db_ok = True
        rows = "".join(
            f"<tr>"
            f"<td>{p[0]}</td>"
            f"<td>{p[1]}</td>"
            f"<td>{p[2]}</td>"
            f"<td>{p[3]}円</td>"
            f'<td><img src="/image/{p[4]}" width="120"></td>'
            f"</tr>"
            for p in products
        )
    except Exception:
        pass

    # ストレージの接続確認
    storage_ok = False
    try:
        client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
        client.get_service_properties()
        storage_ok = True
    except Exception:
        pass

    style = (
        "<style>"
        "body{font-family:sans-serif;text-align:center;padding:100px 20px 40px}"
        "table{border-collapse:collapse;margin:20px auto}th,td{padding:8px;text-align:left}"
        "</style>"
    )
    body = style + f"<h1>{socket.gethostname()}</h1>"
    if db_ok:
        body += (
            "<h2>商品一覧</h2>"
            '<table border="1">'
            "<tr><th>ID</th><th>名称</th><th>カテゴリ</th><th>価格</th><th>画像</th></tr>"
            + rows
            + "</table>"
        )
    else:
        body += "<p>データベース未接続</p>"
    if not storage_ok:
        body += "<p>ストレージ未接続</p>"
    return body


@app.route("/image/<name>")
def image(name):
    # ストレージから商品画像を取得して返す
    try:
        blob_client = BlobServiceClient.from_connection_string(
            STORAGE_CONNECTION_STRING
        )
        container = blob_client.get_container_client(CONTAINER_NAME)
        data = container.download_blob(name).readall()
        return Response(data, mimetype="image/png")
    except Exception:
        return Response("画像を取得できません", status=404)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
