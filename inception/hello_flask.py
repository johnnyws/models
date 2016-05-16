from flask import Flask
from flask import request
app = Flask(__name__)

@app.route("/image", methods = ['POST'])
def image():
  image_data = request.data
  print(request.headers.get("Content-Type"))
  return str(request.headers)

if __name__ == "__main__":
    app.run(debug=True)
