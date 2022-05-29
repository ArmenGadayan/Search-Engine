from flask import Flask, url_for, render_template, request
from searchEngine import SearchEngine
import time


app = Flask(__name__)

sEngine = SearchEngine()

@app.route("/", methods=["GET", "POST"])
def searchPage():
    global links, sEngine
    sh = ""
    if request.method == "POST":
        sh = request.form["search"]
    start = time.time()
    chosenLinks = sEngine.search(sh)
    end = time.time()
    searchTime = round(end-start, 4)

    return render_template("index.html", content = chosenLinks, search = sh, time = searchTime )

if __name__ == "__main__":
    app.debug = True
    app.run()