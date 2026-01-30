from flask import Flask, render_template, request
from llm_utils import request_to_math_expr
from plotter import plot_expression
import traceback

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    plot_url = None
    expr = ""
    error = None

    if request.method == "POST":
        try:
            user_text = request.form["expression"]
            expr = request_to_math_expr(user_text)
            plot_expression(expr, output_path="static/test.png")
            plot_url = "/static/test.png"
        except Exception as e:
            error = str(e)
            traceback.print_exc()

    return render_template("index.html", plot_url=plot_url, expr=expr, error=error)

if __name__ == "__main__":
    app.run(debug=True)
