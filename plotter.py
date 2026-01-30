import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web server
import matplotlib.pyplot as plt
import math
import re

def clean_expression(expr: str) -> str:
    """Clean the expression from LLM output - remove code blocks, quotes, etc."""
    # Remove markdown code blocks (handle ```python, ```, or just ```)
    expr = re.sub(r'```[a-zA-Z]*\s*\n?', '', expr)
    expr = re.sub(r'```\s*', '', expr)
    expr = re.sub(r'`', '', expr)
    
    # Remove quotes if the entire expression is quoted
    expr = expr.strip()
    if (expr.startswith('"') and expr.endswith('"')) or (expr.startswith("'") and expr.endswith("'")):
        expr = expr[1:-1]
    
    # Remove any leading/trailing whitespace and newlines
    expr = expr.strip()
    
    # Remove common prefixes like "y = " or "f(x) = "
    expr = re.sub(r'^(y\s*=\s*|f\(x\)\s*=\s*)', '', expr, flags=re.IGNORECASE)
    
    # Remove any remaining newlines
    expr = expr.replace('\n', ' ').strip()
    
    return expr

def plot_expression(expr: str, output_path="static/plot.png"):
    # Clean the expression first
    expr = clean_expression(expr)
    
    # Validate syntax before trying to eval
    try:
        compile(expr, '<string>', 'eval')
    except SyntaxError as e:
        raise SyntaxError(f"Invalid expression syntax: {expr}. Error: {str(e)}")
    
    x = np.linspace(-10, 10, 400)
    # Make numpy functions and math functions available for eval
    try:
        y = eval(expr, {"np": np, "numpy": np, "math": math, "x": x})
    except NameError as e:
        raise NameError(f"Unknown function or variable in expression: {expr}. Error: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error evaluating expression '{expr}': {str(e)}")
    
    # Check if y is a valid array
    if not isinstance(y, np.ndarray):
        try:
            y = np.array(y)
        except:
            raise ValueError(f"Expression '{expr}' did not produce a valid array")
    
    if len(y) != len(x):
        raise ValueError(f"Expression '{expr}' produced array of length {len(y)}, expected {len(x)}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"y = {expr}")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
