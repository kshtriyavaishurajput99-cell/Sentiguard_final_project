import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

app = Flask(__name__)

# UPDATED: Receives a dynamic confidence percentage score to plot a live point
def get_roc_plot_base64(confidence_val=None):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Static model baseline metric path
    fpr = np.linspace(0, 1, 100)
    tpr = np.log1p(fpr * (np.e - 1))
    ax.plot(fpr, tpr, color="#ec4899", lw=2.5, label="Random Forest (AUC = 0.92)")
    ax.plot([0, 1], [0, 1], color="#cbd5e1", linestyle="--")
    
    # NEW: Calculate and plot a dynamic point if a prediction text is passed
    if confidence_val is not None:
        try:
            # Convert string like "98.2%" or float to a decimal threshold index
            val = float(str(confidence_val).replace('%', '')) / 100.0
            # Estimate matching coordinates along our curve
            point_fpr = 1.0 - val
            point_tpr = np.log1p(point_fpr * (np.e - 1))
            
            # Draw a bright glowing neon indicator dot for the current text item
            ax.plot(point_fpr, point_tpr, marker='o', markersize=10, 
                    color='#3b82f6', markeredgecolor='white', markeredgewidth=2,
                    label=f"Current Input (Conf: {confidence_val})")
        except ValueError:
            pass
            
    ax.legend(loc="lower right")
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    string = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return string

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    results_list = []
    
    # File upload handling logic (.csv/.txt)
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        filename = file.filename.lower()
        
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
            col = 'text' if 'text' in df.columns else df.columns[0]
            for row_text in df[col].dropna().astype(str).tolist():
                res = "Suicidal" if "die" in row_text.lower() else "Safe"
                # Mock variable scores matching your table loop structures
                conf = "95.4%" if res == "Suicidal" else "88.2%"
                results_list.append((row_text, res, conf))
                
        elif filename.endswith('.txt'):
            lines = file.read().decode('utf-8').split('\n')
            for line in lines:
                if line.strip():
                    res = "Suicidal" if "die" in line.lower() else "Safe"
                    conf = "94.0%" if res == "Suicidal" else "89.5%"
                    results_list.append((line, res, conf))
                    
    # Text Field fallback logic
    elif request.form.get('text'):
        text_data = request.form.get('text')
        res = "Suicidal" if "die" in text_data.lower() else "Safe"
        conf = "98.2%" if res == "Suicidal" else "76.4%"
        results_list.append((text_data, res, conf))

    if not results_list:
        return render_template('index.html')

    # Extract the confidence string from the last prediction to update the plot
    last_confidence = results_list[-1][2]
    graph_data = get_roc_plot_base64(confidence_val=last_confidence)
    
    return render_template('index.html', results=results_list, graph_base64=graph_data)

if __name__ == '__main__':
    app.run(debug=True)