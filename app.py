from flask import Flask,request,jsonify
import pandas as pd
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

POSITIVE_WORDS = ['good','great','love','happy','amazing','awesome','excellent','best','beautiful','nice','wonderful','fantastic','enjoy','like','perfect','super','brilliant']
NEGATIVE_WORDS = ['bad','sad','hate','terrible','awful','worst','angry','depressed','lonely','pain','hurt','cry','suicide','kill','die','hopeless','tired','alone','fear','anxious','stress']

def analyze_text(text):
    text_lower = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in text_lower)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
    risk_words = ['suicide','kill myself','want to die','end my life','hopeless']
    is_risk = any(w in text_lower for w in risk_words)
    if is_risk:
        return 'High Risk - Needs Support', 95, pos, neg, True
    if pos > neg:
        conf = 60 + min(pos*10, 35)
        return 'Positive', conf, pos, neg, False
    elif neg > pos:
        conf = 60 + min(neg*10, 35)
        return 'Negative', conf, pos, neg, False
    else:
        return 'Neutral', 55, pos, neg, False

def make_all_graphs(pos_count, neg_count, neu_count, results):
    graphs = {}
    labels = ['Positive', 'Negative', 'Neutral']
    values = [pos_count, neg_count, neu_count]
    colors = ['#22c55e', '#ef4444', '#f59e0b']

    # 1. BAR GRAPH
    plt.figure(figsize=(4,3))
    plt.bar(labels, values, color=colors)
    plt.title('1. Bar Graph - Sentiment Count')
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
    graphs['bar'] = base64.b64encode(buf.read()).decode('utf-8'); plt.close()

    # 2. PIE CHART
    plt.figure(figsize=(4,3))
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('2. Pie Chart - Percentage')
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
    graphs['pie'] = base64.b64encode(buf.read()).decode('utf-8'); plt.close()

    # 3. DONUT CHART
    plt.figure(figsize=(4,3))
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, wedgeprops=dict(width=0.45))
    plt.title('3. Donut Chart')
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
    graphs['donut'] = base64.b64encode(buf.read()).decode('utf-8'); plt.close()

    # 4. HORIZONTAL BAR
    plt.figure(figsize=(4,3))
    plt.barh(labels, values, color=colors)
    plt.title('4. Horizontal Bar Graph')
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
    graphs['hbar'] = base64.b64encode(buf.read()).decode('utf-8'); plt.close()

    # 5. LINE GRAPH - Mood Trend
    scores = []
    for r in results:
        if r['prediction'] == 'Positive': scores.append(1)
        elif r['prediction'] == 'Negative': scores.append(-1)
        else: scores.append(0)
    plt.figure(figsize=(5,3))
    plt.plot(range(1, len(scores)+1), scores, marker='o', color='#8b5cf6', linewidth=2)
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('5. Mood Trend - Line Graph')
    plt.xlabel('Text Number'); plt.ylabel('Mood (1=Happy, -1=Sad)')
    plt.yticks([-1,0,1], ['Negative','Neutral','Positive'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
    graphs['line'] = base64.b64encode(buf.read()).decode('utf-8'); plt.close()
    return graphs

@app.route('/')
def home():
    with open('templates/index.html',encoding='utf-8')as f:return f.read()

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '')
    if not text.strip():
        return jsonify({'error': 'Please enter some text'})
    label, conf, pos, neg, risk = analyze_text(text)
    return jsonify({'text': text, 'prediction': label, 'confidence': conf, 'risk': risk})

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    results = []
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            col = df.columns[0]
            for txt in df[col].astype(str).head(20):
                label, conf, _, _, risk = analyze_text(txt)
                results.append({'text': txt[:100], 'prediction': label, 'confidence': conf, 'risk': risk})
        elif file.filename.endswith('.txt'):
            content = file.read().decode('utf-8')
            lines = [l for l in content.split('\n') if l.strip()]
            for txt in lines[:20]:
                label, conf, _, _, risk = analyze_text(txt)
                results.append({'text': txt[:100], 'prediction': label, 'confidence': conf, 'risk': risk})
        else:
            return jsonify({'error': 'Only .txt and .csv allowed'})
    except Exception as e:
        return jsonify({'error': str(e)})
    pos_c = sum(1 for r in results if r['prediction']=='Positive')
    neg_c = sum(1 for r in results if r['prediction']=='Negative')
    neu_c = len(results) - pos_c - neg_c
    graphs = make_all_graphs(pos_c, neg_c, neu_c, results)
    return jsonify({'results': results, 'total': len(results), 'positive': pos_c, 'negative': neg_c, 'neutral': neu_c, 'graphs': graphs})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)