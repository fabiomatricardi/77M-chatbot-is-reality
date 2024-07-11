# 77M-chatbot-is-reality
Run encoder-decoder LaMini-Flan-T5 with streamlit

test the powers of encoder-decoder models

<img src='https://github.com/fabiomatricardi/77M-chatbot-is-reality/raw/main/encoder-decoder-comparisons.png' width=800>

### How to use it
works with Python 3.11+ <br>
tested on Windows 11 machine
- Create a new folder for the project (mine is `LaminiST`)
- Create a virtual environment and activate it

```
python -m venv venv
venv\Scripts\activate
pip install streamlit==1.36.0 transformers torch langchain langchain-community tiktoken accelerate
```
  
#### Clone the repo into your project directory
```
git clone https://github.com/fabiomatricardi/77M-chatbot-is-reality.git .
```

- From the terminal, with the venv activated run
```
streamlit run st-laminiChat.py
```


<img src='https://github.com/fabiomatricardi/77M-chatbot-is-reality/raw/main/Streamlit-intrface.png' width=800>
