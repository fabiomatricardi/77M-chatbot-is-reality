########################################################################
#    https://huggingface.co/nicholasKluge/Aira-2-355M                  #
#    https://huggingface.co/Felladrin/gguf-Aira-2-355M/tree/main       #
########################################################################
import streamlit as st
import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import datetime
from rich.markdown import Markdown
import warnings
warnings.filterwarnings(action='ignore')
import datetime
from rich.console import Console
console = Console(width=90)
import tiktoken
import random
import string
from time import sleep

encoding = tiktoken.get_encoding("r50k_base") #context_count = len(encoding.encode(yourtext))

#AVATARS  👷🐦  🥶🌀
av_us = 'man.png'  #"🦖"  #A single emoji, e.g. "🧑‍💻", "🤖", "🦖". Shortcodes are not supported.
av_ass = 'ass.png'

# Set the webpage title
st.set_page_config(
    page_title="Your LocalGPT with LaMini-Flan-T5-77M",
    page_icon="🦙",
    layout="wide")

convHistory = ''
#modelfile = "MBZUAI/LaMini-Flan-T5-248M"
repetitionpenalty = 1.3
contextlength=512
logfile = 'LaMini77M_logs.txt'


@st.cache_resource 
def create_chat():   
    LaMini = './model77M/'
    tokenizer = AutoTokenizer.from_pretrained(LaMini)
    model = AutoModelForSeq2SeqLM.from_pretrained(LaMini,
                                                device_map='cpu',
                                                torch_dtype=torch.float32)
    llm = pipeline('text2text-generation', 
                    model = model,
                    tokenizer = tokenizer,
                    max_length = 512, 
                    do_sample=True,
                    temperature=0.35,
                    top_p=0.8,
                    repetition_penalty = 1.3,
                    top_k = 4,
                    penalty_alpha = 0.6
                    )
    return llm

def writehistory(filename,text):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

def genRANstring(n):
    """
    n = int number of char to randomize
    """
    N = n
    res = ''.join(random.choices(string.ascii_uppercase +
                                string.digits, k=N))
    return res


# Create a header element
st.image('logo77m.png')

# create THE SESSIoN STATES
if "logfilename" not in st.session_state:
## Logger file
    logfile = f'{genRANstring(5)}_log.txt'
    st.session_state.logfilename = logfile
    #Write in the history the first 2 sessions
    writehistory(st.session_state.logfilename,f'{str(datetime.datetime.now())}\n\nYour own LocalGPT with 🦙 LaMini-77M\n---\n🧠🫡: You are a helpful assistant.')    
    writehistory(st.session_state.logfilename,f'🌀: How may I help you today?')

if "limiter" not in st.session_state:
    st.session_state.limiter = 0

if "numoftokens" not in st.session_state:
    st.session_state.numoftokens = 0

if "bufstatus" not in st.session_state:
    st.session_state.bufstatus = "**:green[Good]**"

if "prompt" not in st.session_state:
    st.session_state.prompt = ''

if "maxlength" not in st.session_state:
    st.session_state.maxlength = 350

# Point to the local server
llm = create_chat()
 
# CREATE THE SIDEBAR
with st.sidebar:
    st.image('assistant.png', width=200)
    st.session_state.maxlength = st.slider('Max prompt:', min_value=100, max_value=400, value=350, step=10)
    n_tokens = st.markdown(f"Prompt Tokens: {st.session_state.numoftokens}")
    st.markdown(f"Buffer status: {st.session_state.bufstatus}")
    st.markdown(f"**Logfile**: {st.session_state.logfilename}")
    btnClear = st.button("Clear History",type="primary", use_container_width=True)

# We store the conversation in the session state.
# This will be used to render the chat conversation.
# We initialize it with the first message we want to be greeted with.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are LaMini-Flan-T5, a helpful assistant. You reply only to the user questions. You always reply in the language of the instructions.",},         
        {"role": "user", "content": "Hi, I am Fabio."},
        {"role": "assistant", "content": "Hi there Fabio, I am LaMini-Flan-T5: with my 77M parameters I can be useful to you. how may I help you today?"}
    ]

def clearHistory():
    st.session_state.messages = [
        {"role": "system", "content": "You are LaMini-Flan-T5, a helpful assistant. You reply only to the user questions. You always reply in the language of the instructions.",},         
        {"role": "user", "content": "Hi, I am Fabio."},
        {"role": "assistant", "content": "Hi there Fabio, I am LaMini-Flan-T5: with my 77M parameters I can be useful to you. how may I help you today?"}
    ]
    st.session_state.len_context = len(st.session_state.messages)
if btnClear:
      clearHistory()  
      st.session_state.len_context = len(st.session_state.messages)

# We loop through each message in the session state and render it as
# a chat message.
for message in st.session_state.messages[1:]:
    if message["role"] == "user":
        with st.chat_message(message["role"],avatar=av_us):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"],avatar=av_ass):
            st.markdown(message["content"])


def countTokens():
    encoding = tiktoken.get_encoding("r50k_base") #context_count = len(encoding.encode(yourtext))
    st.session_state.numoftokens = len(encoding.encode(st.session_state.prompt))
    print(st.session_state.numoftokens)
    if st.session_state.numoftokens > st.session_state.maxlength:
        n_tokens.markdown(f"**⚠️ Prompt Tokens: {st.session_state.numoftokens}**")
        return False
    else:
        n_tokens.markdown(f"**✅ Prompt Tokens: {st.session_state.numoftokens}**")
        return True
    
# We take questions/instructions from the chat input to pass to the LLM
if user_prompt := st.chat_input("Your message here. Shift+Enter to add a new line", key="user_input"):
    st.session_state.prompt = user_prompt 
    if countTokens():
        # Add our input to the session state
        st.session_state.messages.append(
            {"role": "user", "content": user_prompt}
        )

        # Add our input to the chat window
        with st.chat_message("user", avatar=av_us):
            st.markdown(user_prompt)
            writehistory(st.session_state.logfilename,f'👷: {user_prompt}')

        
        with st.chat_message("assistant",avatar=av_ass):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                response = ''
                conv_messages = []
                st.session_state.len_context = len(st.session_state.messages) 
                st.session_state.bufstatus = "**:green[Good]**"
                full_response = ""
                completion = llm(user_prompt)[0]['generated_text']
                for chunk in completion:
                            full_response += chunk
                            sleep(0.012)
                            message_placeholder.markdown(full_response + "🌟")                                           
                message_placeholder.markdown(full_response)
                writehistory(st.session_state.logfilename,f'🌟: {full_response}\n\n---\n\n') 
    else:
            # Add our input to the session state
        st.session_state.messages.append(
            {"role": "user", "content": user_prompt}
        )

        # Add our input to the chat window
        with st.chat_message("user", avatar=av_us):
            st.markdown(user_prompt)
            writehistory(st.session_state.logfilename,f'👷: {user_prompt}')        
        with st.chat_message("assistant",avatar=av_ass):
            message_placeholder = st.empty()
            st.session_state.len_context = len(st.session_state.messages) 
            st.session_state.bufstatus = "**:red[BAD]**"
            full_response = ""
            completion = "⚠️ Your prompt is too long for me. Shorten it, please."
            for chunk in completion:
                        full_response += chunk
                        sleep(0.012)
                        message_placeholder.markdown(full_response + "🌟")                                           
            message_placeholder.markdown(full_response)
            writehistory(st.session_state.logfilename,f'🌟: {full_response}\n\n---\n\n')             
    
    # Add the response to the session state
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
    st.session_state.len_context = len(st.session_state.messages)