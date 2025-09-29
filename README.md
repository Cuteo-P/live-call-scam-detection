
# Live scam call detection model (POC)

**Predicts if the caller is a scammer or not using voice input**

## How does it work?

**The voice is transcribed into text using ASR [openai/whisper-medium.en](https://huggingface.co/openai/whisper-medium.en) model**
**Transcribed text is then feeded as chunks into finetuned [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) to get embeddings**
**The embeddings are then feeded into trained classifier that returns probability(prob for short) in format [prob, prob] e.g. [0.9745, 0.0032]**

## Disclaimer
**As this is a POC(proof of concept) model it has some limitations and flaws(for now)**

1. **It works only with English language**
2. **The model was trained on monologue phrases, which means that any dialogue type conversation might lead to false triggers**
3. **Right now it can only detect explicitly good/explicitly bad. If the real bank worker would call it will most likely be classified as scam, since the wording is highly matches that of a scammer. This is a very close borderline that current model can't distinguish**

##  Installation

1. **Create the environment from the `environment.yml` file**:  
   Run the following command in your terminal:
   ```bash
   conda env create -f environment.yml --name your_env_name # Replace "your_env_name" with a new name
   ```

2. **Activate the environment**:  
   Activate your environment using the command:
   ```bash
   conda activate your_env_name  # Replace "your_env_name" with the name you created in the previous step
   ```
3. **Install sentence-transformers**:
   ```
   pip install -U sentence-transformers
   ```
4. **Get finetuned model from HuggingFace**:
   ```python
   from sentence_transformers import SentenceTransformer
   
   model = SentenceTransformer("cuteo23/scam_finetuned")

## Usage

### 1. Ensure your are in the project's root directory

The script uses a custom class import, so if you try to run it from somewhere else, the SBERTClassifier() won't be imported.

### 2. Running the model

Run the "scam_detection.py" in your terminal. This might take a while but if everything worked out you'll see the "Recording... Ctrl+C to stop recording" message.

or

If you wish to see the code wrapped with a small simple UI:
1. Install the streamlit via pip
'''bash
pip install streamlit
'''
2. Run the script in your terminal
'''bash
streamlit run streamlit.py
'''

### 3. Get the model's predictions

Start speaking into the microphone. Every 5 seconds the output that looks like this will be generated:
'''bash
 Hello there, this is David from the Security Department.
⚠️ Possible scam
 We've detected suspicious activity on your account, and we think someone...
⚠️ Possible scam
 might try to get your money overseas.
⚠️ Possible scam
 if it was not you, you should probably...
⚠️ Possible scam
 Get this safe.
🚨 Scammer detected! Call aborted.
'''

There are 3 types of output:

✅ Your call is safe
⚠️ Possible scam
🚨 Scammer detected! Call aborted.

The "🚨 Scammer detected! Call aborted." can be seen only if the model predicts the chunks to be "⚠️ Possible scam" 8 consecutive times. In this case, the script breaks out of the loop and ends.

---

## ⚠️ Note

- **I am new to this**: If anything doesn’t work, or if you have any suggestions or improvements, please feel free to contact me via Telegram.

---

##  Contact

- **Telegram**: [https://t.me/frozenfoxby]
