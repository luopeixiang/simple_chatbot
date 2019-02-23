# A Simple ChatBot Based Encoder-Decoder

This is a  implementation of simplified and modular version of pytorch [CHATBOT TUTORIAL](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)



## Data

`datasets/conversations.csv` contain formatted and preprocessed conversations from [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).



## train model

`python3 main.py`

You can configure encoder-decoder model in `main.py`



## Evaluate model

Modify `model_path` variable in `evaluate.py` into your trained model path

Run `python3 evaluate.py` to chat with your trained model



**output example**:

```shell
Load checkpint...
Input> hello?
Chat_bot>  hello .
Input> where am I?
Chat_bot>  you re going to bed .
Input> how are you doing?
Chat_bot>  fine .
Input> are you my friend?
Chat_bot>  yes .
Input> you're under arrest
Chat_bot>  you re not here .
Input> i'm just kidding
Chat_bot>  no you re not .
Input> where are you from?
Chat_bot>  i m here .
Input> it's time for me to leave
Chat_bot>  i m sorry .

```



