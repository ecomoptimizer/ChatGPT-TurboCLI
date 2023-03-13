# ChatGPT TurboCLI

## Key features
### Assistant personas via assistants.yml
![image](https://user-images.githubusercontent.com/20763070/224506786-67076c6c-0b27-4084-9523-7518477bf7f2.png)
### Few shot learning (context loading) through stories.yml
![image](https://user-images.githubusercontent.com/20763070/224506819-56a2f1ba-b7f1-42a1-baee-e2c0c27c5b55.png)
### File input while in CLI
![fileexample](https://user-images.githubusercontent.com/20763070/224507231-7b9e405d-3be8-433d-be00-aa25a934133c.png)



| Feature | Benefits | Status |
| --- | --- | --- |
| File Input for Analysis of Reports | Enables analysis of a wide range of text data including financial reports, news reports, source code, etc. | ✅ |
| Multiline Input for Quick Paste | Allows quick and easy pasting of large amounts of data or complex text data. | ✅ |
| Summarization for Large File Input | Summarizes large text files using natural language processing techniques to identify important trends and insights. | ✅ (Using `textract` module - working on macOS & Linux, Windows not supported) |
| Token Usage for Cost Calculation | Calculates the number of tokens used and their associated cost in the application. | ✅ |
| Adjustable Completion Tokens for Customization | Adjusts how large of an output from model you want at most. | ✅ |
| Chat History for Context-Aware Communication | Keeps track of previous interactions with users, allowing the chatbot to provide more natural and intuitive responses. | ✅ |
| Transcripts to View Communication History | Provides historic questions and outputs, allowing for easier tracking of communication history and referencing of previous interactions with the chatbot. | ✅ |
| Logs for Debugging | Detailed records of all interactions with the application, including input and output data and any errors that occur during processing. Useful for debugging purposes in case of errors. | ✅ |
| Toggleable logging | Choose if you want logging at all | ✅ |
| Summarization on any input over x length | Summarize any input that are exceeding a given length to optimize token usage. | ✅ |
| Load assistant personas from file | Load assistant personas defined in assistants.yml file. This should enable the usage of e.g. jailbreak personas. | ✅ |


## Installation

1. Clone this repository
2. Install the required Python packages:
    ```
    pip install -r requirements.txt
    python3 -m spacy download en_core_web_sm
    ```
3. Create an API key for OpenAI at https://beta.openai.com/signup/
4. Run the script in the command line:
    ```
    python chatgpt-turbocli.py --api_key YOUR_API_KEY --log ERROR --temperature 1
    ```
    Replace YOUR_API_KEY with your OpenAI API key and MODEL_NAME with the name of the OpenAI model you would like to use.

## Usage
Commands supported while in CLI:
```python
QUIT_COMMAND = "quit()"
TOKEN_USAGE_COMMAND = "tokenusage"
FEW_SHOT_COMMAND = "fewshot"
NEW_CHAT_COMMAND = "newchat"
NEW_AI_COMMAND = "newai"
MULTI_LINE_COMMAND = "mlin"
```
- To input multiple lines at once, type `mlin` and then enter your message. Press `ctrl + D` to end the message.

```
usage: chatgpt-turbocli.py [-h] [--model MODEL] [--temperature TEMPERATURE] [--max_tokens MAX_TOKENS] [--summary_tokens SUMMARY_TOKENS]
                           [--completition_limit COMPLETITION_LIMIT] [--api_key API_KEY] [--log {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--transcript TRANSCRIPT]
                           [--logenabled LOGENABLED]

Chatbot CLI tool

options:
  -h, --help            show this help message and exit
  --model MODEL         The name of the model to be used
  --temperature TEMPERATURE
                        The temperature for the model
  --max_tokens MAX_TOKENS
                        The maximum amount of tokens
  --summary_tokens SUMMARY_TOKENS
                        The number of input tokens to start summarizing
  --completition_limit COMPLETITION_LIMIT
                        The max amount of tokens to be used for completition
  --api_key API_KEY     The OpenAI API key
  --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the logging level
  --transcript TRANSCRIPT
                        Write a transcript on exit?
  --logenabled LOGENABLED
                        Logging enabled
```

## License

This project is licensed under the [Mozilla Public License 2.0](./LICENSE).

## References
Inspired by https://www.haihai.ai/chatgpt-api/
