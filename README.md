# ChatGPT TurboCLI

## Key features
ChatGPT TurboCLI is a powerful assistant designed to help coders, writers, and analysts with their work. This tool offers a variety of features including file input analysis, multiline input, chat history, and summarization of large text data. The tool is built on OpenAI's ChatGPT-3.5-turbo model and uses natural language processing techniques to provide useful insights and suggestions. With ChatGPT TurboCLI, users can improve their productivity and efficiency in coding, writing, and data analysis.

| Feature | Benefits | Status |
| --- | --- | --- |
| File Input for Analysis of Reports | Enables analysis of a wide range of text data including financial reports, news reports, source code, etc. | Implemented |
| Multiline Input for Quick Paste | Allows quick and easy pasting of large amounts of data or complex text data. | Implemented |
| Summarization for Large File Input | Summarizes large text files using natural language processing techniques to identify important trends and insights. | Implemented (Using `textract` module - working on MacOS & Linus, windows not supported) |
| Token Usage for Cost Calculation | Calculates the number of tokens used and their associated cost in the application. | Implemented |
| Adjustable Completion Tokens for Customization | Adjusts how large of an output from model you want at most. | Implemented |
| Chat History for Context-Aware Communication | Keeps track of previous interactions with users, allowing the chatbot to provide more natural and intuitive responses. | Implemented |
| Transcripts to View Communication History | Provides historic questions and outputs, allowing for easier tracking of communication history and referencing of previous interactions with the chatbot. | Implemented |
| Logs for Debugging | Detailed records of all interactions with the application, including input and output data and any errors that occur during processing. Useful for debugging purposes in case of errors. | Implemented |
| Toggleable logging | Choose if you want logging at all | Implemented |
| Summarization on any input over x length | Summarize any input that are exceeding a given length to optimize token usage. | Not started |

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
    python chatbot.py --api_key YOUR_API_KEY --model MODEL_NAME
    ```
    Replace YOUR_API_KEY with your OpenAI API key and MODEL_NAME with the name of the OpenAI model you would like to use.

## Usage

- The chatbot will prompt you to enter the type of chatbot you would like to create. Enter your response and press enter.

- You can now start chatting with the chatbot. Type your messages and press enter to send them. The chatbot will respond with a message.

- If you want to exit the chatbot, type `quit()` and press enter.

- To see how many tokens have been used in the current chat or for the whole session, type `tokenusage` and press enter.

- To start a new chat with the same assistant as defined on CLI launch, type `newchat` and press enter.

- To input multiple lines at once, type `mlin` and then enter your message. Press `ctrl + D` to end the message.

Note: If you encounter any errors while using the chatbot, refer to the error message displayed in the terminal for assistance.

```
usage: chatgpt-turbocli.py [-h] [--model MODEL] [--temperature TEMPERATURE] [--max_tokens MAX_TOKENS] [--completition_limit COMPLETITION_LIMIT]
                           [--api_key API_KEY] [--log {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--transcript TRANSCRIPT]

Chatbot CLI tool

options:
  -h, --help            show this help message and exit
  --model MODEL         The name of the model to be used
  --temperature TEMPERATURE
                        The temperature for the model
  --max_tokens MAX_TOKENS
                        The maximum amount of tokens
  --completition_limit COMPLETITION_LIMIT
                        The max amount of tokens to be used for completition
  --api_key API_KEY     The OpenAI API key
  --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the logging level
  --transcript TRANSCRIPT
                        Write a transcript on exit?
```

### Example usage

`python3 turbopersonas.py --api_key <apikey> --model gpt-3.5-turbo-0301 --max_tokens 4096`
```
What type of chatbot would you like to create? 
> coding wizard
Hello! How can I assist you today? Are you looking for a coding wizard?
> Yes. What kind of wizardry can you perform?
As an AI language model, I can assist you with a wide range of programming languages including Python, Java, C++, JavaScript, HTML/CSS, and more. Here are some of the wizardry skills I possess:
1. Debugging and testing code
2. Implementing data structures and algorithms
3. Developing web applications and APIs
4. Building machine learning models and data analysis pipelines
5. Creating and managing databases
6. Automating repetitive tasks and workflows
7. Designing user interfaces and user experiences

Let me know what kind of help you are looking for, and I'll be happy to assist you.
```

### Example of logging

The logging is saved to disk and will look like this:
```
DEBUG: 2023-03-07 23:52:34,605 | get_response | line: 258 | [{'role': 'system', 'content': 'coding wizard'}, {'role': 'user', 'content': 'testing our brand new logging'}, {'role': 'assistant', 'content': "Great! Let's start by defining the purpose of the logging system. What kind of information do you want to log? Do you want to monitor errors or track user behavior? This will help us determine what kind of logging system to implement."}, {'role': 'user', 'content': 'Thanks'}]
DEBUG: 2023-03-07 23:52:42,640 | start | line: 185 | <class 'str'>
DEBUG: 2023-03-07 23:52:42,642 | add_to_history | line: 101 | {'role': 'assistant', 'content': "You're welcome! Once you have a clear purpose in mind, you can choose a logging framework to use. Some popular logging frameworks for different programming languages include:\n\n- Python: logging, loguru, structlog\n- Java: Log4j, Logback, java.util.logging\n- JavaScript: Winston, Bunyan, log4js\n\nOnce you have chosen a logging framework, you can start adding logging statements to your code. These statements will write messages to a log file or other destination when certain events occur, such as errors or user actions.\n\nIt is important to choose the appropriate logging level for each message. The most common levels are DEBUG, INFO, WARNING, ERROR, and CRITICAL. You may also want to include contextual information in your log messages, such as the user ID or timestamp.\n\nFinally, be sure to regularly review your logs to identify patterns or issues that may need attention. With a well-designed logging system in place, you can gain valuable insights into your application's performance and user behavior."}
INFO: 2023-03-07 23:52:45,048 | start | line: 151 | [31mCtrl+C pressed. Exiting.[36m
````

Adjust the wanted level with `--log <LEVEL>`on launch.


## License

This project is licensed under the [Mozilla Public License 2.0](./LICENSE).

## References
Inspired by https://www.haihai.ai/chatgpt-api/
