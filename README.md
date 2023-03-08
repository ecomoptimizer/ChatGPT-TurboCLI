# ChatGPT TurboCLI
![image](https://user-images.githubusercontent.com/20763070/223877791-bbcc2a35-83c8-49ed-a42e-e59d8d34f806.png)


Introducing the ultimate coding wizard - a Python program that lets you create your very own chatbot using the powerful OpenAI API. With its seamless integration of the `openai` library, `spacy` for natural language processing, `textract` for file reading, and `nltk` for text tokenization and analysis, this program is the ultimate tool for creating and interacting with a chatbot. 

Our sophisticated `Chatbot` class lets you effortlessly record messages, store chat history, and get AI-generated responses from OpenAI's industry-leading model. Whether you're looking to create a customer support chatbot, a virtual assistant for your home, or just want to experiment with cutting-edge AI technology, this program has you covered. 

Plus, our command-line interface makes it easy for users of all skill levels to get started. Simply input your OpenAI API key, set your desired model and temperature, and let our program do the rest. You can even upload files for text analysis or input multiline messages for a truly personalized experience. 

Don't miss out on this opportunity to unleash your creativity and take your chatbot game to the next level. Try our coding wizard now and experience the power of OpenAI at your fingertips.

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
