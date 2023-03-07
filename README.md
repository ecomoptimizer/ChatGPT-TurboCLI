# TurboPersonas
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

### Example

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

## License

This project is licensed under the [Mozilla Public License 2.0](./LICENSE).

## References
Inspired by https://www.haihai.ai/chatgpt-api/
