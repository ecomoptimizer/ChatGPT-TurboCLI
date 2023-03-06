# TurboPersonas
This code is a program for creating a chatbot that interacts with the OpenAI API to generate responses. The chatbot can handle multi-line input, limit excessively long messages, and offers command-line arguments to select the preferred model and API key to be used. By following the simple command-line prompts, you can create your own conversational AI assistant without hassle.

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

Once the script is running, the chatbot will prompt the user for the type of chatbot they would like to create. After selecting the type, the chatbot will begin accepting user input and responding with an appropriate message. Users can input "quit()" to exit the chatbot.

If the user types "mlin" they can input multiple lines of text by pressing return after each input. When they are finished typing input, they need to type "stop" on a separate line.

When a response from OpenAI is received, it will be printed to the console in green text.

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
