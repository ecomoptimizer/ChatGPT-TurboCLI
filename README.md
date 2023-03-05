# TurboPersonas
This code is a program for creating a chatbot that interacts with the OpenAI API to generate responses. The chatbot can handle multi-line input, limit excessively long messages, and offers command-line arguments to select the preferred model and API key to be used. By following the simple command-line prompts, you can create your own conversational AI assistant without hassle.

## Installation

1. Clone this repository
2. Install the required Python packages:
    ```
    pip install openai
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

`python3 turbopersonas.py --api_key <api>`

>> What type of chatbot would you like to create? 

> coding wizard

>> Say hello to your new assistant!

> How do you code "hello world" in rust?

>> To print "Hello, world!" in Rust, you can use the `println!` macro:
>>
>>```rust
>>fn main() {
>>    println!("Hello, world!");
>>}
>>```
>>
>>When you run this program, it will print `Hello, world!` to the console.

## License

This project is licensed under the [Mozilla Public License 2.0](./LICENSE).

## References
Inspired by https://www.haihai.ai/chatgpt-api/