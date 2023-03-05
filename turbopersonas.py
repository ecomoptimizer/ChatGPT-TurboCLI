import openai
import os
import argparse
import configparser

red = "\033[31m"
green = "\033[32m"
blue = "\033[34m"
code = "\033[36m"

MAX_TOKEN_COUNT = 4096

class Chatbot:
    """
    A class for creating and interacting with a chatbot.
    """
    def __init__(self, model):
        """
        Initializes a new instance of the Chatbot class.

        Args:
        - model (str): The name of the OpenAI model to be used.
        """
        self.model = model
        self.messages = []
    
    def start(self):
        """
        Starts the chatbot and receives user input.
        """
        # Prompt user for type of chatbot they would like to create
        try:
            system_msg = input(f"{blue}What type of chatbot would you like to create?{code} ")
        except:
            print(f"{red}Invalid input, program ending{code}")
            return
        self.messages.append({"role": "system", "content": system_msg})
        # Print welcome message
        print(f"Say hello to your new assistant!")
        # Loop to receive and respond to user input
        while True:
            # Get user input
            try:
                message = input(f"{blue}> {code}")
            except KeyboardInterrupt:
                print(f"{red}Ctrl+C pressed. Exiting.{code}")
                break
            except:
                print(f"{red}Invalid input, please try again{code}")
                continue
            # Check if user input exceeds maximum length and prompt for new input if true
            if len(message.split()) > MAX_TOKEN_COUNT:
                print(f"{red}Input exceeds maximum length of {MAX_TOKEN_COUNT} tokens. Please try again.{code}")
                continue
            # Break out of loop if user inputs "quit()"
            if message == "quit()":
                break
            # Allow for multi-line input if user inputs "mlin"
            if "mlin" in message:
                message, interrupted = self.get_multiline_input()
                if message is None:
                    break
                if interrupted:
                    break
                message = "\n".join(message)
            # Record user input and get assistant response
            self.messages.append({"role": "user", "content": message})
            response = self.get_response()
            if response:
                self.messages.append({"role": "assistant", "content": response})
                # Print assistant response
                print(f"\n{green}{response}{code}\n")
            else:
                # Remove the last message from the messages list
                self.messages.pop()
                # Handle the error
                print(f"{red}Error occurred while processing request. Please try again.{code}")

    def get_response(self):
        """
        Gets a response from the OpenAI API.

        Returns:
        - str if successful, False if an error occurred.
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.messages)
            return response["choices"][0]["message"]["content"]
        except openai.error.AuthenticationError:
            print(f"{red}Authentication error: Invalid OpenAI API key{code}")
            return False
        except openai.error.InvalidRequestError as e:
            print(f"{red}Invalid request error: {e}{code}")
            return False
        except openai.error.APIError as e:
            print(f"{red}API error: {e}{code}")
            return False
        except Exception as e:
            print(f"{red}Unknown error: {e}{code}")
            return False
    
    def get_multiline_input(self):
        """
        Gets multi-line input from the user.

        Returns:
        - tuple: (list of strings or None, boolean)
        """
        print(f"{green}Enter lines of text. Type {blue}\"stop\"{green} on a separate line to finish.{code}")
        lines = []
        token_count = 0
        interrupted = False
        try:
            while True:
                line = input(f"{blue}> {code}")
                if not line.strip():
                    continue  # ignore empty lines
                if line.strip() == "stop":
                    break
                if token_count + len(line.split()) > MAX_TOKEN_COUNT:
                    print(f"{red}Input exceeds maximum length of {MAX_TOKEN_COUNT} tokens. Please try again.{code}")
                    return lines, interrupted
                lines.append(line)
                token_count += len(line.split())
        except KeyboardInterrupt:
            interrupted = True
            print(f"{red}Ctrl+C pressed. Exiting.{code}")
        return lines, interrupted
    
def main():
    """
    Parses command-line arguments and starts the chatbot.
    """
    parser = argparse.ArgumentParser(description="Chatbot CLI tool")
    parser.add_argument('--model', default='gpt-3.5-turbo', type=str, help='The name of the model to be used')
    parser.add_argument('--api_key', default=None, type=str, help='The OpenAI API key')
    args = parser.parse_args()
    if args.api_key is None:
        print(f"{red}OpenAI API key not found{code}")
        return
    openai.api_key = args.api_key
    # Verify that the model exists
    models = openai.Model.list()
    model_names = [m.id for m in models['data']]
    if args.model not in model_names:
        print(f"{red}Invalid model name{code}")
        return
    chatbot = Chatbot(args.model)
    chatbot.start()

if __name__ == '__main__':
    main()