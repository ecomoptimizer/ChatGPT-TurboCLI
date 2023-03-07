import openai
import os
import argparse
import configparser
import logging
import spacy
from collections import Counter
import tiktoken
from collections import deque
import json

# Configure the logger
logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s: %(asctime)s | %(funcName)s | line: %(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

red = "\033[31m"
green = "\033[32m"
blue = "\033[34m"
code = "\033[36m"

nlp = None
enc = None

def get_nlp():
    global nlp
    if not nlp:
        nlp = spacy.load("en_core_web_sm")
    return nlp

class Chatbot:
    """
    A class for creating and interacting with a chatbot.

    Attributes:
        model (str): The name of the OpenAI model to be used.
        temperature (float): The temperature the OpenAI should use.
        max_token_count (int): The maximum length of an input message in tokens.
        messages (List[Dict]): The list of messages exchanged between the user and the chatbot.
    """
    def __init__(self, model, temperature, max_token_count):
        """
        Initializes a new instance of the Chatbot class.

        Args:
            model (str): The name of the OpenAI model to be used.
            temperature (float): The temperature the OpenAI should use.
            max_token_count (int): The maximum length of an input message in tokens.
        """
        self.model = model
        self.temperature = temperature
        self.max_token_count = max_token_count
        self.messages = []
        self.recent_history = deque([])
        self.kept_history = deque([])

    def add_to_history(self, message):
        """
        Adds a new message to the chat history.
        Args:
            message (Dict): The message to be added.
        """
        # Add the new message to the recent history deque
        self.recent_history.append(message)
        logger.debug(message)
        # If the recent history deque is too long, move the oldest message to the kept history deque
        while len(self.recent_history) > self.max_token_count:
            oldest_message = self.recent_history.popleft()
            self.kept_history.append(oldest_message)

            # If the kept history deque is too long, remove the oldest message
            while len(self.kept_history) > self.max_token_count:
                self.kept_history.popleft()
    
    def remove_from_history(self):
        """
        Removes the most recent message from the chat history.
        """
        if not self.messages:
            return

        message = self.messages.pop()

        if message["role"] == "user":
            # Remove the most recent user message from both history deques
            self.recent_history.pop()
            if not self.recent_history:
                self.kept_history.pop()
        elif message["role"] == "assistant":
            # Remove the most recent assistant message from the recent history deque
            self.recent_history.pop()

    def start(self):
        """
        Starts the chatbot and receives user input.
        """
        # Prompt user for type of chatbot they would like to create
        try:
            system_msg = input(f"{blue}What type of chatbot would you like to create?{code} ")
        except:
            logger.error(f"{red}Invalid input, program ending{code}")
            return
        self.add_to_history({"role": "system", "content": system_msg})
        # Print welcome message
        print(f"Say hello to your new assistant!")
        # Loop to receive and respond to user input
        while True:
            # Get user input
            try:
                message = input(f"{blue}> {code}")
                if os.path.isfile(message):
                    with open(message, 'r') as file:
                        message = file.read()
            except KeyboardInterrupt:
                logger.info(f"\n{red}Ctrl+C pressed. Exiting.{code}")
                break
            except:
                logger.error(f"{red}Invalid input, please try again{code}")
                continue
            # Break out of loop if user inputs "quit()"
            if message == "quit()":
                break
            if "tokenusage" == message:
                messages = [message['content'] for message in self.messages]
                roles = [message['role'] for message in self.messages]
                current_historic_chat = ' '.join(messages) + ' '.join(roles)
                sentences = current_historic_chat.split(".")
                tokens = list(map(enc.encode, sentences))
                token_lengths = [len(token) for token in tokens]
                input_tokens = sum(token_lengths)
                logger.debug(f"Number of tokens in current history: {input_tokens}")
            else:
                # Allow for multi-line input if user inputs "mlin"
                if "mlin" == message:
                    message, interrupted = self.get_multiline_input()
                    if message is None:
                        break
                    if interrupted:
                        break
                    message = "\n".join(message)

                # Record user input and get assistant response
                self.add_to_history({"role": "user", "content": message})
                response = self.get_response()
                logger.debug(type(response))
                if response:
                    self.add_to_history({"role": "assistant", "content": response})
                    # Print assistant response
                    print(f"\n{green}{response}{code}\n")
                else:
                    # Handle the error
                    logger.error(f"{red}Error occurred while processing request. Please try again.{code}")
                    self.remove_from_history()

    def get_response(self):
        """
        Gets a response from the OpenAI API.
        Returns:
        - str if successful, False if an error occurred.
        """
        # create variables to collect the stream of chunks
        collected_chunks = []
        collected_messages = []
        completion_limit = 2048

        try:
            messages = "".join([message["content"] for message in self.messages[-5:]])
            # Split the input into sentences and tokenize each sentence separately
            sentences = messages.split(".")
            tokens = [enc.encode(sentence.strip()) for sentence in sentences if sentence.strip()]
            logger.debug(tokens)
            # Calculate the total number of input tokens
            input_tokens = sum([len(token) for token in tokens])
            logger.debug(input_tokens)
            if (input_tokens + completion_limit) > 4096:
                logger.info("Running nlp analysis on input to extract keywords")

                # Determine the maximum number of tokens allowed for completions
                max_tokens = self.max_token_count
                completion_tokens = max_tokens - input_tokens - completion_limit

                # Truncate the input sentences to fit within the token limit
                truncated_tokens = []
                for token in tokens:
                    if len(truncated_tokens) + len(token) <= completion_tokens:
                        truncated_tokens.extend(token)
                    else:
                        break
                logger.debug(truncated_tokens)

                # Extract the important keywords in the input
                keywords = Counter()
                for sentence in sentences:
                    doc = get_nlp()(sentence)
                    for ent in doc.ents:
                        keywords[ent.text] += 1
                    for token in doc:
                        if token.pos_ in ["NOUN", "VERB", "ADJ"]:
                            keywords[token.text] += 1

                # Sort the keywords by frequency and prioritize the top ones
                priority_keywords = [key for key, _ in keywords.most_common(5)]

                logger.debug(f"Priority keywords: {priority_keywords}")
                try:
                    last_user_index = max([i for i, message in enumerate(self.messages) if message["role"] == "user"])
                    if last_user_index is not None:
                        self.messages[last_user_index]["content"] += f"Priority keywords: {priority_keywords}"
                        logger.debug("Added priority keywords to user message")
                except ValueError as e:
                    logger.error(e)
                    self.messages.append({"role": "user", "content": priority_keywords})

                logger.info("Analysis complete, sending to AI")

            # Combine all user messages into a single string
            #message = [{"role": "user", "content": messages}]
            
            # Send messages as a stream to the API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                stream=True,
                n=1,
                stop=None,
            )
            # Combine all response chunks into a single string
            for chunk in response:
                collected_chunks.append(chunk)  # save the event response
                chunk_message = chunk['choices'][0]['delta'] # extract the message
                collected_messages.append(chunk_message)  # save the message
            
            response_text = ''.join([m.get('content', '') for m in collected_messages])
            return response_text
        except openai.error.AuthenticationError:
            logger.error(f"{red}Authentication error: Invalid OpenAI API key{code}")
            return False
        except openai.error.InvalidRequestError as e:
            logger.error(f"{red}Invalid request error: {e}{code}")
            return False
        except openai.error.APIError as e:
            logger.error(f"{red}API error: {e}{code}")
            return False
        except Exception as e:
            logger.error(f"{red}Unknown error: {e}{code}")
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
                if token_count + len(line.split()) > self.max_token_count:
                    logger.error(f"{red}Input exceeds maximum length of {self.max_token_count} tokens. Please try again.{code}")
                    return lines, interrupted
                lines.append(line)
                token_count += len(line.split())
        except KeyboardInterrupt:
            interrupted = True
            logger.info(f"{red}Ctrl+C pressed. Exiting.{code}")
        return lines, interrupted
    
def main():
    """
    Parses command-line arguments and starts the chatbot.
    """
    parser = argparse.ArgumentParser(description="Chatbot CLI tool")
    parser.add_argument('--model', default='gpt-3.5-turbo', type=str, help='The name of the model to be used')
    parser.add_argument('--temperature', default='0.9', type=float, help='The temperature for the model')
    parser.add_argument('--max_tokens', default='4096', type=int, help='The maximum amount of tokens')
    parser.add_argument('--api_key', default=None, type=str, help='The OpenAI API key')
    args = parser.parse_args()

    if args.api_key is None:
        parser.error("OpenAI API key not found")

    openai.api_key = args.api_key

    # Verify that the model exists
    models = openai.Model.list()
    model_names = [m.id for m in models['data']]
    if args.model not in model_names:
        parser.error("Invalid model name")

    global enc
    enc = tiktoken.encoding_for_model(args.model)

    chatbot = Chatbot(args.model, args.temperature, args.max_tokens)
    chatbot.start()

if __name__ == '__main__':
    main()