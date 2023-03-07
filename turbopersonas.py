import openai
import os
import argparse
import logging
import spacy
from collections import Counter
import tiktoken
from pathlib import Path
import textract
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

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
tokenizer = None

def get_nlp():
    """
    Returns a spaCy NLP object for the 'en_core_web_sm' model.
    """
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
        self.assistant_mode = None
        self.kept_history = []

    def add_to_history(self, message):
        """
        Adds a new message to the chat history.
        Args:
            message (Dict): The message to be added.
        """
        # Add the new message to the recent history deque
        self.kept_history.append(message)
        self.messages.append(message)
        logger.debug(message)
    
    def remove_from_history(self):
        """
        Removes the most recent message from the chat history.
        """
        if not self.messages:
            return

        self.messages.pop()

    def calculate_tokens(self, messagelist):
        messages = [message['content'] for message in messagelist]
        roles = [message['role'] for message in self.messages]
        current_historic_chat = ' '.join(messages) + ' '.join(roles)
        sentences = current_historic_chat.split(".")
        tokens = list(map(tokenizer.encode, sentences))
        token_lengths = [len(token) for token in tokens]
        input_tokens = sum(token_lengths)
        return input_tokens

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
        #self.add_to_history({"role": "system", "content": system_msg})
        self.assistant_mode = {"role": "system", "content": system_msg}
        self.add_to_history(self.assistant_mode)
        # Print welcome message
        print(f"Say hello to your new assistant!")
        # Loop to receive and respond to user input
        while True:
            # Get user input
            try:
                message = input(f"{blue}> {code}")
                if message.startswith("file"):
                    file_path = Path(message.split(" ", 1)[1].replace("'", ""))
                    if not file_path.is_file():
                        logger.error(f"{red}File not found: {file_path}{code}")
                        continue
                    message = textract.process(str(file_path))
                    message = message.decode('utf-8') # or any other encoding used in the file
                    logger.debug(message)
            except KeyboardInterrupt:
                logger.info(f"\n{red}Ctrl+C pressed. Exiting.{code}")
                break
            except EOFError as e:
                logger.error(f"{red}Invalid input, please try again. Error code {e} {code}")
                continue
            except ValueError as e:
                logger.error(f"{red}Invalid input, please try again. Error code {e} {code}")
                continue
            except IndexError as e:
                logger.error(f"{red}Invalid input, please try again. Error code {e} {code}")
                continue
            # Break out of loop if user inputs "quit()"
            if message == "quit()":
                break
            # See the token usage for the CLI
            if "tokenusage" == message:
                logger.info(f"Number of tokens in current chat: {self.calculate_tokens(self.messages)}")
                logger.info(f"Number of tokens used for whole session: {self.calculate_tokens(self.kept_history)}")
            # Start a new chat with the same assistant as defined on CLI launch
            elif "newchat" == message:
                self.messages = []
                self.add_to_history(self.assistant_mode)
                logger.debug("Chat messages has been cleared, ready for a new session.")
            else:
                # Allow for multi-line input if user inputs "mlin"
                if "mlin" == message:
                    message, interrupted = self.get_multiline_input()
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
            messages = "".join([message["content"] for message in self.messages])
            # Split the input into sentences and tokenize each sentence separately
            sentences = messages.split(".")
            tokens = [tokenizer.encode(sentence.strip()) for sentence in sentences if sentence.strip()]
            #logger.debug(tokens)
            # Calculate the total number of input tokens
            input_tokens = self.calculate_tokens(self.messages)
            logger.debug(input_tokens)
            if (input_tokens + completion_limit) > 4096:
                logger.info("Running nlp analysis on input to extract keywords")

                # Tokenize each sentence into words and remove stop words
                stop_words = set(stopwords.words('english'))
                words = []
                for sentence in sentences:
                    words += [word.lower() for word in word_tokenize(sentence) if word.lower() not in stop_words]

                # Calculate the frequency of each word
                freq_dist = FreqDist(words)

                # Rank the sentences based on the frequency of their words
                ranked_sentences = []
                for sentence in sentences:
                    sentence_words = [word.lower() for word in word_tokenize(sentence) if word.lower() not in stop_words]
                    sentence_score = sum([freq_dist[word] for word in sentence_words])
                    ranked_sentences.append((sentence, sentence_score))
                ranked_sentences.sort(key=lambda x: x[1], reverse=True)

                # Generate a summary by combining the top-ranked sentences
                summary_length = 20
                summary = " ".join([sentence for sentence, score in ranked_sentences[:summary_length]])
                logger.debug(f"Summary: {summary}")
                try:
                    logger.debug("creating new message list")
                    last_user_index = max([i for i, message in enumerate(self.messages) if message["role"] == "user"])
                    last_user_message = self.messages[last_user_index]
                    self.messages = []
                    self.messages.append(self.assistant_mode)
                    self.messages.append({"role": "user", "content": summary})
                    #if last_user_index is not None:
                    #    self.messages.append(last_user_message)
                except ValueError as e:
                    logger.error(e)
                    self.messages = []
                    self.messages.append(self.assistant_mode)
                    self.messages.append({"role": "user", "content": historic_extraction})

                logger.info("Analysis complete, sending to AI")

            # Combine all user messages into a single string
            #message = [{"role": "user", "content": messages}]
            logger.debug(self.messages)
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

    global tokenizer
    tokenizer = tiktoken.encoding_for_model(args.model)

    chatbot = Chatbot(args.model, args.temperature, args.max_tokens)
    chatbot.start()

if __name__ == '__main__':
    main()