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
from logging.handlers import RotatingFileHandler
import datetime

# Create logs directory if not exists
if not os.path.exists("logs"):
    os.makedirs("logs")

red = "\033[31m"
green = "\033[32m"
blue = "\033[34m"
code = "\033[36m"

nlp = None
tokenizer = None
logger = None

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
    def __init__(self, model, temperature, max_token_count, completition_limit):
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
        self.completition_limit = completition_limit
        self.sent_history = []

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

    def add_to_sent(self, message):
        self.sent_history.extend(message)
    
    def remove_from_messages(self):
        """
        Removes the most recent message from the current chat messages.
        """
        if not self.messages:
            return

        self.messages.pop()

    def calculate_tokens(self, messagelist):
        """
        Calculates the amount of tokens used in messagelist.

        Args:
            messagelist (list): The list of message dicts to be counted for their tokens.
        """
        logger.debug(messagelist)
        if len(messagelist) == 0:
            return 0
        messages = [message['content'] for message in messagelist]
        roles = [message['role'] for message in self.messages]
        current_historic_chat = ' '.join(messages) + ' '.join(roles)
        sentences = current_historic_chat.split(".")
        tokens = list(map(tokenizer.encode, sentences))
        token_lengths = [len(token) for token in tokens]
        input_tokens = sum(token_lengths)
        return input_tokens

    def calculate_summary_tokens(self, summary):
        sentences = summary.split(".")
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
                logger.info(f"{red}Ctrl+C pressed. Exiting.{code}")
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
                kept_history_tokens = self.calculate_tokens(self.kept_history)
                logger.info(f"Number of tokens in current chat: {self.calculate_tokens(self.messages)}")
                logger.info(f"Number of tokens for whole session: {kept_history_tokens}")
                sent_history_tokens = self.calculate_tokens(self.sent_history)
                logger.debug(sent_history_tokens)
                logger.debug(self.sent_history)
                sent_tokens = kept_history_tokens - sent_history_tokens
                logger.debug(sent_tokens)
                if sent_tokens < 0:
                    sent_tokens = 0
                else:
                    sent_tokens = sent_history_tokens
                logger.debug(sent_tokens)
                logger.info(f"Number of tokens sent to OpenAI for whole session: {sent_tokens}")
                print(f"Number of tokens sent to OpenAI for whole session: {sent_tokens}")
            # Start a new chat with the same assistant as defined on CLI launch
            elif "newchat" == message:
                self.messages = []
                self.add_to_history(self.assistant_mode)
                logger.info("Chat messages has been cleared, ready for a new session.")
            elif "newai" == message:
                # Prompt user for type of chatbot they would like to create
                try:
                    system_msg = input(f"{blue}What type of chatbot would you like to create?{code} ")
                except:
                    logger.error(f"{red}Invalid input, program ending{code}")
                    return
                self.messages = []
                self.assistant_mode = {"role": "system", "content": system_msg}
                self.add_to_history(self.assistant_mode)
                logger.info(f"New asistant mode set: {self.assistant_mode['content']}. Chat messages has been cleared, ready for a new session.")
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
                    self.remove_from_messages()

    def get_response(self):
        """
        Gets a response from the OpenAI API.
        Returns:
        - str if successful, False if an error occurred.
        """
        # create variables to collect the stream of chunks
        collected_chunks = []
        collected_messages = []
        completition_limit = self.completition_limit

        try:
            messages = "".join([message["content"] for message in self.messages])
            # Split the input into sentences and tokenize each sentence separately
            sentences = messages.split(".")
            tokens = [tokenizer.encode(sentence.strip()) for sentence in sentences if sentence.strip()]
            #logger.debug(tokens)
            # Calculate the total number of input tokens
            input_tokens = self.calculate_tokens(self.messages)
            logger.debug(input_tokens)
            if (input_tokens + completition_limit) > self.max_token_count:
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
                logger.debug(f"ranked_senteces: {ranked_sentences}")
                # Generate a summary by combining the top-ranked sentences
                summary_length = 10
                summary = " ".join([sentence for sentence, score in ranked_sentences[:summary_length]])
                summary = summary.replace('\n', ' ')
                logger.debug(summary)
                # Calculate current number of tokens
                current_tokens = self.calculate_summary_tokens(summary)
                logger.debug(f"Current_tokens: {current_tokens}")
                # Increase summary length until token constraint is met
                while (current_tokens + completition_limit) < self.max_token_count and summary_length < len(ranked_sentences):
                    summary_length += 1
                    summary = " ".join([sentence for sentence, score in ranked_sentences[:summary_length]])
                    summary = summary.replace('\n', ' ')
                    current_tokens = self.calculate_summary_tokens(summary)

                # If token constraint is exceeded, backtrack
                while (current_tokens + completition_limit) > self.max_token_count and summary_length > 1:
                    summary_length -= 1
                    summary = " ".join([sentence for sentence, score in ranked_sentences[:summary_length]])
                    summary = summary.replace('\n', ' ')
                    current_tokens = self.calculate_summary_tokens(summary)

                # Finalize summary
                #summary = " ".join([sentence for sentence, score in ranked_sentences[:summary_length]])
                #summary = summary.replace('\n', ' ')
                logger.debug(f"Summary: {summary}")
                logger.debug("creating new message list")
                self.messages = []
                self.messages.append(self.assistant_mode)
                self.messages.append({"role": "user", "content": summary})

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
            self.add_to_sent(self.messages)
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
    parser.add_argument('--completition_limit', default='1024', type=int, help='The max amount of tokens to be used for completition')
    parser.add_argument('--api_key', default=None, type=str, help='The OpenAI API key')
    parser.add_argument('--log', dest='loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    help='Set the logging level')
    args = parser.parse_args()

    if args.api_key is None:
        parser.error("OpenAI API key not found")

    if args.loglevel:
        numeric_level = getattr(logging, args.loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % args.loglevel)
        logging.basicConfig(level=numeric_level)

    openai.api_key = args.api_key

    # Verify that the model exists
    models = openai.Model.list()
    model_names = [m.id for m in models['data']]
    if args.model not in model_names:
        parser.error("Invalid model name")

    # Configure the logger
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s: %(asctime)s | %(funcName)s | line: %(lineno)d | %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Get current date in YYYY-MM-DD format
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    # Create filename with current date
    log_file = os.path.join("logs", f"logfile_{current_date}.log")
    # Create rotating file handler with new file every day
    file_handler = RotatingFileHandler(log_file, maxBytes=1000000, backupCount=7)
    # Set log level and message format
    if args.loglevel:
        numeric_level = getattr(logging, args.loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % args.loglevel)
        file_handler.setLevel(numeric_level)
        logger.setLevel(numeric_level)
    else:
        file_handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s: %(asctime)s | %(funcName)s | line: %(lineno)d | %(message)s'))
    # Add file handler to logger
    logger.addHandler(file_handler)

    global tokenizer
    tokenizer = tiktoken.encoding_for_model(args.model)

    chatbot = Chatbot(args.model, args.temperature, args.max_tokens, args.completition_limit)
    chatbot.start()

if __name__ == '__main__':
    main()