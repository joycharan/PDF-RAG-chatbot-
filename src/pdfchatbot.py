import yaml
import fitz
import gradio as gr
from PIL import Image
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import warnings
warnings.filterwarnings('ignore')

class PDFChatBot:
    def __init__(self, config_path="../config.yaml"):
        """
        Initialize the PDFChatBot instance.

        Parameters:
            config_path (str): Path to the configuration file (default is "config.yaml").
        """
        self.processed = False
        self.page = 0
        self.chat_history = []
        self.config = self.load_config(config_path)
        # Initialize other attributes to None
        self.prompt = None
        self.documents = None
        self.embeddings = None
        self.vectordb = None
        self.llm = None
        self.chain = None

    def load_config(self, file_path):
        """
        Load configuration from a YAML file.

        Parameters:
            file_path (str): Path to the YAML configuration file.

        Returns:
            dict: Configuration as a dictionary.
        """
        with open(file_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
                return config
            except yaml.YAMLError as exc:
                print(f"Error loading configuration: {exc}")
                return None

    def add_text(self, history, text):
        """
        Add user-entered text to the chat history.

        Parameters:
            history (list): List of chat history tuples.
            text (str): User-entered text.

        Returns:
            list: Updated chat history.
        """
        if not text:
            raise gr.Error('Enter text')
        history.append((text, ''))
        return history

    def create_prompt_template(self):
        """
        Create a prompt template for the chatbot.
        """
        template = (
            "You are a helpful assistant that answers questions about PDF documents.\n"
            "Given the following conversation and a follow up question, "
            "rephrase the follow up question to be a standalone question.\n"
            "Chat History:\n{chat_history}\nFollow Up Input: {question}\n"
            "Standalone question:"
        )
        self.prompt = PromptTemplate.from_template(template)

    def load_embeddings(self):
        """
        Load embeddings from Hugging Face and set in the config file.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.get("modelEmbeddings"))

    def load_vectordb(self):
        """
        Load the vector database from the documents and embeddings.
        """
        self.vectordb = Chroma.from_documents(self.documents, self.embeddings)

    def initialize_llm(self):
        """
        Initialize the API-based Language Model (LLM) client.
        This method supports either OpenAI's ChatGPT or Google's Gemini model.
        
        API keys are read from the config.yaml file, or can be set as environment variables:
        - For OpenAI: os.environ['OPENAI_API_KEY'] = 'your-api-key'
        - For Google: os.environ['GOOGLE_API_KEY'] = 'your-api-key'
        """
        # Check if API type is specified in config, otherwise check environment variable
        api_type = self.config.get("LLM_API_TYPE")
        if not api_type:
            api_type = os.environ.get("LLM_API_TYPE", "google").lower()
        else:
            api_type = api_type.lower()
        
        try:
            if api_type == "openai":
                # Check if API key is available in config or environment
                api_key = self.config.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key is not set in config or environment variable")
                
                # Set the API key in environment for the client to use
                os.environ["OPENAI_API_KEY"] = api_key
                    
                # Initialize OpenAI client
                self.llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",  # You can change this to a different OpenAI model
                    temperature=0.7,
                    max_tokens=500
                )
                print("Successfully initialized OpenAI API client")
                
            elif api_type == "google":
                # Check if API key is available in config or environment
                api_key = self.config.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("Google API key is not set in config or environment variable")
                
                # Set the API key in environment for the client to use
                os.environ["GOOGLE_API_KEY"] = api_key
                
                # Get model name from config or use default
                model_name = self.config.get("GOOGLE_MODEL_NAME", "gemini-flash-lite-latest")
                
                # Initialize Google Gemini client with the specified model
                self.llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0.7,
                    max_output_tokens=500
                )
                print(f"Successfully initialized Google Gemini API client with model: {model_name}")
                
            else:
                raise ValueError(f"Unsupported API type: {api_type}. Use 'openai' or 'google'")
                
        except Exception as e:
            print(f"Error initializing API client: {e}")
            raise

    def create_chain(self):
        """
        Create a Conversational Retrieval Chain using an API-based LLM
        """
        qa_prompt_template = (
            "You are a helpful AI assistant answering questions based on PDF document content.\n"
            "Use only the following context to answer the question. If you don't know the answer from the context, "
            "say 'I don't have enough information to answer this question from the PDF.'\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )
        qa_prompt = PromptTemplate.from_template(qa_prompt_template)

        self.chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            chain_type="stuff",
            retriever=self.vectordb.as_retriever(search_kwargs={"k": 4}),  # Retrieve more documents for better context
            condense_question_prompt=self.prompt,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True
        )

    def process_file(self, file):
        """
        Process the uploaded PDF file and initialize necessary components:
        Embeddings, VectorDB and API-based LLM.

        Parameters:
            file (FileStorage): The uploaded PDF file.
        """
        # Check PDF size
        import os
        file_size_mb = os.path.getsize(file.name) / (1024 * 1024)
        if file_size_mb > 50:  # 50MB limit
            print(f"Warning: Large PDF detected ({file_size_mb:.1f}MB). Processing may be slow.")
            if file_size_mb > 100:  # 100MB hard limit
                raise gr.Error(f"PDF too large ({file_size_mb:.1f}MB). Please use a file smaller than 100MB.")
                
        self.create_prompt_template()
        # Load and split documents into smaller chunks
        raw_documents = PyPDFLoader(file.name).load()
        
        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        self.documents = text_splitter.split_documents(raw_documents)
        print(f"Split PDF into {len(self.documents)} chunks for improved processing")
        
        self.load_embeddings()
        self.load_vectordb()
        self.initialize_llm()
        self.create_chain()

    def generate_response(self, history, query, file):
        """
        Generate a response based on user query and chat history.

        Parameters:
            history (list): List of chat history tuples.
            query (str): User's query.
            file (FileStorage): The uploaded PDF file.

        Returns:
            tuple: Updated chat history and a space.
        """
        if not query:
            raise gr.Error(message='Submit a question')
        if not file:
            raise gr.Error(message='Upload a PDF')
        if not self.processed:
            self.process_file(file)
            self.processed = True

        try:
            # Format chat history for the chain
            formatted_history = [(q, a) for q, a in self.chat_history]
            
            # Call the chain with proper parameters
            result = self.chain({
                "question": query, 
                "chat_history": formatted_history
            })
            
            # Extract the answer and update chat history
            answer = result.get("answer", "I couldn't generate an answer based on the PDF content.")
            self.chat_history.append((query, answer))
            
            # Update the page to display
            if 'source_documents' in result and result['source_documents']:
                # Get page from the first source document
                self.page = result['source_documents'][0].metadata.get('page', 0)
            
            # Update the displayed response in the chat history
            history[-1] = (history[-1][0], answer)
            
            return history, ""
        
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            print(f"Error in generate_response: {str(e)}")
            history[-1] = (history[-1][0], error_message)
            return history, ""

    def render_file(self, file):
        """
        Renders a specific page of a PDF file as an image.

        Parameters:
            file (FileStorage): The PDF file.

        Returns:
            PIL.Image.Image: The rendered page as an image.
        """
        if not file:
            return None
            
        try:
            doc = fitz.open(file.name)
            page = doc[self.page]
            pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
            image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
            doc.close()  # Properly close the document
            return image
        except Exception as e:
            print(f"Error rendering PDF: {str(e)}")
            return None
    
    def chat_flow(self, history, text, file):
        """
        Combined function that handles the entire chat flow in one operation
        to avoid session issues.
        
        Parameters:
            history (list): List of chat history tuples
            text (str): User's input text
            file (FileStorage): The uploaded PDF file
            
        Returns:
            tuple: Updated chat history, empty text input, and rendered image
        """
        try:
            if not text:
                raise gr.Error('Please enter a question')
                
            if not file:
                raise gr.Error('Please upload a PDF file first')
                
            # Add user message to history
            history.append((text, ''))
            
            # Process file if not already done
            if not self.processed:
                self.process_file(file)
                self.processed = True
            
            # Format chat history for the chain
            formatted_history = [(q, a) for q, a in self.chat_history]
            
            # Get response from the chain
            result = self.chain({
                "question": text, 
                "chat_history": formatted_history
            })
            
            # Extract the answer
            answer = result.get("answer", "I couldn't generate an answer based on the PDF content.")
            self.chat_history.append((text, answer))
            
            # Update the page to display
            if 'source_documents' in result and result['source_documents']:
                self.page = result['source_documents'][0].metadata.get('page', 0)
            
            # Update the displayed response in history
            history[-1] = (history[-1][0], answer)
            
            # Render the updated page from PDF
            image = self.render_file(file)
            
            return history, "", image
            
        except gr.Error as e:
            # Pass Gradio errors through for proper display
            raise e
        except Exception as e:
            error_message = f"Error: {str(e)}"
            print(f"Error in chat_flow: {str(e)}")
            
            # Check for API quota issues
            if "quota" in str(e).lower():
                error_message = "Google API quota exceeded. Try again later or use a different API key. You can update the API key in config.yaml."
            elif "rate limit" in str(e).lower():
                error_message = "Rate limit exceeded. Please wait a moment before trying again."
                
            # If history exists, update the last message
            if history and len(history) > 0:
                history[-1] = (history[-1][0], error_message)
            else:
                history = [(text, error_message)]
                
            return history, "", None