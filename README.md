# AI-NLP-DL-practice
AI, Data Science & Machine Learning --- Learning Portfolio
A collection of Jupyter Notebook practices, course exercises,
experiments, and case studies completed while learning Python, Data
Science, Machine Learning, NLP, Transformers, and Large Language Model
(LLM) application development.
Note: This repository documents concepts and tools explored so
far. Not every library, model, or method listed here has been used in
every project or mastered in depth. Individual notebooks show the
specific implementation and level of practice.

Table of Contents
- Learning Areas
- Python and Data Analysis
- Data Visualization
- Machine Learning and Evaluation
- NLP and Text Processing
- Transformers and Hugging Face
- LLMs and the OpenAI API
- LangChain and Retrieval-Augmented
  Generation
- PyTorch and Model Training
- Methods and API Reference
- Models and Architectures
  Explored
- Case Studies and Application
  Ideas
- Tools and Workflow
- Suggested Notebook Format
- Running the Notebooks
- Responsible Use and Data Notes
Learning Areas
- Python programming and data handling
- Data cleaning and exploratory data analysis (EDA)
- Data visualization
- Machine Learning classification, regression, and anomaly detection
- Natural Language Processing (NLP)
- Transformer architecture and attention mechanisms
- Text classification and extractive question answering
- OpenAI API and LLM application patterns
- Prompt templates, embeddings, vector search, and RAG concepts
- Practical case-study analysis and model evaluation
Python and Data Analysis
Libraries and modules
  Library / module                    Purpose
  pandas (pd)                     Tabular data, DataFrames, loading
                                      and cleaning datasets
  numpy (np)                      Numerical operations, arrays, and
                                      numeric transformations
  re                                Regular expressions for
                                      pattern-based text cleaning
  random                            Random selection and basic
                                      reproducibility workflows
  cleantext                         Basic text cleaning through
                                      clean()
  enumerate                         Python built-in for iterating with
                                      an index
  Python strings and lists            Formatting, appending, slicing, and
                                  joining text
Methods and patterns practiced
- pd.read_csv() and DataFrame inspection/selection (where used in
  the notebooks)
- DataFrame filtering, column selection, and basic preprocessing
- NumPy array operations
- .append() for adding an item to a list
- enumerate() for indexed iteration
- .format() for string formatting
- Regular-expression substitutions with re
- Text cleaning with cleantext.clean()
Data Visualization
Libraries
- matplotlib / matplotlib.pyplot --- plotting and chart
  customization
- seaborn --- statistical visualizations
- NumPy --- preparing numeric arrays for plotting
Methods and techniques
- sns.barplot() for categorical comparisons
- Matplotlib tick-label rotation and alignment
- ax.grid(True) for adding grid lines
- Converting model scores or tensors to NumPy arrays for visualization
- Labelling charts and presenting model outputs clearly
Machine Learning and Evaluation
Scikit-learn
  Module / function                            Role
  sklearn.model_selection.train_test_split   Split data into training and
                                               testing sets
  sklearn.preprocessing.LabelEncoder         Convert categorical labels into
                                               integer labels
  sklearn model APIs                         Common fit, predict, and evaluation
                                           workflow when used by a notebook
Common ML workflow
1. Load and inspect a dataset.
2. Clean the data and handle missing or inconsistent values.
3. Separate features from target labels.
4. Split data into training and testing sets.
5. Encode labels when needed.
6. Train a suitable model.
7. Generate predictions.
8. Evaluate results and document limitations.
Evaluation concepts
- Training and test sets
- Classification labels and predicted classes
- Model metrics and error analysis
- Comparing model performance without assuming that a single metric
  tells the whole story
NLP and Text Processing
Libraries and modules
- re --- regular expressions and pattern-based cleaning
- cleantext.clean --- text cleanup
- Hugging Face transformers --- tokenizers, pretrained language
  models, pipelines, and training utilities
- datasets --- dataset preparation and handling
- evaluate --- model evaluation utilities
Concepts explored
- Text normalization and preprocessing
- Tokenization and token IDs
- Special tokens such as [CLS], [SEP], and model-specific
  equivalents
- Padding and truncation
- Attention masks
- Sequence classification
- Extractive question answering
- Class labels and label encoding
- Model logits and predicted class IDs
- Token-to-text reconstruction and WordPiece tokens such as ##ing
Transformers and Hugging Face
Main library
transformers provides pretrained model architectures, tokenizers,
inference pipelines, and training interfaces.
Classes and functions explored
  API                                    Purpose
  XLNetTokenizer                       Tokenize text for XLNet
  XLNetForSequenceClassification       XLNet model for sequence/text
                                         classification
  RobertaTokenizer                     Tokenize text for RoBERTa
  RobertaForQuestionAnswering          RoBERTa model for extractive
                                         question answering
  TrainingArguments                    Configure Hugging Face training
                                         settings
  Trainer                              Manage model training and
                                         evaluation workflows
  pipeline()                           High-level API for common NLP
                                         inference tasks
  AutoModelForSequenceClassification   Load a pretrained model with a
                                         sequence-classification head
  AutoTokenizer                        Load a matching tokenizer
                                         automatically
  AutoModel                            Load a base pretrained transformer
                                     model when appropriate
Transformer concepts
- Transformer encoder architecture
- Self-attention and contextual representations
- Bidirectional context
- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP), as used in original BERT pretraining
- Dynamic masking, used in RoBERTa-style training
- Permutation Language Modeling, used by XLNet
- Transformer-XL and longer-range dependency ideas
- Knowledge distillation, used to create smaller models such as
  DistilBERT
- Fine-tuning a pretrained model for a downstream task
LLMs and the OpenAI API
Python package and main client
- openai
- OpenAI client class
Example import and client setup:
from openai import OpenAI

client = OpenAI()
Configure the API key through an environment variable rather than
hard-coding it in a notebook or committing it to GitHub.
API methods explored
  Method / property                   Purpose
  client.responses.create()         Create a model response using the
                                      modern Responses API
  response.output_text              Conveniently access generated text
                                      from a response
  OpenAI                            Create a client for interacting
                                  with the API
Legacy API awareness
- openai.Completion.create() belongs to older API patterns and
  should not be treated as the default for new projects.
- Prefer the current API documented for the model and task you are
  using; API interfaces can evolve.
LLM concepts
- Large Language Models (LLMs)
- Tokens and tokenization
- Context windows
- Prompting and prompt engineering
- Temperature and output-token limits
- Inference versus training and fine-tuning
- Embeddings
- Hallucinations, grounding, and source-supported responses
- Tool/function calling and agent workflows
- API authentication and secret management
LangChain and Retrieval-Augmented Generation
Packages explored
- langchain
- langchain-core
- langchain-openai
- langchain-community
- langchain-text-splitters
Classes and components explored
  Component                           Purpose
  ChatOpenAI                        Connect a LangChain workflow to an
                                      OpenAI chat model
  ChatPromptTemplate                Build reusable chat prompt
                                      templates
  PyPDFLoader                       Load text from PDF documents
  RecursiveCharacterTextSplitter    Split documents into smaller chunks
  OpenAIEmbeddings                  Generate embeddings using an OpenAI
                                      embedding model
  FAISS                             Store and search vectors locally
                                  using FAISS integration
RAG workflow concepts
1. Load source documents.
2. Split documents into chunks.
3. Convert chunks into embeddings.
4. Store embeddings in a vector index/store.
5. Retrieve relevant chunks for a user query.
6. Pass retrieved context to an LLM.
7. Generate an answer grounded in the retrieved information.
8. Check that the answer is relevant and supported by the source
   material.
Other concepts explored include document loaders, retrievers, chains,
tools, agents, chat history, vector stores, and multi-step workflows.
PyTorch and Model Training
Library
- torch / PyTorch --- tensors, model execution, and deep-learning
  workflows
Functions and attributes explored
  Method / attribute                  Purpose
  torch.tensor()                    Create a tensor from data
  torch.no_grad()                   Disable gradient tracking during
                                      inference
  logits                            Raw model scores before converting
                                      them to predicted
                                      probabilities/classes
  argmax()                          Select the index of the largest
                                      score
  start_logits                      Scores for possible answer-start
                                      positions in extractive QA
  end_logits                        Scores for possible answer-end
                                      positions in extractive QA
  .detach()                         Detach a tensor from its
                                      computation graph
  .numpy()                          Convert a compatible CPU tensor to
                                      a NumPy array
  .flatten()                        Flatten an array into one dimension
For a tensor that resides on a GPU, move it to the CPU before calling
.numpy(), for example tensor.detach().cpu().numpy().
Methods and API Reference
This section collects important methods and functions encountered in the
notebooks. Check each library's official documentation for
version-specific details.
Tokenization and text conversion
- tokenizer.encode(text, ...) --- encode text into token IDs.
- tokenizer.convert_ids_to_tokens(ids) --- convert token IDs back to
  token strings.
- tokenizer.sep_token_id --- access the tokenizer's separator-token
  ID when defined.
- tokenizer(...) --- tokenize text and return model inputs such as
  input_ids and attention_mask.
- pipeline(task, ...) --- create a high-level inference pipeline.
Training and evaluation
- train_test_split(...) --- split data into subsets.
- LabelEncoder.fit_transform(...) --- learn label mappings and
  transform labels.
- Trainer(...) --- configure a Hugging Face Trainer.
- TrainingArguments(...) --- define training parameters.
- evaluate package --- load or compute evaluation metrics when
  configured for the task.
Model inference and answer extraction
- model(**inputs) --- pass tokenized inputs to a model.
- torch.no_grad() --- run inference without gradient tracking.
- outputs.logits --- inspect raw prediction scores.
- argmax() --- choose the highest-scoring index.
- Extractive QA --- select answer start and end token positions, slice
  the token span, and convert the selected tokens back into readable
  text.
- WordPiece cleanup --- join token pieces and handle continuation
  markers such as ## carefully.
Visualization and conversion
- sns.barplot(...) --- create a categorical bar plot.
- ax.grid(True) --- show plot grid lines.
- .detach().cpu().numpy().flatten() --- convert a detached tensor
  into a flattened CPU NumPy array.
OpenAI and LangChain
- client.responses.create(...) --- request a response from the
  OpenAI API.
- response.output_text --- retrieve generated text.
- ChatPromptTemplate --- define reusable prompt structure.
- PyPDFLoader --- load PDF content.
- RecursiveCharacterTextSplitter --- split text into chunks.
- OpenAIEmbeddings --- create text embeddings.
- FAISS --- perform vector similarity search through its LangChain
  integration.
Models and Architectures Explored
Language and NLP models
  Model / architecture                Main idea or use
  BERT                            Bidirectional encoder
                                      representation; commonly fine-tuned
                                      for classification and question
                                      answering
  RoBERTa                         BERT-style model trained with an
                                      optimized pretraining recipe,
                                      including dynamic masking
  DistilBERT                      Smaller, faster BERT-style model
                                      produced using knowledge
                                      distillation
  XLNet                           Transformer model using permutation
                                      language modeling
  Transformer Encoder             Encoder architecture using
                                      self-attention and feed-forward
                                      layers
  FinBERT                         BERT-based model adapted for
                                      financial-language tasks such as
                                      sentiment analysis
  LSTM                            Recurrent neural network
                                      architecture used for sequence data
  TF-IDF                          Statistical text-feature
                                      representation, not a neural
                                  language model
Other models discussed in case-study planning
The following were explored as potential methods for different
applications; list them as implemented only if the corresponding
notebook contains an actual implementation.
- Logistic Regression --- classification baseline
- Random Forest --- ensemble of decision trees
- XGBoost --- gradient-boosted decision trees
- LightGBM --- gradient boosting framework
- Artificial Neural Network (ANN) --- layered neural network
- Convolutional Neural Network (CNN) --- convolution-based feature
  learning
- Vision Transformer (ViT) --- transformer architecture for images
- Isolation Forest --- anomaly detection
- Autoencoder --- representation learning and reconstruction-based
  anomaly detection
- Graph Neural Network (GNN) --- learning from graph-structured
  data
  Case Studies and Application Ideas
Topics discussed or considered for practical applications include:
- Fake news detection and text classification
- Extractive FAQ/question-answering bot
Tools and Workflow
- Jupyter Notebook --- interactive coding and documentation
- VS Code --- code editing and notebook development
- Git and GitHub --- version control and portfolio hosting
- Python environments --- isolate project dependencies
- Hugging Face --- pretrained model and dataset ecosystem
Learning principle: This repository is a record of practice and
experimentation. It should grow as new concepts are implemented, tested,
and documented.
- Collaborative Filtering --- recommendation technique
- Generative Adversarial Network (GAN) --- generative model
  family, including cross-modal approaches discussed for sensor data
