import os
import api_keys
import streamlit as st

from langchain.vectorstores import Chroma

from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader


## VARIABLE DEFINITIONS ##
os.environ['OPENAI_API_KEY'] = api_keys.keys['openai']
persist_directory = 'db'
sources_directory = 'sources/'
llm_model = 'gpt-4'
embeddings_model = 'text-embedding-ada-002'
k = 5


## Models
embedding = OpenAIEmbeddings(model = embeddings_model)
llm = OpenAI(model_name = llm_model,
             temperature = 0.1,
             verbose=True)


## EMBED AND STORE THE CORPUS
def embed_and_store(embedding, persist_directory):
    loader = PyPDFDirectoryLoader(sources_directory)
    docs = loader.load()

    vectordb = Chroma.from_documents(documents=docs, 
                                     embedding=embedding, 
                                     persist_directory=persist_directory)
    vectordb.persist()
    vectordb = None
    
if not os.path.exists(persist_directory):
    print('Vectorstore not existent, generating and storing it now.')
    embed_and_store(embedding, persist_directory)

# Now we can load the persisted database from disk, and use it as normal. 
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={'k':k})

## set up the QA chain
qa = RetrievalQA.from_chain_type(llm=llm, 
                                 chain_type="stuff", 
                                 retriever=retriever, 
                                 return_source_documents=True)


## UI
st.header('- QA -')
st.title('All about Marcus Aurelius')
prompt = st.text_area('Ask your question here:', height=200)
print('prompt', prompt)

# run the LLM and get a response
if prompt:
    llm_response = qa(prompt)

    response = llm_response['result']
    st.write(f"**{response}**")
    st.write('\n\nSources\n-------')
    for source in llm_response['source_documents']:
        st.write('**' + source.metadata['source'] + ', page:' +  str(source.metadata['page']) + '**')
        st.write(source.page_content)
        st.write('\n')