import streamlit as st
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase


# load_resource function 
@st.cache_resource
def load_resource():
    # Load the LLM model
    os.environ['OPENAI_API_KEY'] = st.secrets['openai_api_key']
    os.environ['OPENAI_BASE_URL'] = st.secrets['openai_base_url']
    llm = ChatOpenAI(model = 'gpt-4o-mini')
    outputer = StrOutputParser()
    olist = SQLDatabase.from_uri('sqlite:///olist.db')
    olist_info = olist.get_table_info()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization = True)
    return llm, outputer, olist, olist_info, vectorstore

# global load resource
llm, outputer, olist, olist_info, vectorstore = load_resource()


# SQL part
def sql_part(question):
    # SQL prompt to create the SQL query
    sql_prompt = PromptTemplate.from_template(
    """
        - orders → customers : orders.customer_id = customers.customer_id
        - orders → items : orders.order_id = items.order_id
        - orders → payment : orders.order_id = payment.order_id
        - orders → review : orders.order_id = review.order_id
        - items → products : items.product_id = products.product_id
        - items → sellers : items.seller_id = sellers.seller_id
        - customers → geolocation : customers.customer_zip_code_prefix = geolocation.geolocation_zip_code_prefix
        - sellers → geolocation : sellers.seller_zip_code_prefix = geolocation.geolocation_zip_code_prefix
        This is the relationship between the tables.
        please based on question and table info, to write the correct SQL query.
        
        CRITICAL OVERRIDE RULE:
         - WHEN ABOUT RECOMMENDATION, MUST USE DENSE_RANK OR OTHER WINDOW FUNCTION TO SOLVE THE PROBLEM
         - USE WINDOW FUNCTIONS LIKE DENSE_RANK() FOR RECOMMENDATIONS. EXAMPLE
             WITH rk_data AS(
                SELECT a, DENSE_RANK() OVER (PARITTION BY category ORDER BY PRICE DESC) rk FROM category_table
            ) 
            SELECT a FROM rk_data WHERE rk <= 3
            IN THIS SITUATION, MUST PARTITION BY category TO RANK. EXAMPLE
            SELECT DISTINCT a FROM rk_data WHERE rk<=3
            IF HAVING BUDGET LIMIT, MUST USE WHERE TO FILTER PRICE BEFORE RANK. EXAMPLE
             WITH rk_data AS(
                SELECT a, DENSE_RANK() OVER (PARITTION BY category ORDER BY PRICE DESC) rk FROM category_table WHERE price <= 50
            ) 
            RECOMMENDATION NOTICE: WHATEVER USING WHICH FEATRUE TO RANK, SHOULD RANK PRICE TOO. EXAMPLE
               WITH rk_data AS(
                SELECT a, DENSE_RANK() OVER (PARITTION BY category ORDER BY SCORE DESC, PRICE DESC) rk FROM category_table WHERE price <= 50
            ) 
            WHEN FILTERING ON WINDOW FUNCTION RESULT, ALWAYS WRAP IN ANOTHER CTE:
                WITH base AS (...),
                ranked AS (SELECT *, DENSE_RANK() OVER (...) AS rk FROM base)
                SELECT * FROM ranked WHERE rk <= 3;

            
        NOTICE: 
            - ONLY WRITE THE DQL, DO NOT WRITE DML OR DDL
            - ONLY RETURN THE SQL QUERY WITHOUT MANY MARKDOWNS AND ORIGINAL QUESTION, ONLY PURE SQL QUERY
            - FOR RECOMMENDATION PROBLEM, USE DENSE_RANK() OR OTHER WINDOW FUNCTION WITH WHERE rk<=3, DO NOT ADD LIMIT
            - FRO OTHER QUERY WITHOUT WINDOW FUNCTION, USE LIMIT 10
            - NEVER RUTURN SOME MARKDOWNS
            - JUST OUTPUT NESSERARY OUTPUT, WHICH MINIMUM CAN ANSWER THE QUESTION
            - FOR RECOMMENDATION PROBLEM, MUST USE WINDOW FUNCTION LIKE DENSE_RANK(), RANK(), ROW_NUMBER() TO GET THE RANK NUMBER. IF THE QUESTION DO NOT INCLUDE HOW MANY VALUE TO OUTPUT, DEFAULT rk<=3, IF QUESTION SAY TOPN, MUST rk<=N. 
            - OUTPUT MUST DISTINCT, DO NOT OUTPUT SAME
            - PRODUCT CATEGROY IS NOT NULL WHEN QUERY, USING WHERE TO FILTER THE NULL VALUE
            - WHEN ALIGN FOR THE SUB-QUERY OR ECT, AVIDO TO USE THE SEPCIAL NAME IN SQL LIKE RANK, DEN_RANK, ORDER OR OTHERS
            - FOR EACH FEATURE, SHOULD OUTPUT WITH TABLE ALIGN LIKE PRICE SHOULD OBVIOUSLY WRITE FOR P.PRICE, P.PRODUCT_ID AND LIKE THIS.

            
        question:{question}
        table_info:{table_info}
    """
    )
    # Summary prompt to summarize the SQL query result
    summary_prompt = PromptTemplate.from_template(
    """
        You are a company ai chatbot
        Base on question, query_result and table _info, to summary a good answer
        Then answer should be good, not only the data, it still need more humanity, be perfessional
        
        question:{question}
        query_result:{query_result}
        table_info:{table_info}
    """
    )
    table_info = olist_info
    sql_chain = sql_prompt | llm | outputer
    query = sql_chain.invoke({'question':question, 'table_info':table_info})
    query_result = olist.run(query)
    summary_chain = summary_prompt | llm | outputer
    answer = summary_chain.invoke({'question':question, 'query_result':query_result,'table_info':table_info})
    return answer,query_result


# RAG Part
def rag_part(question, product_info = None):
    # RAG prompt
    rag_prompt = PromptTemplate.from_template(
    """
        You are a AI-CHATBOT. 
        You have our company's custmer review. 
        You know all data in our company like customer review, customer order.
        When customer ask a question, you will know the question and base on faiss_result, you will out put a clear, correct answer.
        If having some product name like that, base on them to run.
        
        NOTICE:
            THE ANSWER MUST FOLLOW BY faiss_result AND question, WHEN DO NOT GET ENOUGH INFORMATION, PLEASE OUT A APOLOGIZE TO SAY YOU DO NOT PLEASE TRY ANOTHER PROOBELM.
            BEFORE OUTPUT, YOU SHOULD TRANSALTE THE SUMMARY TO BE ENGLISH
        question:{question}
        faiss_result:{faiss_result}
    """
    )
    faiss_result = [i.page_content for i in vectorstore.similarity_search(question, k = 20)]
    chain = rag_prompt | llm | outputer
    # hybrid usage
    if product_info == None:
        answer = chain.invoke({'question':question,'faiss_result':faiss_result})
    else:
        question = f'{question}{product_info}'
        answer = chain.invoke({'question':question,'faiss_result':faiss_result})
    return answer


# Hybrid Part
def hybrid_part(question):
    # hybrid prompt
    hybrid_prompt = PromptTemplate.from_template(
    """
        You are a AI CAHTBOT, you have some information from our company data,table_info.
        You should combination by question, sql_result and rag result ruturn a better answer.
        Output should translate all in English
        
        question:{question}
        sql_result:{sql_result}
        rag_result:{rag_result}
        table_info:{table_info}
    """
    )
    table_info = olist_info
    _, sql_result = sql_part(question)
    rag_result = rag_part(question, product_info = sql_result)
    hybrid_chain = hybrid_prompt | llm | outputer
    answer = hybrid_chain.invoke({'question':question, 'sql_result':sql_result, 'rag_result':rag_result, 'table_info':table_info})
    return answer


# Normal question, solve by LLM
def llm_part(question):
    llm_prompt = PromptTemplate.from_template(
    """
        You are a AI CHATBOT
        You have our company data info
        Only the problem can not be search in our data you will be used
        Base on the question and olist info

        STEP:
            1. base on customer question, answer the question if you can.
            2. if the question is not about our company product, after answer, tell customer to try our business problem
            3. also can recommend to Live Support if the question is abstract and hard to understand

        question:{question}
        table_info:{table_info}
            
    """
    )
    table_info = olist_info
    llm_chain = llm_prompt | llm | outputer
    answer = llm_chain.invoke({'question':question, 'table_info':table_info})
    return answer


# Router Part
def route_function(question:str):
    route_prompt = PromptTemplate.from_template(
    """
        This prompt to define the question is go to SQL path, RAG path, HYBRID path or LLM path.
        CRITICAL OVERRIDE RULE:
            IF QUESTION INCLUDES BUDGET/PRICE CONSTRAINT:
              - IF ALSO mentions reviews/recommendations/quality → return "hybrid"
              - IF ONLY asks for data/statistics → return "sql"
              NEVER return "rag" when budget is mentioned.
            THIS RULE OVERRIDES ALL OTHER RULES.
        
        1. return rag:
            When the question is about: 
                customer feeling, 
                product review
                problem for product like delivery delay problem, quality problem
                When question is about delivery, like day, feeling

        2. return sql:
            When the question is about:
                Structure problem like top10 selling,
                Math problem for product like total seeling, how much

        3. return hybrid:
            WHen the question is about:
                Some quesiton is both about product info and customer review
                for example: I have 100 and I wanna choose a gift for my friend.
                Llke the example, need both SQL process and RAG process to get the result

        4. return llm
            Not about the product like hello, hi, how are you some problem without a special information

        The output only return sql,rag or llm in lowercase, don't return any other answer
        question: {question}
    """
    )
    route_chain = route_prompt | llm | outputer
    route_path = route_chain.invoke({'question':question}).strip()
    if route_path == 'rag':
        answer = rag_part(question)
    elif route_path == 'sql':
        answer,_ = sql_part(question)
    elif route_path == 'hybrid':
        answer = hybrid_part(question)
    else:
        answer = llm_part(question)
    return answer


# Page title
st.title("Olist E-commerce Chatbot")

# Streamlit page
if 'messages' not in st.session_state:
    st.session_state.messages = []

# loop print messages
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# input question and return answer
question = st.chat_input('input you quesition here: ')
if question:
    # user question input
    with st.chat_message('user'):
        st.markdown(question)
    st.session_state.messages.append({"role":'user', 'content':question})

    # Chatbot answer
    response = route_function(question)
    with st.chat_message('assistant'):
        st.markdown(response)
    st.session_state.messages.append({'role':'assistant', 'content':response})