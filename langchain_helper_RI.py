from langchain_community.llms import GooglePalm
#from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain.prompts import SemanticSimilarityExampleSelector
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate

from few_shots import few_shots

def filter_complex_metadata(documents, allowed_types=(str, int, float, bool, object)):
    filtered_documents = []
    for document in documents:
        filtered_metadata = {}
        for key, value in document.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, allowed_types):
                        filtered_metadata[sub_key] = sub_value
            elif isinstance(value, allowed_types):
                filtered_metadata[key] = value
        filtered_documents.append(filtered_metadata)
    return filtered_documents


def get_few_shot_db_chain():

    api_key = "AIzaSyBC-tfOnEcYXuSYmNQkOmWjSu6e3EFYqWg"

    llm = GooglePalm(google_api_key=api_key, temperature=0.1)

    db_user = "root"
    db_password = "123"
    db_host = "localhost"
    db_name = "perf-custom1"

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}", sample_rows_in_table_info = 3)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    to_vectorize = [" ".join(str(example.values())) for example in few_shots]
    filtered_metadata = filter_complex_metadata(few_shots)
    vectorstore = Chroma.from_texts(to_vectorize, embedding=embeddings, metadatas=filtered_metadata)

    example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2
    )

    
    mysql_prompt = """You are a MySQL expert specializing in tables 'users', 'plans', and 'target_progress_percentage'. Given a question, you must formulate a syntactically correct MySQL query to execute. Then, analyze the query results to provide an answer to the input question.
    By default, unless the user specifies a specific number of examples, query for at most {top_k} results using the LIMIT clause as per MySQL conventions. Ensure that the results are ordered to return the most informative data from the database.
    Adhere to strict column selection guidelines: never query for all columns from a table. Only select the columns necessary to answer the question, and encase each column name in backticks (`) to denote them as delimited identifiers. Be cautious not to query for columns that do not exist and pay attention to column mappings between tables.
    Utilize the CURDATE() function to retrieve the current date, especially if the question involves "today". If something in the inverted commas like 'Nick', always try to find a matching column with strings.
    Format your responses as follows:


    Question: [Question here]
    SQLQuery: [Your SQL query without pre-amble]
    SQLResult: [Result of the SQL query]
    Answer: [Final answer here]

    Avoid unnecessary preambles in your responses.
    """

    prompt = """
    You are an expert in converting English questions to SQL query!. The SQL database has the name 'CUSTOM-PERF1' 
    and you need to consider only on the following tables; PLANS, TARGET_PERCENTAGE, and USERS.\n\nFor example, \nExample1 - How many plans related to Bhagya?,
    the SQL command will be something like this SELECT count(id) FROM plans WHERE userName = 'Bhagya';
    \nExample2 - How many total number of plans are there?, the SQL command will be something like this SELECT count(id) FROM plans;
    \nExample3 - What is the highest target progress plan?, then the SQL command will be something like this SELECT planTitle FROM target_percentage ORDER BY target_progress_percentage DESC LIMIT 1;

    Format your responses as follows:

    Question: [Question here]
    SQLQuery: [Your SQL query without pre-amble]
    SQLResult: [Result of the SQL query]
    Answer: [Final answer here]

    Avoid unnecessary preambles in your responses.
"""

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=_mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"], #These variables are used in the prefix and suffix
    )
    chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
    return chain




if __name__ == '__main__':
    #print(llm)
    chain = get_few_shot_db_chain()
    print(chain.run("How many plans are there?"))