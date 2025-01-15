# -*- coding: utf-8 -*-

import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from langchain_openai import ChatOpenAI
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import numpy as np
from openai import OpenAI
from langchain_community.document_loaders import CSVLoader

# 设置环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["DASHSCOPE_API_KEY"] = "sk-efc56c397df3476e8b1639b2db792239"

# 首先定义 ModelScopeEmbeddings 类


class ModelScopeEmbeddings(Embeddings):
    def __init__(self, model_id="damo/nlp_gte_sentence-embedding_chinese-base"):
        self.pipeline = pipeline(Tasks.sentence_embedding, model=model_id)

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            # 确保文本是字符串类型
            if not isinstance(text, str):
                text = str(text)
            # 将文本作为列表传入
            embedding = self.pipeline({"source_sentence": [text]})[
                'text_embedding']
            # 确保embedding是一维数组
            embedding = np.array(embedding).flatten()
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text):
        # 确保文本是字符串类型
        if not isinstance(text, str):
            text = str(text)
        embedding = self.pipeline({"source_sentence": [text]})[
            'text_embedding']
        # 确保query embedding也是一维数组
        return np.array(embedding).flatten()

# 然后是其他函数定义


def prepare_data(csv_path):
    # 使用 CSVLoader 加载 CSV 文件
    loader = CSVLoader(
        file_path=csv_path,
        csv_args={
            'delimiter': ',',
            'quotechar': '"',
            'fieldnames': ['data']  # 指定列名
        },
        encoding='utf-8',  # 将 encoding 移到外层参数
        source_column="data"  # 指定要处理的列
    )

    # 加载数据
    documents = loader.load()

    # 数据清洗（可选）
    cleaned_documents = []
    for doc in documents:
        # 确保文本是字符串类型
        if not isinstance(doc.page_content, str):
            doc.page_content = str(doc.page_content)
        # 移除多余的空白字符
        cleaned_text = doc.page_content.strip()
        # 移除重复的换行符
        cleaned_text = '\n'.join(line.strip()
                                 for line in cleaned_text.splitlines() if line.strip())
        doc.page_content = cleaned_text
        cleaned_documents.append(doc)

    # 文本分割
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    # 分割文本
    texts = text_splitter.split_documents(cleaned_documents)

    # 确保所有文本都是字符串类型
    for text in texts:
        if not isinstance(text.page_content, str):
            text.page_content = str(text.page_content)

    return texts


def build_vectorstore(texts, save_path="vectorstore"):
    """构建或加载向量数据库"""
    # 检查是否存在已保存的向量数据库
    if os.path.exists(save_path):
        print("找到已存在的向量数据库，正在加载...")
        embeddings = ModelScopeEmbeddings()
        vectorstore = FAISS.load_local(
            save_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("向量数据库加载完成！")
        return vectorstore

    print("未找到向量数���库，正在构建...")
    # 确保所有文本都是字符串类型
    text_strings = []
    for text in texts:
        if hasattr(text, 'page_content'):
            content = text.page_content
        else:
            content = text

        if not isinstance(content, str):
            content = str(content)
        text_strings.append(content)

    # 使用 ModelScope 的 embeddings
    embeddings = ModelScopeEmbeddings()

    # 使用FAISS构建向量数据库
    vectorstore = FAISS.from_texts(
        text_strings,
        embeddings
    )

    # 保存向量数据库到本地
    print("正在保存向量数据库...")
    vectorstore.save_local(save_path)
    print("向量数据库保存完成！")

    return vectorstore


def build_qa_chain(vectorstore, retriever_type="similarity"):
    """构建问答链，支持不同的检索方式"""
    # 基础 Prompt
    basic_prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
    如果无法从中得到答案，请说 "根据已知信息无法回答该问题"。

    已知信息：
    {context}

    问题：{question}
    回答："""

    # CoT Prompt
    cot_prompt_template = """你是一个专业的法律顾问。请基于以下已知信息，通过以下步骤来分析和回答问题：

    1. 首先，仔细阅读问题和相关法律条文
    2. 分析问题涉及的法律要点
    3. 结合法律条文进行推理
    4. 得出最终结论

    已知信息：
    {context}

    问题：{question}

    让我们一步步思考："""

    BASIC_PROMPT = PromptTemplate(
        template=basic_prompt_template,
        input_variables=["context", "question"]
    )

    COT_PROMPT = PromptTemplate(
        template=cot_prompt_template,
        input_variables=["context", "question"]
    )

    llm = ChatOpenAI(
        model_name="qwen-turbo",
        temperature=0,
        openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # 根据检索方式选择不同的retriever
    if retriever_type == "similarity":
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        prompt = BASIC_PROMPT
    elif retriever_type == "mmr":
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 5, "lambda_mult": 0.5}
        )
        prompt = BASIC_PROMPT
    elif retriever_type == "cot":
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        prompt = COT_PROMPT

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    return chain, llm


def main():
    # 1. 准备数据
    texts = prepare_data("law_data_3k.csv")

    # 2. 构建或加载向量数据库
    vectorstore = build_vectorstore(texts, "law_vectorstore")

    # 3. 构建不同的问答链和直接调用LLM
    similarity_chain, llm = build_qa_chain(vectorstore, "similarity")
    mmr_chain, _ = build_qa_chain(vectorstore, "mmr")
    cot_chain, _ = build_qa_chain(vectorstore, "cot")

    # 4. 测试多个问题
    test_questions = [
        "借款人去世，继承人是否应履行偿还义务？",
        "如何通过法律手段应对民间借贷纠纷？",
        "没有赡养老人就无法继承财产吗？",
        "谁可以申请撤销监护人的监护资格？",
        """你现在是一个精通中国法律的法官...""",  # 保持长问题不变
        """你现在是一个精通中国法律的法官..."""   # 保持长问题不变
    ]

    # 创建输出文件
    with open('model_comparison_results.txt', 'w', encoding='utf-8') as f_comparison:
        # 对每个问题进行测试
        for i, question in enumerate(test_questions, 1):
            f_comparison.write(f"\n{'='*50}\n问题 {i}: {question}\n{'='*50}\n")

            # 1. 使用相似性检索的RAG
            f_comparison.write("\n--- 相似性检索RAG结果 ---\n")
            similarity_result = similarity_chain.run(question)
            f_comparison.write(f"{similarity_result}\n")

            # 2. 使用MMR检索的RAG
            f_comparison.write("\n--- MMR检索RAG结果 ---\n")
            mmr_result = mmr_chain.run(question)
            f_comparison.write(f"{mmr_result}\n")

            # 3. 使用CoT Prompt的RAG
            f_comparison.write("\n--- CoT Prompt RAG结果 ---\n")
            cot_result = cot_chain.run(question)
            f_comparison.write(f"{cot_result}\n")

            # 4. 直接使用LLM（无RAG）
            f_comparison.write("\n--- 直接LLM结果（无RAG） ---\n")
            messages = [
                {"role": "system", "content": "你是一个专业的法律顾问。"},
                {"role": "user", "content": question}
            ]
            direct_result = llm.predict_messages(messages).content
            f_comparison.write(f"{direct_result}\n")

            # 在终端显示进度
            print(f"处理完成问题 {i}/{len(test_questions)}")


if __name__ == "__main__":
    main()
