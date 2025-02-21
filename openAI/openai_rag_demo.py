import openai
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def load_file(file_path):
    """ファイルを読み込みテキストを返す"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    """
    テキストをチャンクに分割する
    chunk_size: 1チャンクあたりの文字数
    overlap: チャンク間の重複文字数
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def get_embedding(text, model="text-embedding-ada-002"):
    """OpenAIのEmbedding APIを利用してテキストの埋め込みを取得"""
    client = openai.Client()
    response = client.embeddings.create(
        input=text,
        model=model
    )
    embedding = response.data[0].embedding
    return embedding

def build_embedding_index(chunks):
    """
    各チャンクに対して埋め込みを取得し、
    チャンクテキストと埋め込みベクトルの辞書リストを作成
    """
    index = []
    for chunk in chunks:
        embedding = get_embedding(chunk)
        index.append({"chunk": chunk, "embedding": embedding})
    return index

def cosine_similarity(vec1, vec2):
    """2つのベクトル間のコサイン類似度を計算"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve(query, index, top_k=3):
    """
    クエリに対して、各チャンクの埋め込みとの類似度を計算し、
    上位top_k件を返す
    """
    query_embedding = get_embedding(query)
    scores = []
    for item in index:
        score = cosine_similarity(query_embedding, item["embedding"])
        scores.append(score)
    # 類似度が高い順に並べ替え
    top_indices = np.argsort(scores)[-top_k:][::-1]
    results = [index[i] for i in top_indices]
    return results

def rag_generate(query, retrieved_chunks):
    """
    取得したチャンクを参照情報としてプロンプトに追加し、
    ChatCompletion APIにより回答を生成する
    """
    # 取得したチャンクをひとつの文書にまとめる
    context = "\n\n".join([item["chunk"] for item in retrieved_chunks])
    prompt = (
        f"質問: {query}\n\n"
        f"以下は参考情報です:\n{context}\n\n"
        "上記の情報に基づいて、質問に答えてください。"
    )
    response = openai.chat.completions.create(
         model="gpt-4o-mini-2024-07-18",
         messages=[
             {"role": "system", "content": "あなたは知識豊富なカスタマーサポートエージェントです。"},
             {"role": "user", "content": prompt}
         ]
    )
    answer = response.choices[0].message.content
    return answer

def main(file_path, query):
    # 1. ファイルの読み込み
    text = load_file(file_path)
    # 2. テキストのチャンク分割
    chunks = chunk_text(text)
    # 3. 各チャンクの埋め込み作成（ベクトルデータベースの構築）
    index = build_embedding_index(chunks)
    # 4. クエリに基づく類似度検索（RAGのretrieval部分）
    retrieved_chunks = retrieve(query, index)
    # 5. 取得した情報を用いて回答生成（RAGのgeneration部分）
    answer = rag_generate(query, retrieved_chunks)
    print("回答:")
    print(answer)

if __name__ == "__main__":
    file_path = "openai/AIスキルビルディング.txt"
    query = "Part1 の研修で使用する Kaggle のコンペ名と、その目標精度はいくつですか？"
    
    # 質問例
    # query = "Part0で学習メンバーが身につけるべき前提条件には何がありますか？"
    # query = "Part1 の研修で使用する Kaggle のコンペ名と、その目標精度はいくつですか？"
    # query = "Part2 の研修で使用するデータセットの名前は何ですか？"
    # query = "Part2のHuman Protein Atlasコンペに取り組む前に、何をどのくらい勉強すると書かれていますか？"
    # query = "画像生成AIを学習する際、Google Drive の容量が足りなくなりそうな場合はどうすれば良いと書かれていますか？"
    # query = "研修資料の中に出てくる Slack ルール はどこで確認できますか？"
    # query = "スクラムについて学ぶ際、何分程度を目安にどんな資料を読むよう書かれていますか？"
    main(file_path, query)