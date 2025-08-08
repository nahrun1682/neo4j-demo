import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

def get_available_models():
    """OpenAI APIから利用可能なモデル一覧を取得"""
    api_key = os.getenv("OPENAI_API_KEY")
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.get("https://api.openai.com/v1/models", headers=headers)
        response.raise_for_status()  # エラーがあれば例外を発生
        
        models_data = response.json()
        models = [model["id"] for model in models_data["data"]]
        
        print("🤖 Available OpenAI Models:")
        for model in sorted(models):
            print(f"  - {model}")
        
        return models
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching models: {e}")
        return []

def main():
    # .envファイルから環境変数を読み込む
    load_dotenv()
    
    # 利用可能なモデルを取得
    # available_models = get_available_models()
    # print(f'available_models: {available_models}')

    # OpenAIクライアントを初期化
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    question = """
    あなたはパーティー会場にいる。
    ルーシーと幼女の会話が聞こえてきた。

    どうやらルーシーは「1〜100」のうち、いずれかひとつの数字が書かれた紙をもっているようだ。

    ルーシーは幼女に言った。

    「私に次の4つの質問をしてね。その答えで私の持っている数字が分かるよ」

    ・その数字は2で割り切れる？
    ・その数字は3で割り切れる？
    ・その数字は5で割り切れる？
    ・その数字は7で割り切れる？

    ルーシーは、「私の数字は◯で割り切れるよ」というように4つの質問の答えを幼女にこっそり教えていった。

    ゴキゲンなパーティー参加者がうるさいので、残念ながらあなたが聞き取れたのは4つの答えのうち1つだけだった。

    しかし、それを聞いて「なるほど。その数字で割り切れるんだな」と分かった瞬間、あなたはルーシーの持っている数字が何なのか分かった。

    いったいルーシーの数字は何だろう？

    なお、ルーシーはつねに真実を語る。
    また、あなたはきわめて論理的な存在である。
    """
    
    print(f"Question: {question}")
    print("Answer: ", end="", flush=True)
    
    # ストリーミングでチャット補完
    stream = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "user", "content": question}
        ],
        stream=True,  # ストリーミングを有効化
        reasoning_effort="medium"
    )
    
    # ストリームからレスポンスを一文字ずつ表示
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print()  # 最後に改行


if __name__ == "__main__":
    main()
