from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. טען את המודל וה-tokenizer ---
model_name = "avichr/heBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# --- 2. פונקציה להפקת embedding ממשפט ---
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # ניקח את ה-[CLS] token (המייצג את המשפט כולו)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# --- 3. טען את קובץ השאלות-תשובות ---
df = pd.read_csv("C:/Users/Shilo/Desktop/my_project/whatsapp_messages/qa_dataset.csv")

# ודא שאין ערכים חסרים
df = df.dropna(subset=['question', 'answer'])

# --- 4. הפקת embeddings לכל השאלות ---
print("מחשב Embeddings של השאלות... זה עשוי לקחת רגע...")
question_embeddings = [get_embedding(q) for q in df["question"]]


while True:

    # --- 5. קבלת שאלה חדשה והשוואה ---
    user_question = input("\nשאל שאלה לרב: ")
    user_embedding = get_embedding(user_question)
    if user_question == "":
        break

    # חישוב דמיון קוסינוסי
    similarities = cosine_similarity([user_embedding], question_embeddings)[0]
    best_match_index = similarities.argmax()

    # --- 6. הצגת תוצאה ---
    print("\n❓ השאלה שהכי דומה לשאלתך:")
    print(df.iloc[best_match_index]["question"])
    print("\n✅ תשובת הרב:")
    print(df.iloc[best_match_index]["answer"])
