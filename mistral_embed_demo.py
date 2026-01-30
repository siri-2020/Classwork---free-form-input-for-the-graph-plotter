from mistralai import Mistral
import dotenv
import os
import logging
import json
import numpy as np


logging.basicConfig(
    filename='activity.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    encoding='utf-8'
)

dotenv.load_dotenv()

# Initialize client
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

input_texts = [
    "I'm thinking about the purpose of life. It's very interesting and can lead to deep insights and meaningfull conversations with like-minded people.",
    "What is bothering me now is how to get out of the class faster and go for a lunch",
    "I have and interesting idea of tech-based IT project, like to have more free time to experiment and make it work faster!",
    "I'm leading in my class leaderboard for clash royale mobile game, plan to train more and get a place in the competition team",
    "I have found nice manga but read all the chapter available for now - and the next should be published next month... sad...",
    "For the long time seen no interesting anime - plan the check what is available now and may be spend a day with it!",
    "Tired sitting in the class and with the laptop... want to do some sports, maybe badminton or bicicle!"
]

nicknames = ["Peepat", "Meg", "Arm", "Focus", "North", "Shin", "Cheetah"]

res = client.embeddings.create(model="mistral-embed", inputs=input_texts)
base_embeddings = [res.data[i].embedding for i in range(len(input_texts))]
print(base_embeddings[0])

# Helper function for cosine similarity


def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def find_match(user_nickname, user_message):
    res = client.embeddings.create(
        model="mistral-embed", inputs=[user_message])
    new_emb = res.data[0].embedding

    similarities = []
    for i in range(len(base_embeddings)):
        sim = cosine_similarity(new_emb, base_embeddings[i])
        name = nicknames[i] if i < len(nicknames) else "Anonymous"
        similarities.append((sim, name, input_texts[i]))

    similarities.sort(reverse=True)
    top_3 = similarities[:3]

    logging.info(f"New User: {user_nickname} | Msg: {user_message}")
    logging.info(f"Top 3 Sim: {[(s[1], s[0]) for s in top_3]}")

    candidates_str = "\n".join([f"- {item[1]}: {item[2]}" for item in top_3])

    prompt = f"""
    User says: "{user_message}"
    Potential friends:
    {candidates_str}
    
    Task: Select the ONE best friend from the list who has a truly relevant or similar interest.
    If no one is relevant, say "No good match found".
    If there is a match, reply: "Recommended Friend: [Name] because [Reason]"
    """

    chat_res = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    ai_recommendation = chat_res.choices[0].message.content

    input_texts.append(user_message)
    base_embeddings.append(new_emb)
    nicknames.append(user_nickname)

    return top_3, ai_recommendation


def process_friend_request(user_nickname, user_message):
    res = client.embeddings.create(
        model="mistral-embed", inputs=[user_message])
    new_emb = res.data[0].embedding

    matches = []
    for i in range(len(base_embeddings)):
        sim = cosine_similarity(new_emb, base_embeddings[i])
        name = nicknames[i] if i < len(nicknames) else "Anonymous"
        matches.append({
            "score": float(sim),
            "nickname": name,
            "text": input_texts[i]
        })

    matches.sort(key=lambda x: x["score"], reverse=True)
    top_3 = matches[:3]

    logging.info(f"New User: {user_nickname} | Msg: {user_message}")
    logging.info(f"Top-3: {[m['nickname'] for m in top_3]}")

    candidates_str = "\n".join(
        [f"- {m['nickname']}: {m['text']}" for m in top_3])
    prompt = f"""
    User ({user_nickname}): "{user_message}"
    Top 3 Similar Matches:
    {candidates_str}
    
    Task: Pick the best compatible friend from the list.
    If match found, say: "Recommended: [Name] - [Reason]"
    If no good match, say: "No close match found."
    """

    chat_res = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    ai_comment = chat_res.choices[0].message.content

    input_texts.append(user_message)
    base_embeddings.append(new_emb)
    nicknames.append(user_nickname)

    return top_3, ai_comment