# FAQ-Chatbot
A simple FAQ chatbot using Python, TF-IDF, and cosine similarity.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

questions = [
    "hi",
    "hello",
    "how are you",
    "what is your name",
    "bye"
]

answers = [
    "Hello! How can I help you?",
    "Hello! How can I help you?",
    "I'm fine! How can I help you?",
    "I am an FAQ chatbot.",
    "Goodbye! Have a nice day!"
]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

print("FAQ Chatbot")
print("Type 'exit' to stop.")

while True:
    user = input("You: ")

    if user.lower() == "exit":
        print("Bot: Goodbye!")
        break

    user_vector = vectorizer.transform([user])
    similarity = cosine_similarity(user_vector, question_vectors)

    best_match = similarity.argmax()
    score = similarity[0][best_match]

    if score > 0.3:
        print("Bot:", answers[best_match])
    else:
        print("Bot: Sorry, I don't know the answer.")
