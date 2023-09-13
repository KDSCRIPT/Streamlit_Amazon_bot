import openai
api_key = 'API KEY'
openai.api_key = api_key

def chat_with_bot(user_message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ]
    )
    bot_message = response['choices'][0]['message']['content']
    return bot_message
