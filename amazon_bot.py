import openai
api_key = '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'
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
