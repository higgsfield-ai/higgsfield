from .dataset import CompletionDataset

def chat_to_prompt(chat):
    joined = []
    for message in chat:
        joined.append(f"###{message['role'].upper()}: {message['content']}")
    
    prompt = "\n".join(joined)
    prompt += "\n###ASSISTANT: "
    
    return prompt

class ChatCompletionDataset(CompletionDataset):
    ''' OpenAI's api format:
    chats = [
        [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Who won the world series in 2020?"},
          {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
          {"role": "user", "content": "Where was it played?"}
        ],
    ]
    '''
    def __init__(self, chats, chat_to_prompt=chat_to_prompt):
        self.chat_to_prompt = chat_to_prompt
        
        self.chats = chats
        
        items = []
        for chat in self.chats:
            current_chat = []
            last_user = False
            
            for message in chat:
                if message["role"] == "system":
                    current_chat.append(message)
                    
                elif message["role"] == "user":
                    last_user = True
                    current_chat.append(message)
                    
                elif message["role"] == "assistant":
                    if last_user:
                        items.append([
                            [c for c in current_chat], message["content"]
                        ])
                    current_chat.append(message)
                    
        self.items = items
        
    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, idx):
        chat, completion = self.items[idx]
        prompt = self.chat_to_prompt(chat)
        
        return {
            "prompt": prompt,
            "completion": completion
        }