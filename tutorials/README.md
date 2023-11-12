# Dataset formats

We support the following dataset formats:

- **Prompt Completion format**
- **The ChatGPT format**
- **The Plain Text format**


Before uploading your dataset to Hugging Face, please make sure that it is in one of the above formats.

We provide a tutorial on how to convert your dataset to each format.

- **Prompt Completion format** (https://github.com/higgsfield-ai/higgsfield/tutorials/prompt_completion.ipynb)
- **ChatGPT format** (https://github.com/higgsfield-ai/higgsfield/tutorials/chatgpt.ipynb)
- **Plain Text format** (https://github.com/higgsfield-ai/higgsfield/tutorials/text_format.ipynb)

### Format: Prompt Completion
```json
prompt_completion = {
        "prompt": [
            "prompt1",
            "prompt2",
        ],
        "completion": [
            "completion1",
            "completion2",
        ]
    }
```

### Format: ChatGPT
```json
chatgpt_format = {
    "chatgpt": [
        [
            {"role": "system", "content": "You are a human."},
            {"role": "user", "content": "No I am not."},
            {"role": "assistant", "content": "I am a robot."},
        ],
    ]
}
```

### Format: Text
```json
text_format = {
    "text": ["text"]
}
```