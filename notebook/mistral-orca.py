import replicate

prompt_template = """<|im_start|>system
You are MistralOrca, a large language model trained by Alignment Lab AI. You are given the content of an email and your answer should tell the user if a followup action is expected from the author of the email or if the email is informative only. Only use answers 'followup' or 'informative'.
<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant"""

prompts = [
    "The port of New Jersey usually has good weather.",
    "Pollution has been a real issue in the port of Guayaquil.",
    "Can you remind me the cost of demurrage for the shipment arriving on december 13th?",
    "How long will the cargo detention last for the Wayfair container?"
]

for p in prompts:
    output = replicate.run(
    "nateraw/mistral-7b-openorca:7afe21847d582f7811327c903433e29334c31fe861a7cf23c62882b181bacb88", 
    input={"prompt": p, 
            "temperature":0.2,
            "max_new_tokens":10, 
            "top_p": 0.95,
            "top_k": 50,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "prompt_template": prompt_template}, 
    )


    answer = []
    for element in output:
        answer.append(element.strip())

    print(p, "CLASS:", "".join(answer[1:]))
