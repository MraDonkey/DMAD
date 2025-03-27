from LLaVA.llava.eval.run_llava import disable_torch_init, get_model_name_from_path, load_pretrained_model, eval_model
import google.generativeai as genai  # pip install -q -U google-generativeai
from pathlib import Path
import time
import base64
import requests


completion_tokens = 0
prompt_tokens = 0 


def get_usage(model):
    global completion_tokens, prompt_tokens
    if model == "gpt-4o":
        cost = completion_tokens / 1000 * 0.015 + prompt_tokens / 1000 * 0.005
    elif model == "gpt-4o-mini-2024-07-18":
        cost = completion_tokens / 1000 * 0.0006 + prompt_tokens / 1000 * 0.00015
    elif model == "gemini-1.5-flash":
        cost = completion_tokens / 10**6 * 0.075 + prompt_tokens / 10**6 * 0.3
    elif "llava" in model:
        cost = 0
    total_tokens = completion_tokens + prompt_tokens
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "total_tokens": total_tokens, "cost": cost}


def load_model(args, dev = "cuda"):
    if args.model == "llava-1.6-13b":
        # Model
        disable_torch_init()

        args.model_path = "liuhaotian/llava-v1.6-vicuna-13b"
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path, args.model_base, model_name, device = dev
        )
        if dev == "cpu":
            model = model.float()
    elif args.model == "llava-1.6-7b":
        # Model
        disable_torch_init()

        args.model_path = "liuhaotian/llava-v1.6-vicuna-7b"
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path, args.model_base, model_name
        )
        if dev == "cpu":
            model = model.float()
            
    return model_name, tokenizer, model, image_processor 


class Gemini:
    def __init__(self, model="gemini-pro-vision"):
        self.model = genai.GenerativeModel(model)

    def get_response(self, args) -> str:
        global prompt_tokens, completion_tokens
        # Query the model
        text = ""
        counts = 0
        assert args.messages or args.query != None
        while len(text) < 1 and counts < 25:
            image_path = Path(args.image_path)
            image = {
                "mime_type": f"image/{image_path.suffix[1:].replace('jpg', 'jpeg')}",
                "data": image_path.read_bytes()
            }
            if args.system != None:
                self.system_instruction = args.system
            if args.messages != None:
                roles = ["user", "model"]
                messages = []
                assert len(args.messages) % 2 == 1
                for i, message in enumerate(args.messages):
                    if i == 0:
                        messages.append({"role": roles[i%2], "parts": [image, message]})
                    else:
                        messages.append({"role": roles[i%2], "parts": [message]})
            elif args.query:
                messages = [image, args.query]
            try:
                response = self.model.generate_content(messages)
                text = response.text
                prompt_tokens += response.usage_metadata.prompt_token_count
                completion_tokens += response.usage_metadata.candidates_token_count
            except Exception as error:
                text = ""
                print(error)
                print("Sleeping for 10 seconds")
                time.sleep(10)
            counts += 1
        if counts == 25:
            return None
        return text.strip()
    
    
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def MLLM_load(args):
    model_name = None
    tokenizer = None
    model = None
    image_processor = None
    model_gemini = None
    if "llava" in args.model:
        model_name, tokenizer, model, image_processor = load_model(args)
    elif "gemini" in args.model:
        genai.configure(api_key=args.google_api_key)
        model_gemini = Gemini(model=args.model)
    return model_name, tokenizer, model, image_processor, model_gemini


def MLLM_generate(args):
    answer = None
    counts = 0
    while(answer == None):
        counts += 1
        if "gemini" in args.model:
            answer = args.model_gemini.get_response(args)       
        elif "llava" in args.model:
            messages_origin = args.messages
            if args.messages != None:
                messages = []
                roles = ["USER", "ASSISTANT"]
                assert len(args.messages) % 2 == 1
                for i, message in enumerate(args.messages):
                    messages.append([roles[i%2], message])
                messages.append([roles[1], None])
                args.messages = messages
            answer = eval_model(args, "cuda", args.model_name, args.tokenizer, args.llava, args.image_processor)
            args.messages = messages_origin
        elif "gpt" in args.model:
            global completion_tokens, prompt_tokens
            image_path = args.image_path
            base64_image = encode_image(image_path)

            headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.openai_api_key}"
            }
            assert args.messages or args.query != None
            messages = []
            if args.system != None:
                messages.append({"role": "system", "content": [{"type": "text", "text": args.system}]})
            if args.messages != None:
                roles = ["user", "assistant"]
                assert len(args.messages) % 2 == 1
                for i, message in enumerate(args.messages):
                    if i == 0:
                        messages.append({"role": roles[i%2], 
                                         "content": [{"type": "text", "text": message},
                                                     {"type": "image_url",
                                                        "image_url": {
                                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                                        }
                                                        }]})
                    else:
                        messages.append({"role": roles[i%2], "content": [{"type": "text", "text": message}]})
            else:
                messages.append(
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": args.query
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                    }
                ]
                }
                )
            payload = {
            "model": args.model,
            "messages": messages,
            }

            response = requests.post(args.openai_completions_url, headers=headers, json=payload).json()
            answer = response["choices"][0]["message"]["content"]
            prompt_tokens += response["usage"]["prompt_tokens"]
            completion_tokens += response["usage"]["completion_tokens"]
            
            try:
                if args.mode == "choose":
                    prompt_choose_tokens += response["usage"]["prompt_tokens"]
                    completion_choose_tokens += response["usage"]["completion_tokens"]
                elif args.mode == "generate":
                    prompt_generate_tokens += response["usage"]["prompt_tokens"]
                    completion_generate_tokens += response["usage"]["completion_tokens"]
            except:
                pass
            print(answer)
    return answer

