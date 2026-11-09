import logging
import spacy
import torch
from retriever import BM25, SGPT
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_sm")


class BasicGenerator:
    def __init__(self, model_name_or_path):
        logger.info(f"Loading model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model_config = AutoConfig.from_pretrained(model_name_or_path,
                    trust_remote_code = "falcon" in model_name_or_path or "qwen" in model_name_or_path.lower())
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto", 
            trust_remote_code = "falcon" in model_name_or_path or "qwen" in model_name_or_path.lower(),
        )

        self.model_name_or_path = model_name_or_path.lower()
        self.is_qwen = "qwen" in self.model_name_or_path
        self.is_llama = self.model_config.model_type == "llama"

        logger.info(f"Model type detected - Qwen: {self.is_qwen}, Llama: {self.is_llama}")

        if self.is_llama:
            self.space_token = "▁"
        else:
            self.space_token = self.tokenizer.tokenize(' ')[0]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _prepare_input(self, input_text, for_retrive=False):
        if self.is_qwen:
            if for_retrive is False:
                messages = [
                    {"role": "system", "content": '''
    You must follow exactly the example format below. For Question, produce:
    1) A short factual extraction about each relevant person/entity.
    2) A concise linking sentence that draws the conclusion.
    3) The final line must start with "So the answer is " followed by the succinct answer (single word or short phrase). 
    4) Do not mention anythiong except the reasoning process and the answer and do not use serial numbers.
    '''},
                    {"role": "user", "content": input_text}
                ]
            else:
                messages = [
                    {"role": "system", "content": '''
    You are an assistant that extracts concise factual phrases for retrieval.

    Given a question and a list of important words, your task is to extract several short factual phrases **strictly from the given text**. 
    Do not invent, infer, or add any information that does not appear in the text.
    The extracted phrases must appear **in the same order as they occur in the original text**.

    Each phrase should represent a possible fact or relation relevant to answering the question and can be directly used for BM25 or sparse retrieval.

    ### Guidelines
    - Only use words and phrases that appear in the provided text.  
    - Preserve the original order of appearance in the text.  
    - Each phrase should be self-contained and meaningful (e.g., “Scott Derrickson born Denver”).  
    - Combine adjacent related entities, attributes, or relations where possible.  
    - Avoid stop words, pronouns, or vague references (no “he”, “they”, “it”, etc.).  
    - If no valid factual phrase is found, return an empty list (`[]`).  
    - **Output only valid JSON** with the following format:
    ```json
    {
        "phrases": [
        "phrase_1",
        "phrase_2",
        "phrase_3"
        ]
    }```
                     '''}, 
                    {"role": "user", "content": input_text}
                ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return text
        else:
            if for_retrive:
                B_INST, E_INST = "[INST]", "[/INST]"
                B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

                system_prompt = '''You are an assistant that extracts concise factual phrases for retrieval.

    Given a question and a list of important words, your task is to extract several short factual phrases **strictly from the given text**. 
    Do not invent, infer, or add any information that does not appear in the text.
    The extracted phrases must appear **in the same order as they occur in the original text**.

    Each phrase should represent a possible fact or relation relevant to answering the question and can be directly used for BM25 or sparse retrieval.

    ### Guidelines
    - Only use words and phrases that appear in the provided text.  
    - Preserve the original order of appearance in the text.  
    - Each phrase should be self-contained and meaningful (e.g., "Scott Derrickson born Denver").  
    - Combine adjacent related entities, attributes, or relations where possible.  
    - Avoid stop words, pronouns, or vague references (no "he", "they", "it", etc.).  
    - If no valid factual phrase is found, return an empty list (`[]`).  
    - **Output only valid JSON** with the following format:
    ```json
    {
        "phrases": [
        "phrase_1",
        "phrase_2",
        "phrase_3"
        ]
    }```'''

                input_text = f"{B_INST} {B_SYS}{system_prompt}{E_SYS}{input_text.strip()} {E_INST}"
            return input_text

    def generate(self, input_text, max_length, return_logprobs=False, for_retrive=False):
        prepared_text = self._prepare_input(input_text,for_retrive)
        input_ids = self.tokenizer.encode(prepared_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        if return_logprobs:
            outputs = self.model.generate(
                input_ids = input_ids,
                attention_mask = attention_mask,
                max_new_tokens = max_length,
                do_sample = False,
                return_dict_in_generate = True,
                output_scores = True,
            )
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )

            generated_tokens = outputs.sequences[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            tokens = [self.tokenizer.decode(t) for t in generated_tokens[0]]
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            return text, tokens, logprobs

        else:
            outputs = self.model.generate(
                input_ids = input_ids,
                max_new_tokens = max_length,
                attention_mask = attention_mask,
                do_sample = False,
            )
            generated_tokens = outputs[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            return text, None, None


    def generate_caar(self, input_text, max_length, layer_id=20, caar_context_text=None):
        prepared_text = self._prepare_input(input_text)
        input_ids = self.tokenizer.encode(prepared_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
            max_new_tokens=max_length,
            do_sample=False,
        )

        generated_tokens = outputs.sequences[:, input_length:]
        tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens[0])
        text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        end_token_idx = len(tokens)
        for i, token in enumerate(tokens):
            if '<|im_end|>' in token or token == '<|endoftext|>':
                end_token_idx = i
                break

        tokens = tokens[:end_token_idx]
        generated_tokens = generated_tokens[:, :end_token_idx]

        range_ = []
        for i, t in enumerate(tokens):
            if i == 0 or t.startswith(self.space_token) or (i < len(generated_tokens[0]) and generated_tokens[0][i] == 13) or (i > 0 and tokens[i - 1] == '</s>'):
                range_.append([i, i])
            else:
                range_[-1][-1] += 1

        if caar_context_text is None:
            forward_input_ids = input_ids
            context_length = input_length
        else:
            forward_text = self._prepare_input(caar_context_text)
            forward_input_ids = self.tokenizer.encode(forward_text, return_tensors="pt")
            forward_input_ids = forward_input_ids.to(self.model.device)
            context_length = forward_input_ids.shape[1]
        full_sequence = torch.cat([forward_input_ids, generated_tokens], dim=1)

        with torch.inference_mode():
            atten = self.model(full_sequence, output_attentions=True)
        attentions = atten.attentions

        layer_id = min(layer_id, len(attentions) - 1)
        attn = attentions[layer_id][0]

        gen_start = context_length
        gen_end = context_length + end_token_idx

        diag_indices = torch.arange(gen_start, gen_end, device=attn.device)
        diag_vals = attn[:, diag_indices, diag_indices]
        diag_vals = torch.clamp(diag_vals, min=1e-12)
        caar_scores = -diag_vals.log().mean(dim=0)

        seqlist = []
        token_scores = []
        for r in range_:
            tokenseq = "".join(tokens[r[0]: r[1] + 1]).replace(self.space_token, "")
            score = torch.mean(caar_scores[r[0]: r[1] + 1]).item()
            seqlist.append(tokenseq)
            token_scores.append(score)

        return text, seqlist, token_scores


class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve, 
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated, 
            "token_count": self.token - other_counter.token, 
            "sentence_count": self.sentence - other_counter.sentence 
        }


class BasicRAG:
    def __init__(self, args):
        args = args.__dict__
        for k, v in args.items():
            setattr(self, k, v)
        self.generator = BasicGenerator(self.model_name_or_path)
        if "retriever" in self.__dict__:
            self.retriever_type = self.retriever
            if self.retriever_type == "BM25":
                self.retriever = BM25(
                    tokenizer = self.generator.tokenizer,
                    index_name = "wiki" if "es_index_name" not in args else self.es_index_name, 
                    engine = "elasticsearch",
                )
            elif self.retriever_type == "SGPT":
                self.retriever = SGPT(
                    model_name_or_path = self.sgpt_model_name_or_path,
                    sgpt_encode_file_path = self.sgpt_encode_file_path,
                    passage_file = self.passage_file
                )
            else:
                raise NotImplementedError

        self.counter = Counter()

    def retrieve(self, query, topk=1, max_query_length=64):
        self.counter.retrieve += 1
        if self.retriever_type == "BM25":
            _docs_ids, docs = self.retriever.retrieve(
                    queries = [query],
                    topk = topk,
                    max_query_length = max_query_length,
                )
            return docs[0]
        elif self.retriever_type == "SGPT":
            docs = self.retriever.retrieve(
                queries = [query],
                topk = topk,
            )
            return docs[0]
        else:
            raise NotImplementedError

    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else ""
