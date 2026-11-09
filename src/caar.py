import torch
from generate import BasicRAG, nlp


class CAARRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def modifier(self, text, tokens, token_scores, question=None):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        tid = 0

        for sid, sent in enumerate(sentences):
            tl, tr = tid, tid
            if sid == len(sentences) - 1:
                tl, tr = tid, len(tokens)
            else:
                for i in range(tid + 1, len(tokens)):
                    seq = ""
                    for j, tok in enumerate(tokens[tl:i]):
                        if j == 0:
                            seq = tok
                        elif tok in ['.', ',', '!', '?', ';', ':', ')', ']', '}', "'", '"'] or tok.startswith("'"):
                            seq += tok
                        else:
                            seq += " " + tok
                    if sent in seq:
                        tr = i
                        break
                tid = tr

            scores = token_scores[tl:tr]

            thres = [1 if score < self.hallucination_threshold else 0 for score in scores]

            if 1 in thres:
                if "check_real_words" in self.__dict__ and self.check_real_words:
                    doc = nlp(sent)
                    real_words = set(token.text for token in doc if token.pos_ in
                        ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
                    def match(tok):
                        for word in real_words:
                            if word in tok:
                                return True
                        return False
                    for i in range(len(thres)):
                        if not match(tokens[tl + i]):
                            thres[i] = 0

                if "check_question_overlap" in self.__dict__ and self.check_question_overlap and question is not None:
                    question_lower = question.lower()

                    for i in range(len(thres)):
                        if thres[i] == 1:
                            tok_lower = tokens[tl + i].lower().strip()
                            if tok_lower in question_lower:
                                thres[i] = 0

                if not any(thres):
                    continue

                prev = "" if sid == 0 else " ".join(sentences[:sid])
                return True, prev, tokens[tl:tr], thres

        return False, text, None, None


    def keep_real_words_with_head_selection(self, prev_text, curr_tokens, curr_hit, attentions_layer_id=-1):
        curr_text = " ".join(curr_tokens)
        all_text = prev_text + " " + curr_text
        input_ids = self.generator.tokenizer.encode(all_text, return_tensors="pt")
        input_ids = input_ids.to(self.generator.model.device)
        input_length = input_ids.shape[1]
        tokens_tmp = self.generator.tokenizer.convert_ids_to_tokens(input_ids[0])

        atten_tmp = self.generator.model(input_ids, output_attentions=True).attentions[attentions_layer_id][0]

        range_ = []
        for i, t in enumerate(tokens_tmp):
            if i == 0 or t.startswith(self.generator.space_token) or input_ids[0][i] == 13:
                range_.append([i, i])
            else:
                range_[-1][-1] += 1
        tokens = []
        for r in range_:
            tokenseq = "".join(tokens_tmp[r[0]: r[1]+1]).replace(self.generator.space_token, "")
            tokens.append(tokenseq)

        curr_st = len(tokens) - len(curr_tokens)

        num_heads = atten_tmp.shape[0]

        if hasattr(self, 'attention_head_ratio'):
            head_ratio = self.attention_head_ratio
        else:
            head_ratio = 0.25

        num_keep_heads = max(1, int(num_heads * head_ratio))

        head_variances = []
        for head_idx in range(num_heads):
            head_attn = atten_tmp[head_idx]
            variance = torch.var(head_attn).item()
            head_variances.append((variance, head_idx))

        head_variances.sort(reverse=True, key=lambda x: x[0])
        selected_head_indices = [idx for _, idx in head_variances[:num_keep_heads]]
        selected_head_indices.sort()

        selected_heads = atten_tmp[selected_head_indices]
        atten_tmp = torch.mean(selected_heads, dim=0)

        attns = []
        for r in range_:
            att = torch.zeros(input_length)
            for i in range(r[0], r[1] + 1):
                if i == 0:
                    continue
                v = atten_tmp[i-1][:r[0]]
                v = v / v.sum()
                t = torch.zeros(input_length)
                t[:r[0]] = v
                att += t
            att /= (r[1] - r[0] + 1)
            att = torch.tensor([att[rr[0]:rr[1]+1].sum().detach().item() for rr in range_])
            attns.append(att)

        forward_attns = torch.zeros(len(tokens))
        hit_cnt = 0
        for i in range(len(curr_hit)):
            if curr_hit[i] == 1:
                forward_attns += attns[curr_st + i]
                hit_cnt += 1
        forward_attns /= hit_cnt
        forward_attns = forward_attns.tolist()

        doc = nlp(all_text)
        real_words = set(token.text for token in doc if token.pos_ in
                      ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])

        def match(token):
            for word in real_words:
                if word in token:
                    return True
            return False

        real_pairs = []
        for i in range(len(tokens)):
            tok, att = tokens[i], forward_attns[i]
            if i >= curr_st and curr_hit[i - curr_st]:
                continue
            if match(tok):
                real_pairs.append((att, tok, i))

        if "retrieve_keep_top_k" in self.__dict__:
            top_k = min(self.retrieve_keep_top_k, len(real_pairs))
        elif "retrieve_keep_ratio" in self.__dict__:
            top_k = int(len(real_pairs) * self.retrieve_keep_ratio)

        real_pairs = sorted(real_pairs, key = lambda x:x[0], reverse=True)
        real_pairs = real_pairs[:top_k]
        real_pairs = sorted(real_pairs, key = lambda x:x[2])
        return " ".join([x[1] for x in real_pairs])

    def keep_real_words(self, prev_text, curr_tokens, curr_hit, attentions_layer_id= -1):
        curr_text = " ".join(curr_tokens)
        all_text = prev_text + " " + curr_text
        input_ids = self.generator.tokenizer.encode(all_text, return_tensors="pt")
        input_ids = input_ids.to(self.generator.model.device)
        input_length = input_ids.shape[1]
        tokens_tmp = self.generator.tokenizer.convert_ids_to_tokens(input_ids[0])

        atten_tmp = self.generator.model(input_ids, output_attentions=True).attentions[attentions_layer_id][0]
        range_ = []
        for i, t in enumerate(tokens_tmp):
            if i == 0 or t.startswith(self.generator.space_token) or input_ids[0][i] == 13:
                range_.append([i, i])
            else:
                range_[-1][-1] += 1
        tokens = []
        for r in range_:
            tokenseq = "".join(tokens_tmp[r[0]: r[1]+1]).replace(self.generator.space_token, "")
            tokens.append(tokenseq)

        curr_st = len(tokens) - len(curr_tokens)

        atten_tmp = torch.mean(atten_tmp, dim=0)

        attns = []
        for r in range_:
            att = torch.zeros(input_length)
            for i in range(r[0], r[1] + 1):
                if i == 0:
                    continue
                v = atten_tmp[i-1][:r[0]]
                v = v / v.sum()
                t = torch.zeros(input_length)
                t[:r[0]] = v
                att += t
            att /= (r[1] - r[0] + 1)
            att = torch.tensor([att[rr[0]:rr[1]+1].sum().detach().item() for rr in range_])
            attns.append(att)

        forward_attns = torch.zeros(len(tokens))
        hit_cnt = 0
        for i in range(len(curr_hit)):
            if curr_hit[i] == 1:
                forward_attns += attns[curr_st + i]
                hit_cnt += 1
        forward_attns /= hit_cnt
        forward_attns = forward_attns.tolist()

        doc = nlp(all_text)
        real_words = set(token.text for token in doc if token.pos_ in
                      ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])

        def match(token):
            for word in real_words:
                if word in token:
                    return True
            return False

        real_pairs = []
        for i in range(len(tokens)):
            tok, att = tokens[i], forward_attns[i]
            if i >= curr_st and curr_hit[i - curr_st]:
                continue
            if match(tok):
                real_pairs.append((att, tok, i))

        if "retrieve_keep_top_k" in self.__dict__:
            top_k = min(self.retrieve_keep_top_k, len(real_pairs))
        elif "retrieve_keep_ratio" in self.__dict__:
            top_k = int(len(real_pairs) * self.retrieve_keep_ratio)

        real_pairs = sorted(real_pairs, key = lambda x:x[0], reverse=True)
        real_pairs = real_pairs[:top_k]
        real_pairs = sorted(real_pairs, key = lambda x:x[2])
        return " ".join([x[1] for x in real_pairs])

    def inference(self, question, demo, case):
        text = ""
        while True:
            old_len = len(text)
            prompt = "".join([d["case"]+"\n" for d in demo])
            tmp_li = [case, text]
            prompt += " ".join(s for s in tmp_li if len(s) > 0)
            layer_id = getattr(self, 'caar_layer_id', 20)
            caar_context_text = " ".join(s for s in [case, text] if len(s) > 0)

            new_text, tokens, token_scores = self.generator.generate_caar(
                prompt,
                self.generate_max_length,
                layer_id=layer_id,
                caar_context_text=caar_context_text
            )

            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)

            hallucination, ptext, curr_tokens, curr_hit = self.modifier(
                new_text, tokens, token_scores, question=question
            )

            if not hallucination:
                text = text.strip() + " " + new_text.strip()
            else:
                forward_all = [question, text, ptext]
                forward_all = " ".join(s for s in forward_all if len(s) > 0)

                def fetch_last_n_tokens(text, num, tokenizer=self.generator.tokenizer):
                    tokens = tokenizer.tokenize(text)
                    if num >= len(tokens):
                        return text
                    last_n_tokens = tokens[-num:]
                    last_n_sentence = ' '.join(last_n_tokens)
                    return last_n_sentence

                if self.query_formulation == "current":
                    retrieve_question = " ".join(curr_tokens)

                elif self.query_formulation == "current_wo_wrong":
                    retrieve_question = " ".join(
                        list(curr_tokens[i] if curr_hit[i] == 0 else "" for i in range(len(curr_tokens)))
                    )

                elif self.query_formulation == "forward_all":
                    retrieve_question = forward_all

                elif self.query_formulation == "last_sentence":
                    retrieve_question = self.get_last_sentence(forward_all)

                elif self.query_formulation == "last_n_tokens":
                    assert "retrieve_keep_top_k" in self.__dict__
                    retrieve_question = fetch_last_n_tokens(
                        forward_all, self.retrieve_keep_top_k)

                elif self.query_formulation == "real_words": 
                    retrieve_question = self.keep_real_words(
                        prev_text = question + " " + text + " " + ptext, 
                        curr_tokens = curr_tokens,
                        curr_hit = curr_hit,
                        attentions_layer_id=layer_id,
                    )

                elif self.query_formulation == "real_words_head_select":
                    retrieve_question = self.keep_real_words_with_head_selection(
                        prev_text = question + " " + text + " " + ptext,
                        curr_tokens = curr_tokens,
                        curr_hit = curr_hit,
                        attentions_layer_id=layer_id,
                    )
                elif self.query_formulation == "init":
                    retrieve_question = question
                else:
                    raise NotImplementedError(self.query_formulation)
                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)

                prompt = "".join([d["case"]+"\n" for d in demo])
                prompt += "Context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i+1}] {doc}\n"
                prompt += "Answer in the same format as before.\n"
                tmp_li = [case, text, ptext.strip()]
                prompt += " ".join(s for s in tmp_li if len(s) > 0)
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                new_text = self.get_top_sentence(new_text)
                tmp_li = [text.strip(), ptext.strip(), new_text.strip()]
                text = " ".join(s for s in tmp_li if len(s) > 0)

            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text
