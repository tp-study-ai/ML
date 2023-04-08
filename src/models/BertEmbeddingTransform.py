import torch


class BertEmbeddingTransform(object):
    def __init__(self, bert_model, tokenizer, device='cpu'):
        bert_model.eval()
        bert_model = bert_model.to(device)
        bert_model.share_memory()
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, sample):
        code_tokens = self.tokenizer.tokenize(sample)
        tokens = code_tokens
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        done_tok = torch.split(torch.tensor(tokens_ids, device=self.device), 510)
        with torch.no_grad():
            embedings = []
            for input_tok in done_tok:
                input_tok = torch.cat(
                    (torch.tensor([0], device=self.device), input_tok, torch.tensor([2], device=self.device)))
                temp = self.bert_model(input_tok.clone().detach()[None, :], output_hidden_states=True)
                embedings.append(temp[1][-2])
            return torch.concat(embedings, dim=1).squeeze().mean(dim=0)
