import torch

# Few Shot Objects
special_tokens_dict = {'additional_special_tokens': ['<title>', '<subtitle>', '<text>']}


class Collator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, arr):

        """
        :param arr:
        :param encode_labels:
        :return:
        """

        x, y = [x[0] for x in arr], [y[1] for y in arr]
        enc_x = self.tokenizer(text=x, padding='longest', truncation=True, max_length=128)
        if isinstance(y[0], str):
            enc_y = self.tokenizer(text=y, padding='longest', truncation=True, max_length=128)
            return (torch.tensor(enc_x['input_ids']), torch.tensor(enc_x['attention_mask']),
                    torch.tensor(enc_y['input_ids']))
        else:
            return torch.tensor(enc_x['input_ids']), torch.tensor(enc_x['attention_mask']), torch.tensor(y)

