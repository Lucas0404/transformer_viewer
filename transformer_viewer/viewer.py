import torch
import json
import copy
import numpy as np

from colr import color

    
class Glimpse(object):
    def __init__(self, model, tokenizer, adaptor_embed, adaptor_model, device, spliter=' ', id2label=None, step=20):
        self.model = copy.deepcopy(model)
        self.tokenizer = tokenizer
        self.adaptor_embed = adaptor_embed
        self.adaptor_model = adaptor_model
        self.spliter = spliter
        self.device = device
        self.id2label = id2label
        self.step = step
        
        self.model.to(device=device)      
        self.embedding_value = copy.deepcopy(self.adaptor_embed(self.model, 'value'))
        self.special_tokens_idx = [self.tokenizer.convert_tokens_to_ids(item) for item in self.tokenizer.special_tokens_map.values()]


    def _normalization(self, scores):
        if len(scores) == 1:
            return [1.0]
        else:
            _max = np.max(scores)
            _min = np.min(scores)
            return [(s-_min)/(_max-_min) for s in scores]

    
    def _plot_colored_text(self, token, score):
        if score >= 0:
            return color(token, fore=(255*score, 0, 0))
        else:
            return color(token, fore=(0, 0, -255*score))

        
    def color_bar(self):
        bar = [self._plot_colored_text(x, y) for x, y in zip("■"*20, [float((i+1)/20) for i in range(20)])]
        bar = [self._plot_colored_text("min ", 0)] + bar
        bar += [self._plot_colored_text(" max", 1)]
        print("".join(bar))

        
    def _plot_text(self, input_idx, score):
        print(self.spliter.join([self._plot_colored_text(x, y) for x, y in zip([self.tokenizer.convert_ids_to_tokens(idx) for idx in input_idx], score)]))

        
    def _plot_label(self, predict_id, golden):
        predict = self.id2label[predict_id] if self.id2label is not None else predict_id

        if golden is not None:
            if isinstance(golden, int):
                golden = self.id2label[golden] if self.id2label is not None else golden
            if predict == golden:
                print(" ".join([color("Label: ", fore=(0, 0, 0), style="bold"), color(golden, fore=(0, 200, 0), style="bold")]))
                print(" ".join([color("Prediction: ", fore=(0, 0, 0), style="bold"), color(predict, fore=(0, 200, 0), style="bold")]))
            else:
                print(" ".join([color("Label: ", fore=(0, 0, 0), style="bold"), color(golden, fore=(0, 200, 0), style="bold")]))
                print(" ".join([color("Prediction: ", fore=(0, 0, 0), style="bold"), color(predict, fore=(220, 0, 0), style="bold")]))
        else:
            print(f"Prediction: {predict}")
        print('------------------')

    
    def view(self, text, label=None):        
        score, input_idx, label_id = self._integrated_gradients(text)
        
        self._plot_label(label_id, label)
        self._plot_text(input_idx, score)
        
        
    def _integrated_gradients(self, text):
        input_idx, label_id = self.adaptor_model(text, self.model, self.tokenizer, self.device)
        target = torch.tensor(label_id, dtype=torch.long)

        path_gradients = []
        for i in range(self.step):
            alpha = 1.0 * i / (self.step - 1)
            weights = self.adaptor_embed(self.model, 'value')
            weights = torch.nn.Parameter(self.embedding_value * alpha, requires_grad=True)

            self.model.zero_grad()
            loss = self.adaptor_model(text, self.model, self.tokenizer, self.device, target)
            loss.backward()

            grad = self.adaptor_embed(self.model, 'grad')
            grad = grad.cpu().detach().numpy()
            path_gradients.append(grad)

        ig = np.array(path_gradients)
        ig = (ig[:-1] + ig[1:]) / 2.0  # trapezoidal rule

        integral = np.average(ig, axis=0)
        integrated_gradients = self.embedding_value.cpu().detach().numpy() * integral
        integrated_gradients = np.sum(integrated_gradients, axis=-1)
        integrated_gradients = np.abs(integrated_gradients)
        
        idxs = []
        score = []
        for idx in input_idx[0]:
            if idx not in self.special_tokens_idx:
                idxs.append(int(idx.cpu().detach().numpy()))
                weight = integrated_gradients[idx]
                score.append(weight)
        
        score_normal = self._normalization(score)
        return score_normal, idxs, label_id[0]