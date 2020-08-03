import torch
import json
import copy
import numpy as np

from colr import color


class Glimpse(object):
    def __init__(self, model, embed_name, id2word, id2label, tokenizer, special_tokens, loss_pos=None, step=20):
        self.model = copy.deepcopy(model.cpu())
        self.state_dict = self.model.state_dict()
        self.weight_name = embed_name + '.weight'
        self.id2word = id2word
        self.id2label = id2label
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.embedding_value = copy.deepcopy(self.state_dict[self.weight_name].data)
        self.embedding_index = self._get_embedding_index()
        self.loss_pos = loss_pos
        self.step = step
  
    def _get_embedding_index(self):
        return list(self.state_dict.keys()).index(self.weight_name)

    def _normalization(self, scores):
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
        print("".join([self._plot_colored_text(x, y) for x, y in zip([self.id2word[idx] for idx in input_idx], score)]))
    
    def _plot_label(self, predict_id, golden):
        predict = self.id2label[predict_id]
        
        if golden is not None:
            if isinstance(golden, int):
                golden = self.id2label[golden]
            if predict == golden:
                print(" ".join([color("Label: ", fore=(0, 0, 0), style="bold"), color(golden, fore=(0, 200, 0), style="bold")]))
                print(" ".join([color("Prediction: ", fore=(0, 0, 0), style="bold"), color(predict, fore=(0, 200, 0), style="bold")]))
            else:
                print(" ".join([color("Label: ", fore=(0, 0, 0), style="bold"), color(golden, fore=(0, 200, 0), style="bold")]))
                print(" ".join([color("Prediction: ", fore=(0, 0, 0), style="bold"), color(predict, fore=(220, 0, 0), style="bold")]))
        else:
            print(f"Prediction: {predict}")
#             print("".join([color("Prediction: ", fore=(0, 0, 0), style="bold"), color(predict, fore=(0, 200, 0), style="bold")]))
        print('------------------')
        
    def view(self, text, label=None):
        input_idx = self.tokenizer(text)
        
        score, label_id = self._integrated_gradients(input_idx)
        
        self._plot_label(label_id, label)
        self._plot_text(input_idx, score)
        
    def _integrated_gradients(self, input_idx):
        input_idx = torch.tensor([input_idx], dtype=torch.long)
        
        label_id = None
        with torch.no_grad():
            pred = self.model(input_idx)
            pred = pred.cpu().detach().numpy()
            label_id = np.argmax(pred[0])
            target = torch.tensor(label_id, dtype=torch.long)
    
        path_gradients = []
        for i in range(self.step):
            alpha = 1.0 * i / (self.step - 1)
#             model.embeddings.weight.data = torch.nn.Parameter(embedding_value * alpha, requires_grad=True)
            self.state_dict["embeddings.weight"] = torch.nn.Parameter(self.embedding_value * alpha, requires_grad=True)
            self.model.load_state_dict(self.state_dict)

            self.model.zero_grad()
            if self.loss_pos is not None:
                output = self.model(input_idx, target)
                loss = output[self.loss_pos]
            else:
                loss = self.model(input_idx, target)
            loss.backward()

#             grad = model.embeddings.weight.grad
            grad = list(self.model.named_parameters())[self.embedding_index][1].grad
            path_gradients.append(grad.cpu().detach().numpy())

        ig = np.array(path_gradients)
        ig = (ig[:-1] + ig[1:]) / 2.0  # trapezoidal rule

        integral = np.average(ig, axis=0)
        integrated_gradients = self.embedding_value.cpu().detach().numpy() * integral
        integrated_gradients = np.sum(integrated_gradients, axis=-1)
        integrated_gradients = np.abs(integrated_gradients)

        score = []
        for idx in input_idx[0]:
            if idx not in self.special_tokens:
                weight = integrated_gradients[idx]
                score.append(weight)
        
        score_normal = self._normalization(score)
        return score_normal, label_id