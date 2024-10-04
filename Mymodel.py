import torch
from transformers import AutoImageProcessor, ViTModel
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput
from PIL import Image
import timm
import torch.nn as nn


class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1,1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0,3,1,2)

        return self.classifier(embeddings)


class ViTForSemanticSegmentation(nn.Module):
    def __init__(self, model, num_labels, device):
        super(ViTForSemanticSegmentation, self).__init__()

        self.model = model
        self.classifier = LinearClassifier(768, 14, 14, num_labels)

    def forward(self, pixel_values, output_hidden_states=False, output_attentions=False, labels=None):
        # use frozen features
        with torch.no_grad():
            outputs = self.model(pixel_values)
        # get the patch embeddings - so we exclude the CLS token
        patch_embeddings = outputs.last_hidden_state[:,1:,:]
        # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(patch_embeddings)
        logits = nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False) 
        '''
        logits = nn.functional.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ) 
        '''

        #print("logits shape: ", logits.size())
        #print("labels shape: ", labels.size())
        loss = None
        if labels is not None:
            # important: we're going to use 0 here as ignore index instead of the default -100
            # as we don't want the model to learn to predict background
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(logits.squeeze(), labels.squeeze())
        
        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
     

def load_model(device, label_num=20):

    #vitmodel = timm.create_model('vit_small_patch16_224', pretrained=True)
    vitmodel = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    mymodel = ViTForSemanticSegmentation(vitmodel, label_num, device)

    mymodel.to(device)
    print("model load successfully")
    return mymodel

if __name__ == '__main__':
   
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("check model")
    model = load_model( device)


