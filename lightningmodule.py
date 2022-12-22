import torch
import torch.nn as nn
import pytorch_lightning as pl
from dataset import Dataset
import torchmetrics

class Seq2Seq(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.model = nn.Sequential(
            encoder,
            decoder
        )

        # Define Optimizer, Loss Function, and Evaluation Metric
        self.criterion = nn.MSELoss()

        # use RMSE as evaluation metric. USe MSE then pass argument False
        self.metric  = torchmetrics.MeanSquaredError(squared=False)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters())
        return opt
        
    def forward(self, x):
        print("FORWARD PASS")
        # set initial states
        h_n, c_n = self.encoder(x)
        # print(f"Encoder output: {h_n.shape}, {c_n.shape}")
        # create initial input (start with zeros)
        decoder_input = torch.zeros(x.size(0), 1, 1)
        print(decoder_input.device)
        # init output tensor
        # print(x.size(1), x.size(0))
        outputs = torch.zeros(x.size(1), x.size(0),  1)
        print(outputs.device)
        # decode hidden state of last time step
        for t in range(x.size(1)):
            decoder_output, (h_n, c_n) = self.decoder(decoder_input, h_n, c_n)
            # print(f"Decoder output: {decoder_output.shape}, {h_n.shape}, {c_n.shape}")
            # decoder_output = decoder_output.squeeze(1)
            outputs[t] = decoder_output.squeeze(1)
            decoder_input = decoder_output
        return outputs


    def training_step(self, batch, batch_idx):
        # get input and targets and get to cuda. Remember, input is on [:,:,0] and [:,:,2], while target is on [:,:,1]
        input = batch[:, :, [0,2]]
        target = batch[:, :, 1]
        # forward prop
        print("BEFORE FORWARD PASS")
        output = self.model(input)
        # output = [batch size, seq len, output dim]
        # target = [batch size, seq len, output dim]
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        target = target.contiguous().view(-1, output_dim)
        # loss and backprop
        loss = criterion(output, target)
        # record loss
        epoch_loss += loss.item()

        rmse_value = metric(output, target)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss, rmse_value

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x[0] for x in outputs]).mean()
        avg_rmse = torch.stack([x[1] for x in outputs]).mean()
        self.log('avg_train_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('avg_train_rmse', avg_rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    
