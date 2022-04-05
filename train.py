import os.path
from model import Transformer
from torch import nn
from config import *
from dataloader import train_loader, test_loader
from tqdm import tqdm

filepath = os.path.join(checkpoint_path, 'checkpoint_best_model.pth')
if os.path.isfile(filepath):
    print("model loaded!")
    model = torch.load(filepath)
else:
    model = Transformer()
model = model.to(device)
print(model.parameters)

loss_fn = nn.CrossEntropyLoss(ignore_index=0)
loss_fn = loss_fn.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=lr)


best_test_loss = 1e5
print("start training!")
for epoch in range(num_epochs):
    model.train()
    with tqdm(train_loader, unit="batch") as tepoch:
        for enc_inputs, dec_inputs, dec_outputs in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1} train")
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
            outputs, enc_attention, dec_attention1, dec_attention2 = model(enc_inputs, dec_inputs)
            loss = loss_fn(outputs, dec_outputs.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item())
    model.eval()
    total_loss = 0.
    with tqdm(test_loader, unit="batch") as tepoch:
        for enc_inputs, dec_inputs, dec_outputs in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1} test")
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
            outputs, enc_attention, dec_attention1, dec_attention2 = model(enc_inputs, dec_inputs)
            loss = loss_fn(outputs, dec_outputs.view(-1))
            total_loss += loss.item()
            tepoch.set_postfix(total_loss=total_loss)
    if total_loss < best_test_loss:
        filepath = os.path.join(checkpoint_path, 'checkpoint_best_model.pth')
        torch.save(model, filepath)
        best_test_loss = total_loss
