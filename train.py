import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
from model import ElectronicComponentCNN
import os

# --- 1. Argumentos ---
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="Ruta al directorio de entrenamiento")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--epoch_save", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--save_dir", type=str, default="./checkpoints")
parser.add_argument("--gitsave", type=int, default=0, help="Número de epochs para hacer commit y push automático. 0 = desactivar")

args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

# --- 2. Transformaciones ---
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- 3. Dataset y DataLoader ---
train_dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

# --- 4. Modelo, pérdida y optimizador ---
device = torch.device("cpu") #"cuda" if torch.cuda.is_available() else 
model = ElectronicComponentCNN(num_classes=len(train_dataset.classes))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# --- 5. Entrenamiento ---
for epoch in range(1, args.epochs + 1):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch}/{args.epochs} - Loss: {epoch_loss:.4f}")

    # Guardar checkpoint cada epoch_save
    if epoch % args.epoch_save == 0:
        save_path = os.path.join(args.save_dir, f"model_epoch{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Checkpoint guardado: {save_path}")
    import subprocess

   import os
import subprocess

# Dentro del loop de entrenamiento
if args.gitsave > 0 and epoch % args.gitsave == 0:
    commit_msg = f"Commit automático después de epoch {epoch}"
    try:
        # 1️⃣ git add y commit normal
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)

        # 2️⃣ git push usando GH_TOK
        gh_token = os.environ.get("GH_TOK")
        if gh_token is None:
            print("Variable de entorno GH_TOK no encontrada, no se puede hacer push")
        else:
            # Obtén la URL remota actual
            remotes = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                check=True
            )
            remote_url = remotes.stdout.strip()

            # Solo soporta HTTPS
            if remote_url.startswith("https://"):
                # Insertar token en la URL
                remote_url_with_token = remote_url.replace(
                    "https://", f"https://{gh_token}@"
                )
                subprocess.run(["git", "push", remote_url_with_token], check=True)
                print(f"Git commit y push realizado: {commit_msg}")
            else:
                print("Remote no es HTTPS, no se puede usar GH_TOK para push")
    except subprocess.CalledProcessError as e:
        print("Error al ejecutar git:", e)


# Guardar modelo final
final_path = os.path.join(args.save_dir, "model_final.pth")
torch.save(model.state_dict(), final_path)
print(f"Modelo final guardado: {final_path}")
