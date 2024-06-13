import os 
import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    os.makedirs('plots', exist_ok=True)
    with open('logs/EGNN_loma_0.json', 'r') as f:
        loma_res = json.load(f)
    with open('logs/EGNN_torch_0.json', 'r') as f:
        torch_res = json.load(f)
    
    # Plot Loss
    plt.style.use('bmh')
    fig, ax = plt.subplots(dpi=300)
    loss_loma = loma_res['loss']
    loss_torch = torch_res['loss']
    
    interval = 10
    x_range = range(1, len(loss_loma) + 1, interval)
    loss_loma = [sum(loss_loma[k:k+interval]) / float(len(loss_loma[k:k+interval])) for k in range(0, len(loss_loma), interval)]
    loss_torch = [sum(loss_torch[k:k+interval]) / float(len(loss_torch[k:k+interval])) for k in range(0, len(loss_torch), interval)]
    ax.plot(x_range, loss_loma, label='EGNN_loma')
    ax.plot(x_range, loss_torch, label='EGNN_torch')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Train Loss')
    ax.legend()
    plt.savefig('plots/loss.pdf')
    
    # Plot Val MAE
    plt.style.use('bmh')
    fig, ax = plt.subplots(dpi=300)
    val_loma = loma_res['val_mae']
    val_torch = torch_res['val_mae']
    
    x_range = range(1, len(val_loma) + 1)
    ax.set_xticks(x_range[::2])
    ax.plot(x_range, val_loma, label='EGNN_loma')
    ax.plot(x_range, val_torch, label='EGNN_torch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.set_title('Validation MAE')
    ax.legend()
    plt.savefig('plots/val.pdf')