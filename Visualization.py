import numpy as np
import matplotlib.pyplot as plt

def summarize_results(title, acc_clean, acc_adv, eps):
    print(f"\n=== {title} ===")
    print(f"Clean accuracy: {acc_clean:.4f}")
    print(f"Attack accuracy (eps={eps}): {acc_adv:.4f}")

def plot_loss_curves_both(res_onehot, res_bit4, save_path=None):
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(res_onehot['train_losses']) + 1), res_onehot['train_losses'], marker='o', label='Train (10-out)')
    plt.plot(range(1, len(res_onehot['val_losses']) + 1), res_onehot['val_losses'], marker='s', label='Val (10-out)')
    plt.plot(range(1, len(res_bit4['train_losses']) + 1), res_bit4['train_losses'], marker='^', label='Train (4-bit)')
    plt.plot(range(1, len(res_bit4['val_losses']) + 1), res_bit4['val_losses'], marker='v', label='Val (4-bit)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves: 10-output vs 4-bit')
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
    plt.show()
def plot_loss_curve(loaded_model, save_path=None):
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(loaded_model.train_losses)+1), loaded_model.train_losses, marker='o', label="Train Loss")
    plt.plot(range(1, len(loaded_model.val_losses)+1), loaded_model.val_losses, marker='s', label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{loaded_model.loss.upper()} Loss Curve")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_predictions_grid(model, X, y, mode_title, decode_pred_fn=None, decode_true_fn=None, n=5, save_path=None):
    idx = np.random.choice(X.shape[0], n, replace=False)
    Xs = X[idx]
    if decode_pred_fn is None and decode_true_fn is None:
        y_true = np.argmax(y[idx], axis=1)
        y_pred = model.predict(Xs)
    else:
        # use provided decoders
        y_true = decode_true_fn(y[idx])
        y_pred = decode_pred_fn(model.forward(Xs))
    fig, axes = plt.subplots(1, n, figsize=(2.2*n, 2.6))
    for i, ax in enumerate(axes):
        ax.imshow(Xs[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
        ax.set_title(f"T:{int(y_true[i])} P:{int(y_pred[i])}")
    plt.suptitle(f"{mode_title}: Predictions (T=true, P=pred)")
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_fgsm_comparison_grid(model, X, Y, eps, mode_title, decode_pred_fn=None, decode_true_fn=None, n=5, save_path=None):
    # compute clean accuracy
    if decode_pred_fn is None and decode_true_fn is None:
        logits_clean = model.forward(X)
        acc_clean = np.mean(np.argmax(logits_clean, 1) == np.argmax(Y, 1))
    else:
        y_pred_clean = decode_pred_fn(model.forward(X))
        y_true_clean = decode_true_fn(Y)
        acc_clean = np.mean(y_pred_clean == y_true_clean)

    # generate adversarial examples
    from Functions import fgsm_attack
    X_adv = fgsm_attack(model, X, Y, epsilon=eps)

    # compute adv accuracy
    if decode_pred_fn is None and decode_true_fn is None:
        logits_adv = model.forward(X_adv)
        acc_adv = np.mean(np.argmax(logits_adv, 1) == np.argmax(Y, 1))
    else:
        y_pred_adv = decode_pred_fn(model.forward(X_adv))
        acc_adv = np.mean(y_pred_adv == y_true_clean)

    # visualize pairs
    idx = np.random.choice(X.shape[0], n, replace=False)
    plt.figure(figsize=(2.2*n, 4.4))
    for i, j in enumerate(idx):
        plt.subplot(2, n, i + 1)
        plt.imshow(X[j].reshape(28, 28), cmap='gray')
        plt.title('Clean')
        plt.axis('off')
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(X_adv[j].reshape(28, 28), cmap='gray')
        plt.title('FGSM')
        plt.axis('off')
    plt.suptitle(f"{mode_title} FGSM (eps={eps})")
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
    plt.show()

    return acc_clean, acc_adv, X_adv

def plot_df_grid(model, X, Y, eps, n=5, save_path=None):
    from Functions import deepfool_multiclass
    # compute clean accuracy
    logits_clean = model.forward(X)
    acc_clean = np.mean(np.argmax(logits_clean, 1) == np.argmax(Y, 1))

    # generate adversarial examples
    X_adv = deepfool_multiclass(model, X, Y, epsilon=eps, max_iter=50)

    # compute adv accuracy
    logits_adv = model.forward(X_adv)
    acc_adv = np.mean(np.argmax(logits_adv, 1) == np.argmax(Y, 1))

    # visualize pairs
    idx = np.random.choice(X.shape[0], n, replace=False)
    plt.figure(figsize=(2.2*n, 6.6))
    for i, j in enumerate(idx):
        plt.subplot(3, n, i + 1)
        plt.imshow(X[j].reshape(28, 28), cmap='gray')
        plt.title('Clean')
        plt.axis('off')
        plt.subplot(3, n, i + 1 + n)
        plt.imshow(X_adv[j].reshape(28, 28), cmap='gray')
        plt.title('Deepfool')
        plt.axis('off')
        plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow((X_adv[j]-X[j]).reshape(28, 28)*10, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.title('Perturbationx10')
        plt.axis('off')
    plt.suptitle(f"Deepfool (eps={eps})")
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
    plt.show()

    return acc_clean, acc_adv

    
