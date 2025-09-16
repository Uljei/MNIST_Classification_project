import numpy as np
from model import MLPModel, load_mnist
from Functions import fgsm_attack, bit4_logits_to_int, bit4_bits_to_int, mse, cross_entropy, bce_logits
def hyperparam_study(
    label_mode,
    configs,
    seed=42,
    early_stopping=True,
    patience=5,
    eval_metric='val_acc',      # You can also use 'val_loss' as a selection criterion.
    return_best_model=False,    # if True return { 'best_model': model, ... }
    fgsm_eval=False,            # Do an FGSM robustness evaluation on the validation set when True
    fgsm_eps_list=(0.2,),       # more eps supported 
    n_eval_fgsm=1000            
):
    """
    Grid search / batch experimenter:
      - Support hidden as int or list (multiple hidden layers)
      - support activations as list, length = number of hidden layers + 1 (last layer activated)
        * If activations are not provided, use default as per label_mode:
          onehot: ['sigmoid', 'softmax']
          bit4: ['sigmoid', None] (BCE-with-logits)
      - Early stop: based on eval_metric (val_acc or val_loss), stop when PATIENCE is reached
      - Optional: do FGSM robustness evaluation on validation set (log clean vs adv accuracy)
    return: list of results prioritized by eval_metric (each element is a dict)
          If return_best_model=True, with 'best_model' in addition
    """
    rng = np.random.RandomState(seed)
    all_results = []
    best_pack = None

    # 统一载数据（每个 cfg 可单独设 seed 则重复载入也 OK）
    def _load(label_mode):
        return load_mnist('mnist.pkl.gz', label_mode=label_mode)

    for cfg in configs:
        # 解析配置
        _seed      = cfg.get('seed', seed)
        rng.seed(_seed)
        np.random.seed(_seed)

        epochs     = int(cfg.get('epochs', 15))
        lr         = float(cfg.get('lr', 0.3))
        batch_size = int(cfg.get('batch_size', 32))

        hidden     = cfg.get('hidden', 30)
        if isinstance(hidden, int):
            hidden_sizes = [hidden]
        else:
            hidden_sizes = list(hidden)

        # 设定激活（允许覆盖）
        acts = cfg.get('activations', None)
        if acts is None:
            if label_mode == 'onehot':
                # 隐藏层全部用 sigmoid，输出 softmax
                acts = ['sigmoid'] * len(hidden_sizes) + ['softmax']
                loss = 'cross_entropy'
                out_size = 10
            else:
                # 隐藏层 sigmoid，输出 logits（None）+ BCE-with-logits
                acts = ['sigmoid'] * len(hidden_sizes) + [None]
                loss = 'bce_logits'
                out_size = 4
        else:
            # 用户自定义激活，推断输出大小 & loss
            acts = list(acts)
            if label_mode == 'onehot':
                out_size = 10
                loss = 'cross_entropy' if acts[-1] == 'softmax' else cfg.get('loss', 'cross_entropy')
            else:
                out_size = 4
                loss = 'bce_logits' if acts[-1] is None else cfg.get('loss', 'bce_logits')

            if len(acts) != len(hidden_sizes) + 1:
                raise ValueError("activations 长度需等于 隐藏层数 + 1（包含输出层激活）。")

        # 构图
        layers = [784] + hidden_sizes + [out_size]
        num_layers = len(layers) - 2  # 隐藏层数

        # 加载数据
        X_train, y_train, X_val, y_val, X_test, y_test = _load(label_mode)

        # 初始化模型
        model = MLPModel(
            input_size=layers[0],
            output_size=layers[-1],
            hidden_sizes=layers[1:-1],
            num_layers=num_layers,
            activations=acts,
            loss=loss
        )

        # ----- 训练循环（带早停） -----
        best_val_acc  = -np.inf
        best_val_loss = np.inf
        best_epoch    = -1
        no_imp        = 0

        train_losses, val_losses = [], []

        for epoch in range(epochs):
            # 打乱
            idx = rng.permutation(X_train.shape[0])
            X_shuf, y_shuf = X_train[idx], y_train[idx]

            # 小批量 SGD
            for i in range(0, X_shuf.shape[0], batch_size):
                Xb = X_shuf[i:i+batch_size]
                yb = y_shuf[i:i+batch_size]
                y_hat = model.forward(Xb)
                model.backward(yb, y_hat)
                model.update(lr)

            # 计算当轮的 train/val loss
            y_hat_tr  = model.forward(X_train)
            y_hat_val = model.forward(X_val)

            if loss == "mse":
                trL = mse(y_train, y_hat_tr);  vaL = mse(y_val, y_hat_val)
            elif loss == "cross_entropy":
                trL = cross_entropy(y_train, y_hat_tr);  vaL = cross_entropy(y_val, y_hat_val)
            elif loss == "bce_logits":
                trL = bce_logits(y_train, y_hat_tr);     vaL = bce_logits(y_val, y_hat_val)
            else:
                raise ValueError("Unsupported loss.")

            train_losses.append(trL);  val_losses.append(vaL)

            # 计算当轮的 val_acc
            val_acc = model.evaluate(X_val, y_val)

            # 依据指标判定是否更新“最优”
            improved = False
            if eval_metric == 'val_acc':
                if val_acc > best_val_acc:
                    best_val_acc, best_val_loss = val_acc, vaL
                    best_epoch, improved = epoch, True
            elif eval_metric == 'val_loss':
                if vaL < best_val_loss:
                    best_val_loss, best_val_acc = vaL, val_acc
                    best_epoch, improved = epoch, True
            else:
                raise ValueError("eval_metric 仅支持 'val_acc' 或 'val_loss'")

            # 早停计数
            if improved:
                no_imp = 0
                # 可选：如需“真正回滚”参数，可在此深拷贝 model（当前实现记录指标即可）
                best_snapshot = {
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'val_loss': vaL,
                }
            else:
                no_imp += 1
                if early_stopping and no_imp >= patience:
                    # print(f"[EarlyStop] epoch={epoch+1} no_improve={no_imp}")
                    break

        # 选一个统一的“测试集表现”做记录（注意：未回滚参数，则是最后一轮的）
        test_acc = model.evaluate(X_test, y_test)

        # 可选：验证集 FGSM 稳健性评估
        fgsm_records = []
        if fgsm_eval:
            n_eval = min(n_eval_fgsm, X_val.shape[0])
            Xe = X_val[:n_eval]; Ye = y_val[:n_eval]
            if label_mode == 'onehot':
                clean_acc = np.mean(np.argmax(model.forward(Xe), 1) == np.argmax(Ye, 1))
                for eps in fgsm_eps_list:
                    Xadv = fgsm_attack(model, Xe, Ye, epsilon=eps)
                    adv_acc = np.mean(np.argmax(model.forward(Xadv), 1) == np.argmax(Ye, 1))
                    fgsm_records.append({'eps': float(eps), 'clean_acc': float(clean_acc), 'adv_acc': float(adv_acc)})
            else:
                y_pred_clean = bit4_logits_to_int(model.forward(Xe))
                y_true_clean = bit4_bits_to_int(Ye)
                clean_acc = float(np.mean(y_pred_clean == y_true_clean))
                for eps in fgsm_eps_list:
                    Xadv = fgsm_attack(model, Xe, Ye, epsilon=eps)
                    y_pred_adv = bit4_logits_to_int(model.forward(Xadv))
                    adv_acc = float(np.mean(y_pred_adv == y_true_clean))
                    fgsm_records.append({'eps': float(eps), 'clean_acc': clean_acc, 'adv_acc': adv_acc})

        result = {
            'label_mode': label_mode,
            'hidden': hidden if isinstance(hidden, int) else tuple(hidden),
            'activations': tuple(acts),
            'loss': loss,
            'lr': float(lr),
            'batch_size': int(batch_size),
            'epochs_run': best_epoch + 1 if early_stopping else epochs,
            'final_val_loss': float(val_losses[-1]),
            'final_val_acc': float(model.evaluate(X_val, y_val)),
            'best_val_loss': float(best_val_loss),
            'best_val_acc': float(best_val_acc),
            'test_acc': float(test_acc),
            'seed': int(_seed),
        }
        if fgsm_eval:
            result['fgsm_val'] = fgsm_records

        pack = {'result': result, 'model': model}
        all_results.append(result)

        # 维护全局最优
        better = False
        if best_pack is None:
            better = True
        else:
            if eval_metric == 'val_acc':
                better = (result['best_val_acc'] > best_pack['result']['best_val_acc'])
            else:
                better = (result['best_val_loss'] < best_pack['result']['best_val_loss'])
        if better:
            best_pack = pack

    # 排序输出
    if eval_metric == 'val_acc':
        all_results.sort(key=lambda d: d['best_val_acc'], reverse=True)
    else:
        all_results.sort(key=lambda d: d['best_val_loss'])

    if return_best_model:
        return {
            'results': all_results,
            'best_model': best_pack['model'],
            'best_result': best_pack['result']
        }
    return all_results

if __name__ == "__main__":
    
    cfgs = [
        # 10 类 one-hot：softmax + CE
        {'hidden': 30, 'epochs': 30, 'lr': 0.3, 'batch_size': 32},
        {'hidden': [128, 64], 'epochs': 30, 'lr': 0.2, 'batch_size': 64, 'activations': ['ReLU', 'ReLU', 'softmax']},
        {'hidden': [64, 64, 32], 'epochs': 40, 'lr': 0.15, 'batch_size': 64,
         'activations': ['tanh', 'tanh', 'tanh', 'softmax']},

        # 4-bit：logits(None) + BCE-with-logits
        {'hidden': 32, 'epochs': 30, 'lr': 0.3, 'batch_size': 32},
        {'hidden': [64, 32], 'epochs': 35, 'lr': 0.25, 'batch_size': 64, 'activations': ['sigmoid', 'sigmoid', None]},
        {'hidden': [128, 64, 32], 'epochs': 40, 'lr': 0.2, 'batch_size': 64,
         'activations': ['ReLU', 'ReLU', 'ReLU', None]},
    ]

    res10 = hyperparam_study('onehot', cfgs[:3], early_stopping=True, patience=6,
                             eval_metric='val_acc', fgsm_eval=True, fgsm_eps_list=(0.1, 0.2), return_best_model=True)
    resb4 = hyperparam_study('bit4', cfgs[3:], early_stopping=True, patience=6,
                             eval_metric='val_acc', fgsm_eval=True, fgsm_eps_list=(0.1, 0.2), return_best_model=True)

    print("Top (onehot):", res10['best_result'])
    print("Top (bit4):  ", resb4['best_result'])