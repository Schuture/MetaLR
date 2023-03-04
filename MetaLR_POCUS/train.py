import os
import sys
import time
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim

from meta import MetaSGD
from my_dataset import COVIDDataset


def set_seed(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def get_optimizer(model, now_lr):
    '''
    Get initial optimizer for the model with layer-wise LRs
    Params:
        model: initial model
        now_lr: initial lr
    Return:
        optimizer: the initial optimizer
    '''
    ignored_params = list(map(id, model.layer2.parameters())) + list(map(id, model.layer3.parameters())) + \
                     list(map(id, model.layer4.parameters())) + list(map(id, model.fc.parameters())) \
                     + list(map(id, model.fc.parameters())) + list(map(id, model.layer1.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())  # constant lr, bn layers, conv1

    optimizer = optim.Adam([
        {'params': base_params, 'lr': now_lr[0]}, # conv1 + bn1
        {'params': model.layer1[0].parameters(), 'lr': now_lr[1]},
        {'params': model.layer1[1].parameters(), 'lr': now_lr[2]},
        {'params': model.layer2[0].parameters(), 'lr': now_lr[3]},
        {'params': model.layer2[1].parameters(), 'lr': now_lr[4]},
        {'params': model.layer3[0].parameters(), 'lr': now_lr[5]},
        {'params': model.layer3[1].parameters(), 'lr': now_lr[6]},
        {'params': model.layer4[0].parameters(), 'lr': now_lr[7]},
        {'params': model.layer4[1].parameters(), 'lr': now_lr[8]},
        {'params': model.fc.parameters(), 'lr': now_lr[9]}
    ], weight_decay=args.weight_decay)

    return optimizer


def vis_lrs(layer_lrs, fold):
    layer_names = ['Conv 1', 'Block 1-1', 'Block 1-2', 'Block 2-1', 'Block 2-2',
                   'Block 3-1', 'Block 3-2', 'Block 4-1', 'Block 4-2', 'FC']
    layer_lrs = list(zip(*layer_lrs))
    
    plt.figure(figsize=(7,4), dpi=200)
    for i in range(len(layer_names)):
        plt.plot(layer_lrs[i], label=layer_names[i])
    num1 = 1.02
    num2 = 0
    num3 = 3
    num4 = 0
    plt.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
    plt.title('LRs on POCUS Pneumonia Detection Task (fold {})'.format(fold))
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    
    plt.tight_layout()
    
    plt.savefig('lr_curve_pocus_fold{}.svg'.format(fold), format='svg', dpi=200)
    plt.show()
    

def train_meta_step(inputs, labels, model, meta_model, now_lr, 
               meta_loader, optimizer, criterion, epoch):
    '''
    Update the main model for one step.
    
    If this process is done without meta learning, we forward pass the data
    and get the gradient to update the model directly. If this process is
    done with meta learning, we firstly use meta model to calculate its θ_{t+1}
    and use updated meta model to get the loss and gradient for learning
    rate. And then we use the updated wnet to calculate sample weights and
    update the main model with weighted loss.

    now_lr: a list, all floats
    '''
    # Update meta model and learning rate
    meta_model.load_state_dict(model.state_dict())  # load the same parameters as model

    # 1. Forward pass for meta model.
    outputs = meta_model(inputs)
    loss_hat = criterion(outputs, labels)
    
    # 2. Update the meta model with the now_lr to get θ_{t+1}.
    meta_model.zero_grad()
    grads = torch.autograd.grad(loss_hat, (meta_model.parameters()),
                                create_graph=True, allow_unused=True)
    
    # create a learning rate list that can get the grads
    lrs = torch.tensor([now_lr[i] for i in range(len(now_lr))], dtype=torch.float64, requires_grad=True)
    lrs.retain_grad()
    
    pseudo_optimizer = MetaSGD(meta_model, meta_model.parameters(), lr=0.001)
    pseudo_optimizer.meta_step(grads, lrs)

    del grads

    # 3. Compute upper level objective.
    try:
        meta_inputs, meta_labels = next(meta_iterator)
    except:
        meta_iterator = iter(meta_loader)
        meta_inputs, meta_labels = next(meta_iterator)
    meta_inputs = meta_inputs.to(device)
    meta_labels = meta_labels.to(device)
    meta_outputs = meta_model(meta_inputs)
    meta_outputs.retain_grad()
    loss_meta = criterion(meta_outputs, meta_labels)

    # 4. Use meta loss to update lrs
    # with second order gradients.
    lr_grads = torch.autograd.grad(loss_meta, lrs, create_graph=True, allow_unused=True)
    new_lr = lrs * (1 - lr_grads[0] * args.hyper_lr)
    upper_bound = torch.zeros_like(lrs) + 1e-2
    lower_bound = torch.zeros_like(lrs) + 1e-6
    new_lr = torch.max(new_lr, lower_bound)
    new_lr = torch.min(new_lr, upper_bound)
    
    new_lr = new_lr.detach().tolist()
    if epoch <= 5:
        new_lr[-1] = args.lr
    
    s = optimizer.state_dict()
    for i, group in enumerate(s['param_groups']):
        group['lr'] = new_lr[i]
    optimizer.load_state_dict(s)

    # 5. Get the outputs with updated lr
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # 6. Backward to get the gradient for main model.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss, outputs, new_lr


def main(fold):
    # ============================ step 1/5 Data ============================

    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomResizedCrop(size=64, scale=(0.8, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.18,0.18,0.18], std=[0.24,0.24,0.24])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.18,0.18,0.18], std=[0.24,0.24,0.24])
    ])

    train_folds = [0,1,2,3,4]
    train_folds.remove(fold%5)
    train_data = COVIDDataset(data_dir=args.data_path, fold=train_folds, transform=train_transform)
    meta_data = COVIDDataset(data_dir=args.data_path, fold=train_folds, transform=valid_transform)
    test_data = COVIDDataset(data_dir=args.data_path, fold=[fold%5], transform=valid_transform)

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    meta_loader = DataLoader(dataset=meta_data, batch_size=args.batch_size, num_workers=args.workers)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=args.workers)

    # ============================ step 2/5 LModel ============================
    model = models.resnet18()
    
    state_dict = torch.load(args.pretrained_path)
    state_dict = {k:state_dict[k] for k in list(state_dict.keys()) if not (k.startswith('l') or k.startswith('fc'))} # 去掉2层MLP的参数
    state_dict = {k:state_dict[k] for k in list(state_dict.keys()) if not k.startswith('classifier')} # 去掉classifier的参数
    
    con_layer_names = list(state_dict.keys())
    target_layer_names = list(model.state_dict().keys())
    new_dict = {target_layer_names[i]:state_dict[con_layer_names[i]] for i in range(len(con_layer_names))}

    model_dict = model.state_dict()
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    print('\nThe self-supervised trained parameters are loaded.\n')

    # replace the final FC layer, requires_grad=True
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    model = model.to(device)

    meta_model = models.resnet18()
    num_ftrs = meta_model.fc.in_features
    meta_model.fc = nn.Linear(num_ftrs, 3)
    meta_model = meta_model.to(device)

    # ============================ step 3/5 Loss ============================
    criterion = nn.CrossEntropyLoss()

    # ============================ step 4/5 优化器 ============================
    now_lr = [0.1*args.lr] * 9 + [args.lr]
    optimizer = get_optimizer(model, now_lr)

    # ============================ step 5/5 训练 ============================
    print('\nTraining start!\n')
    start = time.time()
    max_acc_test = 0.
    reached_test = 0
    layer_lrs = []
    layer_lrs.append(now_lr)

    # the statistics of classification result: classification_results[true][pred]
    classification_results = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    best_classification_results = None

    if apex_support and fp16_precision:
        model, optimizer = amp.initialize(model, optimizer,
                                        opt_level='O2',
                                        keep_batchnorm_fp32=True)
    for epoch in range(1, args.max_epoch + 1):

        loss_mean = 0.
        correct = 0.
        total = 0.

        model.train()
        for i, data in enumerate(train_loader):

            # forward
            iter_start = time.time()
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.train()
            
            loss, outputs, now_lr = train_meta_step(inputs, labels, model, meta_model, now_lr, 
                                               meta_loader, optimizer, criterion, epoch)
            layer_lrs.append(now_lr)
            
            # Meta model must be rebuild to make a new computational graph.
            meta_model = models.resnet18()
            num_ftrs = meta_model.fc.in_features
            meta_model.fc = nn.Linear(num_ftrs, 3)
            meta_model = meta_model.to(device)

            # Classification results
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).cpu().squeeze().sum().numpy()

            # Print training information
            loss_mean += loss.item()
            if (i+1) % args.log_interval == 0:
                loss_mean = loss_mean / args.log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} Time:{:.4f}s".format(
                    epoch, args.max_epoch, i+1, len(train_loader), loss_mean, correct / total, time.time()-iter_start))
                loss_mean = 0.

        # Validate the model
        if epoch % args.val_interval == 0:

            correct_test = 0.
            total_test = 0.
            model.eval()
            with torch.no_grad():
                for j, data in enumerate(test_loader):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).cpu().squeeze().sum().numpy()
                    for k in range(len(predicted)):
                        classification_results[labels[k]][predicted[k]] += 1 # "label" is regarded as "predicted"
                acc_test = correct_test / total_test
                print("Test:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Acc:{:.2%}\n".format(
                    epoch, args.max_epoch, j+1, len(test_loader), acc_test))
                
                # record the validation result
                if acc_test > max_acc_test: # use the best validated model for testing
                    max_acc_test = acc_test
                    reached_test = epoch
                    best_classification_results = classification_results
                classification_results = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        
        if epoch % 1 == 0:
            if isinstance(now_lr, torch.Tensor):
                now_lr = [round(lr.item(), 6) for lr in now_lr]
            else:
                now_lr = [round(lr, 6) for lr in now_lr]
            print('The learning rate for epoch {} is {}'.format(epoch, now_lr))

    print('\nTraining finish, the time consumption of {} epochs is {}s\n'.format(args.max_epoch, round(time.time() - start)))
    print('The max testing accuracy is: {:.2%}, reached at epoch {}.\n'.format(max_acc_test, reached_test))
    
    vis_lrs(layer_lrs, fold)
    
    return best_classification_results, layer_lrs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='linear evaluation')
    parser.add_argument('-p', '--pretrained_path', default='pretrained/best_model.pth', help='path of ckpt')
    parser.add_argument('-d', '--data_path', default='data/pocus_data_all.pkl', help='path of dataset')
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-w', '--workers', type=int, default=0)
    parser.add_argument('-e', '--max-epoch', type=int, default=30)
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-l', '--lr', type=float, default=0.01)
    parser.add_argument('--hyper-lr', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--log-interval', type=int, default=2)
    parser.add_argument('--val-interval', type=int, default=1)
    args = parser.parse_args()

    try:
        sys.path.append('./apex')
        from apex import amp
        print("Apex on, run on mixed precision.")
        apex_support = True
    except:
        print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
        apex_support = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\nRunning on:", device)

    if device == 'cuda':
        device_name = torch.cuda.get_device_name()
        print("The device name is:", device_name)
        cap = torch.cuda.get_device_capability(device=None)
        print("The capability of this device is:", cap, '\n')
    
    print('State dict path:', args.pretrained_path)
    fp16_precision = False
    
    print('\n=================Evaluation on POCUS dataset start===================')
    results = {}
    set_seed(args.seed)
    print('random seed:', args.seed)
    confusion_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for i in [1,2,3,4,5]:
        print('\n' + '='*20 + 'The training of fold {} start.'.format(i) + '='*20)
        best_classification_results, layer_lrs = main(fold=i-1)
        confusion_matrix = confusion_matrix + np.array(best_classification_results)

    print('\nThe confusion matrix is:')
    print(confusion_matrix)
    print('\nThe precision of class 0 is:', confusion_matrix[0,0] / sum(confusion_matrix[:,0]))
    print('The precision of class 1 is:', confusion_matrix[1,1] / sum(confusion_matrix[:,1]))
    print('The precision of class 2 is:', confusion_matrix[2,2] / sum(confusion_matrix[:,2]))
    print('\nThe recall of class 0 is:', confusion_matrix[0,0] / sum(confusion_matrix[0]))
    print('The recall of class 1 is:', confusion_matrix[1,1] / sum(confusion_matrix[1]))
    print('The recall of class 2 is:', confusion_matrix[2,2] / sum(confusion_matrix[2]))
    
    acc = (confusion_matrix[0,0]+confusion_matrix[1,1]+confusion_matrix[2,2])/confusion_matrix.sum()
    acc = round(acc, 4) * 100
    print('\nTotal acc is:', acc)






