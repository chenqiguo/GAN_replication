# this is chenqi's modification for custom datasets!
# version 2: based on v1, do the following updates:
# (1) in the func test(), instead of kNN on class, do kNN on img index! <-- then each image represents a class during implementation.
# (2) in data-aug for training, replace color jitter with Gaussian blur (+ Gaussian noise?) .
# (3) During training simCLR, add fake (gan generated) images to the original dataset to train!

import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm

#import utils_chenqi # import utils
import utils_chenqi_v2
from model import Model

# newly added:
from PIL import Image
from torchvision import transforms, datasets


# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    
    for pos_1, pos_2, target in train_bar: # target.shape: torch.Size([batch_size])
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True) # pos_1.shape: torch.Size([batch_size, img_ch, img_h, img_w])
        # note: feature: h (the embedding we want to do NN query), of shape: torch.Size([batch_size, 2048])
        #       out: z (the projection used to maximize agreement) of shape: torch.Size([batch_size, feature_dim]).
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0) # shape: torch.Size([2*batch_size, feature_dim])
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            
            # to check error: for debug:
            #torch.max(sim_labels.view(-1, 1)) # cls_num-1
            #torch.min(sim_labels.view(-1, 1)) # 0
                
            # error here!!!
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True) # torch.Size([26, 102])
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100



def get_mean_std_forDataset(data_dir,img_size,batch_size,isGray):
    # newly added: compute the mean and std for transforms.Normalize using whole dataset:
    tmp_data = datasets.ImageFolder(root=data_dir, transform=transforms.Compose([transforms.Resize(img_size),
                                                                                 transforms.CenterCrop(img_size),
                                                                                 transforms.ToTensor()]))
    tmp_loader = DataLoader(tmp_data, batch_size=batch_size, shuffle=False, num_workers=16)
    
    mean = 0.
    std = 0.
    nb_samples = 0.
    if not isGray:
        for data, _ in tmp_loader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples
    #else: for MNIST
    
    mean /= nb_samples
    std /= nb_samples
    
    return (mean, std)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=26, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=2000, type=int, help='Number of sweeps over the dataset to train')
    # newly added:
    parser.add_argument('--dataset', default='FLOWER_128', type=str, help='Name of the training dataset, eg, FLOWER_128')
    parser.add_argument('--data_dir', default='/eecf/cbcsl/data100b/Chenqi/new_metrics/SimCLR/data/FLOWER_gan/', type=str, help='Dir of the original & GAN generated fake training dataset')
    #parser.add_argument('--label_file', default='/eecf/cbcsl/data100b/Chenqi/data/flower_labels.txt', type=str, help='Path to the txt file with class labels')
    # maybe also add arg like: choices of data-aug...
    
    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    # newly added:
    dataset, data_dir = args.dataset, args.data_dir
    img_size = int(dataset.split('_')[-1])
    #label_file = args.label_file
    
    # newly added: note: we should compute transforms.Normalize for our custom dataset each time! <-- will later modify it
    # also note: for MNIST (gray-scale imgs), needs to modify color jitter & random gray & normalize!! <-- will later modify it
    if 'MNIST' not in dataset:
        # newly added: compute the mean and std for transforms.Normalize using whole dataset:
        img_means, img_stds = get_mean_std_forDataset(data_dir,img_size,batch_size,isGray=False)
        if 'FLOWER' in dataset:
            train_transform = transforms.Compose([
                transforms.Resize(img_size),transforms.CenterCrop(img_size), # NOT use random crop! use resize & center crop!!!
                #transforms.RandomHorizontalFlip(p=0.5), # for FLOWER & MNIST: NOT do this!
                transforms.GaussianBlur(51, sigma=(0.1, 1.0)), # NOT jitter that much for FLOWER!! Add Gaussian blurring.
                #transforms.RandomGrayscale(p=0.2), 
                transforms.RandomAffine(degrees=10, translate=None, scale=None, shear=10), # maybe also add affain warping?
                transforms.ToTensor(),
                transforms.Normalize(img_means, img_stds)]) # ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) cifar10
        elif 'CelebA' in dataset:
            train_transform = transforms.Compose([
                transforms.Resize(img_size),transforms.CenterCrop(img_size), # NOT use random crop! use resize & center crop!!!
                transforms.RandomHorizontalFlip(p=0.5), # for FLOWER & MNIST: NOT do this!
                transforms.GaussianBlur(51, sigma=(0.1, 1.0)), # NOT jitter that much for FLOWER!! Add Gaussian blurring.
                transforms.RandomGrayscale(p=0.2), 
                #transforms.RandomAffine(degrees=5, translate=None, scale=None, shear=5), # maybe also add affain warping?
                transforms.ToTensor(),
                transforms.Normalize(img_means, img_stds)]) # ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) cifar10
        
        test_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(img_means, img_stds)]) # ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) for cifar10
    # else:... (for MNIST)
    
    # data prepare
    # newly modified: to adjust to custom dataset!
    """
    # original old code:
    train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=True)
    """
    # new code for custom dataset:
    #train_data = datasets.ImageFolder(root=data_dir, transform=train_transform)
    train_data = utils_chenqi_v2.MyCustomDataset_v2(root=data_dir, transform=train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=True)
    """
    # original old code:
    memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.test_transform, download=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    """
    # new code for custom dataset:
    #memory_data = datasets.ImageFolder(root=data_dir, transform=test_transform)
    memory_data = utils_chenqi_v2.MyCustomDataset_v2(root=data_dir, transform=test_transform)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    """
    # original old code:
    test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    """
    # new code for custom dataset:
    #test_data = datasets.ImageFolder(root=data_dir, transform=test_transform)
    test_data = utils_chenqi_v2.MyCustomDataset_v2(root=data_dir, transform=train_transform) # make the testing set to be the transformed original image!!!
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    
    # model setup and optimizer config
    model = Model(feature_dim).cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    c = len(memory_data.classes)
    """
    # for debug:
    print('***************** DEBUG *****************')
    print('c = ' + str(c))
    print('memory_data.classes = ')
    print(memory_data.classes)
    assert(False)
    """
    
    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}_{}'.format(feature_dim, temperature, k, batch_size, epochs)
    # newly modified:
    if not os.path.exists('results_v2/' + dataset + '/'):
        os.mkdir('results_v2/' + dataset + '/')
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        # newly modified:
        data_frame.to_csv('results_v2/' + dataset + '/' + '{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        
        """
        # original code: only save the "best" model:
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            # newly modified:
            torch.save(model.state_dict(), 'results_v2/' + dataset + '/' + '{}_model.pth'.format(save_name_pre))
        """
        # new code: save all the models!!! (while also keep track on the "best" model):
        torch.save(model.state_dict(), 'results_v2/' + dataset + '/' + 'epoch{}'.format(epoch) + '_{}_model.pth'.format(save_name_pre))
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            # newly modified:
            torch.save(model.state_dict(), 'results_v2/' + dataset + '/' + 'best_{}_model.pth'.format(save_name_pre))
        