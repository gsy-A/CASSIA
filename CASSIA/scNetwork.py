# BCE
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
import os
import numpy as np
import math
from sklearn import metrics
from Generate import *
import torch.nn.functional as F


def buildNetwork(layers, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)


class scNetwork(nn.Module):
    
    def __init__(self, input_dim, z_dim, encodeLayer=[256, 128], decodeLayer=[128, 256], 
                 activation="relu", sigma=0.5, alpha=1., 
                 gamma1=1., gamma2=1., gamma_adv=0.1,
                 device="cuda"):
        super(scNetwork, self).__init__()
        
        self.z_dim = z_dim
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.device = device
        
        self.encoder = buildNetwork([input_dim] + encodeLayer, activation=activation) 
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)  
     
        self.decoder = buildNetwork([z_dim] + decodeLayer, activation=activation)
        self._dec_prob = nn.Sequential(
            nn.Linear(decodeLayer[-1], input_dim),  
            nn.Sigmoid() 
        )
        
        self.bce_loss = nn.BCELoss()  
        
        self.to(device)

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        
    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q
    
    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def forwardAE(self, x):

        h = self.encoder(x + torch.randn_like(x) * self.sigma)
        z = self._enc_mu(h)
        
        h_dec = self.decoder(z)
        prob = self._dec_prob(h_dec)
        
        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        
        return z0, prob  

    def forward(self, x):

        h = self.encoder(x + torch.randn_like(x) * self.sigma)
        z_noise = self._enc_mu(h)
        
        h_dec = self.decoder(z_noise)
        prob = self._dec_prob(h_dec)
        
        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        q = self.soft_assign(z0)
        
        return z0, q, prob  

    def encodeBatch(self, X, batch_size=256):
        self.eval()
        encoded = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs = Variable(xbatch).to(self.device)
            z, _ = self.forwardAE(inputs)  
            encoded.append(z.data)
        encoded = torch.cat(encoded, dim=0)
        return encoded.to(self.device)
    
    def cluster_loss(self, p, q):

        p = torch.clamp(p, min=1e-8)
        q = torch.clamp(q, min=1e-8)

        m = (p + q) / 2
        m = torch.clamp(m, min=1e-8)

        log_p = torch.log(p)
        log_q = torch.log(q)
        log_m = torch.log(m)

        kl_pm = torch.sum(p * (log_p - log_m), dim=-1)
        kl_qm = torch.sum(q * (log_q - log_m), dim=-1)
        js = (kl_pm + kl_qm) / 2
    
        return torch.mean(js)

    def triplet_loss(self, anchor, positive, negative, margin_constant):

        pos_euclidean = torch.norm(anchor - positive, p=2, dim=1)
        neg_euclidean = torch.norm(anchor - negative, p=2, dim=1)
    
        pos_cosine = torch.nn.functional.cosine_similarity(anchor, positive, dim=1)
        neg_cosine = torch.nn.functional.cosine_similarity(anchor, negative, dim=1)
    
        confidence = torch.abs(pos_cosine - neg_cosine)
    
        temperature = 5.0  
        euclidean_weight = 1.0 - torch.sigmoid(confidence * temperature)
        cosine_weight = torch.sigmoid(confidence * temperature)

        euclidean_loss = torch.clamp(
            pos_euclidean - neg_euclidean + margin_constant, min=0
        ) / margin_constant
    
        cosine_loss = torch.clamp(
            neg_cosine - pos_cosine + margin_constant, min=0  
        ) / margin_constant

        weighted_loss = (
            euclidean_weight * euclidean_loss +
            cosine_weight * cosine_loss
        )

        return torch.mean(weighted_loss)

    def pretrain_autoencoder(self, X, batch_size=256, lr=0.001, epochs=100,
                             ae_save=True, ae_weights='AE_weights.pth.tar'):

        self.train()
        dataset = TensorDataset(torch.Tensor(X))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print("Pretraining stage...")
        
        optimizer = optim.Adam(self.parameters(), lr=lr, amsgrad=True)
        
        for epoch in range(epochs):
            bce_val = 0.
            
            for batch_idx, (x_batch,) in enumerate(dataloader):
                x_tensor = x_batch.to(self.device)

                z, prob = self.forwardAE(x_tensor)

                loss = self.bce_loss(prob, x_tensor)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                bce_val += loss.item() * len(x_batch)

            print('Pretrain epoch %3d  BCE=%.6f' % (epoch+1, bce_val/X.shape[0]))

        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, ae_weights)

    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)

    def fit(self, X, n_clusters, generate, margin, ratio,
            init_centroid=None, y=None, y_pred_init=None, 
            lr=1., batch_size=256, num_epochs=10, 
            update_interval=1, tol=1e-3, save_dir=""):
        
        self.train()
        
        X = torch.tensor(X, dtype=torch.float)
        
        self.mu = Parameter(torch.Tensor(n_clusters, self.z_dim).to(self.device))

        optimizer = optim.Adadelta(self.parameters(), lr=lr, rho=.9)

        print("Initializing cluster centers with K-Means...")
        if init_centroid is None:
            kmeans = KMeans(n_clusters, init='k-means++', n_init=10)
            data = self.encodeBatch(X)
            self.y_pred = kmeans.fit_predict(data.data.cpu().numpy())
            self.y_pred_last = self.y_pred
            self.mu.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float))
        else:
            self.mu.data.copy_(torch.tensor(init_centroid, dtype=torch.float))
            self.y_pred = y_pred_init
            self.y_pred_last = self.y_pred

        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))

        final_acc, final_nmi, final_ari, final_ami, final_homo = 0, 0, 0, 0, 0
        loss_history = []
        
        for epoch in range(num_epochs):
            if epoch > 0:
                loss_history.append(total_loss / num)
                
            if epoch % update_interval == 0:
                latent = self.encodeBatch(X.to(self.device))
                q = self.soft_assign(latent)
                p = self.target_distribution(q).data
                self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()

                final_acc = np.round(cluster_acc(y, self.y_pred), 5)
                final_nmi  = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                final_ari  = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                final_ami = np.round(metrics.adjusted_mutual_info_score(y, self.y_pred), 5)
                final_homo  = np.round(metrics.homogeneity_score(y, self.y_pred), 5)

                print('\t\tClustering %d: NMI=%.4f, ARI=%.4f, ACC=%.4f, AMI=%.4f, Homo=%.4f' 
                      % (epoch, final_nmi, final_ari, final_acc, final_ami, final_homo))

                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(float) / num
                if (epoch > 0 and delta_label < tol) or epoch % 25 == 0:
                    self.save_checkpoint({
                        'epoch': epoch+1,
                        'state_dict': self.state_dict(),
                        'mu': self.mu,
                        'y_pred': self.y_pred,
                        'y_pred_last': self.y_pred_last,
                        'y': y}, epoch+1, filename=save_dir)

                self.y_pred_last = self.y_pred
                if epoch > 0:
                    if delta_label < tol:
                        print('delta_label ', delta_label, '< tol ', tol)
                        print("Reach tolerance threshold. Stopping training.")
                        break
                    elif epoch > 5:
                        if np.mean(abs(np.diff(loss_history[-6:]))) < tol * 5:
                            print("Reach tolerance threshold. Stopping running.")
                            break
 
            bce_loss_val = cluster_loss_val = 0.0
            
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                pbatch = p[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                
                inputs = Variable(xbatch).to(self.device)
                target = Variable(pbatch).to(self.device)
                
                optimizer.zero_grad()
                z0, qbatch, prob = self.forward(inputs)
                
                cluster_loss = self.cluster_loss(target, qbatch)
                bce_loss = self.bce_loss(prob, inputs) 
                
                loss1 = cluster_loss * self.gamma1 + bce_loss
                loss1.backward()
                optimizer.step()
                
                cluster_loss_val += cluster_loss.item() * len(inputs)
                bce_loss_val += bce_loss.item() * len(inputs)

            X_sub, y_sub, p_sub = cell_select(X, y, p, ratio)
            anchor, positive, negative = generate_triplets(y_sub, generate=generate)
            tri_num = anchor.shape[0]
            tri_num_batch = int(math.ceil(1.0 * anchor.shape[0] / batch_size))

            triplet_loss_val = 0.0
            for tri_batch_idx in range(tri_num_batch):
                xt1 = X_sub[anchor[tri_batch_idx * batch_size: min(tri_num, (tri_batch_idx + 1) * batch_size)]]
                xt2 = X_sub[positive[tri_batch_idx * batch_size: min(tri_num, (tri_batch_idx + 1) * batch_size)]]
                xt3 = X_sub[negative[tri_batch_idx * batch_size: min(tri_num, (tri_batch_idx + 1) * batch_size)]]
                
                optimizer.zero_grad()
                inputs1 = Variable(xt1).to(self.device)
                inputs2 = Variable(xt2).to(self.device)
                inputs3 = Variable(xt3).to(self.device)
                
                _, q_t1, _ = self.forward(inputs1) 
                _, q_t2, _ = self.forward(inputs2)
                _, q_t3, _ = self.forward(inputs3)
                
                triplet_loss = self.triplet_loss(q_t1, q_t2, q_t3, margin)
                loss2 = triplet_loss * self.gamma2
                loss2.requires_grad_(True)
                loss2.backward()
                optimizer.step()
                triplet_loss_val += triplet_loss.item() * len(xt1)

            print("Epoch%3d:" % (epoch + 1))
            print("\t\tBCE Loss: %.5f, Clustering Loss: %.5f" % (bce_loss_val / num, cluster_loss_val / num))
            if tri_num_batch > 0:
                print("\t\tTriplet Loss: %.5f" % (triplet_loss_val / num))

            total_loss = bce_loss_val + cluster_loss_val * self.gamma1 + triplet_loss_val * self.gamma2
            print("\t\t***Total Loss: %.5f" % (total_loss / num))

        with torch.no_grad():
            self.eval()
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            h = self.encoder(X_tensor)  
            embedding = self._enc_mu(h).cpu().numpy()  

        return self.y_pred, final_nmi, final_ari, final_acc, final_ami, final_homo, embedding