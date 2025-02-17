import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM
import torch.nn.functional as F

class PretrainBert(nn.Module):
    def __init__(self, args, data):
        super(PretrainBert, self).__init__()
        self.args = args
        self.num_labels = data.n_coarse
        self.model_name = args.model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name, cache_dir=self.args.cache_dir)
        self.classifier = nn.Linear(768, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.backbone.to(self.device)
        self.classifier.to(self.device)

    def forward(self, X, output_hidden_states=False, output_attentions=False):
        """logits are not normalized by softmax in forward function"""
        outputs = self.backbone(**X, output_hidden_states=True)
        CLSEmbedding = outputs.hidden_states[-1][:,0]
        CLSEmbedding = self.dropout(CLSEmbedding)
        logits = self.classifier(CLSEmbedding)
        output_dir = {"logits": logits}
        if output_hidden_states:
            output_dir["hidden_states"] = outputs.hidden_states[-1][:, 0]
        if output_attentions:
            output_dir["attentions"] = outputs.attention
        return output_dir

    def mlmForward(self, X, Y):
        outputs = self.backbone(**X, labels=Y)
        return outputs.loss

    def loss_ce(self, logits, Y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, Y)
        return output
    
    def save_backbone(self, save_path):
        self.backbone.save_pretrained(save_path)

class MCNBert(nn.Module):
    
    def __init__(self, args, n_fine, n_coarse, feat_dim=128):
        super(MCNBert, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_fine = n_fine
        self.n_coarse = n_coarse
        self.args = args
        self.model_name = args.model_name
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name, cache_dir=self.args.cache_dir)
        hidden_size = self.backbone.config.hidden_size
        self.temperature = args.temperature
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, feat_dim)
        )
        self.classifier = nn.Linear(feat_dim, n_coarse)
        self.backbone.to(self.device)
        self.head.to(self.device)





        self.output_dim = args.output_dim
        self.feat_dim = args.feat_dim

        self.fine_labels_embedding = nn.Embedding(self.n_fine, self.feat_dim).to(self.device)

        self.fc_mu = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.output_dim)
        ).to(self.device)

        self.fc_sigma = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.output_dim)
        ).to(self.device)


        self.base = nn.Parameter(torch.tensor([10.0]))



    # the usage of proj_conversion is to convert the embeddings to Gaussian distributions, mu and sigma. For exmaple, the i-th sample's embedding is x_i, then mu_i, sigma_i = proj_conversion(x_i)
    def proj_conversion(self, x):
        mu = self.fc_mu(x)

        sigma = F.elu(self.fc_sigma(x)) + (1 + 1e-14 )
        return mu, sigma

    def compute_kl_divergence_batch(self, mu_i, sigma_i, mu_j, sigma_j):
        """
        Compute batched KL divergence between two sets of Gaussian distributions.
        Each set is described by mean vectors (mu) and variance vectors (sigma) of diagonal covariance matrices.

        Args:
        - mu_i (Tensor): Mean vectors of the first set of distributions. Shape: [batch_size_i, feature_dim].
        - sigma_i (Tensor): Variances of the first set of distributions. Shape: [batch_size_i, feature_dim].
        - mu_j (Tensor): Mean vectors of the second set of distributions. Shape: [batch_size_j, feature_dim].
        - sigma_j (Tensor): Variances of the second set of distributions. Shape: [batch_size_j, feature_dim].

        Returns:
        - Tensor: The symmetrized KL divergences for all pairs. Shape: [batch_size_i, batch_size_j].
        """
        # Expand mu and sigma for broadcasting: [batch_size_i, 1, feature_dim] x [1, batch_size_j, feature_dim]
        epsilon = 1e-8

        mu_i_exp = mu_i.unsqueeze(1)
        sigma_i_exp = sigma_i.unsqueeze(1) + epsilon
        # the shape of mu_i_exp is (batch_size_i, 1, feat_dim), the shape of sigma_i_exp is (batch_size_i, 1, feat_dim)



        mu_j_exp = mu_j.unsqueeze(0)
        sigma_j_exp = sigma_j.unsqueeze(0) + epsilon
        # the shape of mu_j_exp is (1, batch_size_j, feat_dim), the shape of sigma_j_exp is (1, batch_size_j, feat_dim)


        # Compute element-wise difference of mu, and ratio and log-ratio of sigma
        diff_mu = mu_i_exp - mu_j_exp
        # the shape of diff_mu is (batch_size_i, batch_size_j, feat_dim)


        sigma_ratio = sigma_i_exp / (sigma_j_exp + epsilon)
        # the shape of sigma_ratio is (batch_size_i, batch_size_j, feat_dim)


        log_sigma_ratio = torch.log(sigma_ratio + epsilon)
        # the shape of log_sigma_ratio is (batch_size_i, batch_size_j, feat_dim)


        # Compute quadratic term for the KL divergence formula
        quad_term = (diff_mu ** 2) / sigma_j_exp
        # the shape of quad_term is (batch_size_i, batch_size_j, feat_dim)


        # Compute the KL divergence using the formula for diagonal covariance matrices
        kl_div_ij = 0.5 * (torch.sum(log_sigma_ratio + sigma_ratio + quad_term - 1, dim=2))
        # the shape of kl_div_ij is (batch_size_i, batch_size_j)


        # Compute the reverse KL divergence (KL(j||i))
        kl_div_ji = 0.5 * (torch.sum(torch.log(sigma_j_exp / sigma_i_exp) + sigma_j_exp / sigma_i_exp + (diff_mu ** 2) / sigma_i_exp - 1, dim=2))
        # the shape of sigma_j_exp / sigma_i_exp is (batch_size_i, batch_size_j, feat_dim)
        # the shape of torch.log(sigma_j_exp / sigma_i_exp) is (batch_size_i, batch_size_j, feat_dim)
        # the shape of kl_div_ji is (batch_size_i, batch_size_j)

        # Symmetrize the KL divergence
        sym_kl_div = (kl_div_ij + kl_div_ji) / 2.0
        # the shape of sym_kl_div is (batch_size_i, batch_size_j)

        return sym_kl_div


    def batched_kl_divergence(self, features, k):
        kl_div_matrix = torch.zeros(features.size(0), k.size(0), device=features.device)
        batch_size_f = 32   # Adjust based on your memory constraints
        batch_size_k = 3000  # Adjust based on your memory constraints
        for i in range(0, features.size(0), batch_size_f):
            # print('the feature chunk index is:', i)
            # print('date time is:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            mu_i, sigma_i = self.proj_conversion(features[i:i + batch_size_f])
            # the shape of mu_i is (batch_size_f, feat_dim), the shape of sigma_i is (batch_size_f, feat_dim)

            for j in range(0, k.size(0), batch_size_k):
                mu_j, sigma_j = self.proj_conversion(k[j:j + batch_size_k])
                # the shape of mu_j is (batch_size_k, feat_dim), the shape of sigma_j is (batch_size_k, feat_dim)

                # Compute KL divergence for each pair in the batch
                sym_kl_div = self.compute_kl_divergence_batch(mu_i, sigma_i, mu_j, sigma_j)
                # the shape of sym_kl_div is (batch_size_f, batch_size_k)

                # the shape of kl_div_ij is (batch_size_f, batch_size_k)
                # kl_div_ji = self.compute_kl_divergence_batch(mu_j, sigma_j, mu_i, sigma_i)
                # # the shape of kl_div_ji is (batch_size_k, batch_size_f)

                # sym_kl_div = 0.5 * (kl_div_ij + kl_div_ji)

                # Update the corresponding block in kl_div_matrix
                kl_div_matrix[i:i + batch_size_f, j:j + batch_size_k] = sym_kl_div

        return kl_div_matrix
        # the shape of kl_div_matrix is (features.size(0), k.size(0))





    def forward(self, X, output_logits=False):
        """logits are not normalized by softmax in forward function"""
        outputs = self.backbone(**X, output_hidden_states=True, output_attentions=True)
        cls_embed = outputs.hidden_states[-1][:,0]
        # the shape of cls_embed is [batch_size, hidden_size]
        features = self.head(cls_embed)
        logits = self.classifier(features)

        cls_embed = F.normalize(cls_embed, dim=1)
        
        if output_logits:
            return cls_embed, logits
        return cls_embed

    def save_backbone(self, save_path):
        self.backbone.save_pretrained(save_path)

    def fine_loss(self, features, k, mask, temperature):
        # the shape of features is (batch_size, hidden_size)
        # the shape of k is (training_sample_num, hidden_size)
        # the shape of mask is (batch_size, training_sample_num)

        mask_KL = self.batched_kl_divergence(features, k)
        # the shape of mask_KL is (batch_size, training_sample_num)
        logits = (-mask_KL) * (1 / temperature)
        # the shape of KL_similarity is (batch_size, training_sample_num)
        exp_logits = torch.exp(logits)

        # Use the KL divergence mask to weight the exponential logits
        #######  add_mask_KL = torch.exp(mask_KL)
        #######  add_mask_KL = torch.exp(mask_KL * torch.log(torch.tensor(10.0)))

        #    self.base.data = torch.clamp(self.base.data, min=2, max=10)
        

        ## here, it is scalar base
        # add_mask_KL = torch.exp(mask_KL * torch.log(torch.tensor(66.0)    )      )







        ##  here, it is trainable base
        add_mask_KL  = torch.exp(mask_KL * torch.log(self.base))
        #   print(torch.isnan(add_mask_KL).any())


        mask4KL = torch.nn.functional.normalize(mask_KL, p=2, dim=1) + 1.0
        # the shape of add_mask_KL is (batch_size, training_sample_num)
        weighted_exp_logits = exp_logits      ####  * mask4KL  #####     * add_mask_KL
        # the shape of weighted_exp_logits is (batch_size, training_sample_num)

        log_prob = logits - torch.log(weighted_exp_logits.sum(1, keepdim=True))
        # the shape of log_prob is (batch_size, training_sample_num)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss1 = - mean_log_prob_pos
        loss1 = loss1.mean()

        logits_eu = F.cosine_similarity(features.unsqueeze(1), k.unsqueeze(0), dim=2) / temperature
        # the shape of logits is (batch_size, training_sample_num)
        exp_logits_eu = torch.exp(logits_eu)
        # the shape of exp_logits is (batch_size, training_sample_num)
        weighted_exp_logits_eu = exp_logits_eu * add_mask_KL     # we remove KL weight for ablation
        log_prob_eu = logits_eu - torch.log(weighted_exp_logits_eu.sum(1, keepdim=True))
        mean_log_prob_pos_eu = (mask * log_prob_eu).sum(1) / mask.sum(1)
        loss2 = - mean_log_prob_pos_eu
        loss2 = loss2.mean()
        
        # we remove KL loss term for ablation study
        loss = 0.05 * loss1 + loss2

        # loss = loss2

        return loss     

    def coarse_loss(self, logits, labels):
        loss_ce = nn.CrossEntropyLoss()(logits, labels)
        return loss_ce


